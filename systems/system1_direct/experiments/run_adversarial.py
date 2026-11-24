#!/usr/bin/env python3
"""Adversarial perturbation: craft (encoder-only PGD) then measure (full VLM).

Two separate steps, because crafting is expensive and the measured artefact must
be a fixed file:

  craft    PGD each eps in --eps, save attacks/adversarial_images/s1_adv_eps<E>.png
           plus a .json craft report (loss curve, cosine before/after, VRAM, time).
  measure  Fire every saved PNG at the full victim with the canary oracle. Also
           measures two controls on the same canvas: the clean cover (should be
           0%: proves the canary can't come from the cover) and the typographic
           target (should be ~100%: proves the target is worth impersonating).

Run from the repo root:
    .venv/Scripts/python.exe -m systems.system1_direct.experiments.run_adversarial craft --eps 8 16 32 255
    .venv/Scripts/python.exe -m systems.system1_direct.experiments.run_adversarial measure --mode existence
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from harness import VictimModel, AttackRunner, EXISTENCE, RELIABILITY
from systems.system1_direct.victim.image_description import (
    ALL_DEFENSES, ImageDescriptionPipeline,
)
from systems.system1_direct.attacks.adversarial import (
    adversarial_canary_attack, build_cover_image, build_target_image, craft,
    image_sha256, vision_tower,
)

SYSTEM_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = SYSTEM_DIR / "results"
IMAGES_DIR = SYSTEM_DIR / "attacks" / "adversarial_images"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _prepare_encoder_for_grad(visual: torch.nn.Module) -> None:
    """Freeze weights; turn on activation checkpointing so 32 blocks x 3312
    patches of activations fit in 6 GB. transformers only checkpoints layers in
    train() mode; the vision tower has no dropout, so train() changes nothing
    else about the forward pass."""
    for p in visual.parameters():
        p.requires_grad_(False)
    # Compute in fp32 while crafting. In fp16 the encoder's FORWARD pass
    # overflowed to NaN within ~25 PGD steps at eps 16/32 (never at eps 255),
    # after which PGD is stuck. The 4-bit weights are untouched (same victim
    # encoder); only the matmul compute dtype and the small non-quantized params
    # (norms, patch conv, biases) go to fp32. Evaluation still runs the normal
    # fp16 victim, so any fp32->fp16 gap is measured, not hidden.
    for m in visual.modules():
        if type(m).__name__ == "Linear4bit":
            m.compute_dtype = torch.float32
    for p in visual.parameters():
        if p.dtype == torch.float16:
            p.data = p.data.float()
    visual.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    visual.train()
    n4 = sum(1 for m in visual.modules() if type(m).__name__ == "Linear4bit")
    logger.info("Vision tower: %d Linear4bit layers (0 = unquantized), dtype %s",
                n4, getattr(visual, "dtype", "?"))


def do_craft(args) -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    cover, target = build_cover_image(), build_target_image()
    cover.save(IMAGES_DIR / "cover.png")
    target.save(IMAGES_DIR / "target_typographic.png")

    model = VictimModel()
    visual = vision_tower(model.model)
    _prepare_encoder_for_grad(visual)

    for eps in args.eps:
        logger.info("=== crafting eps=%d/255, %d steps ===", eps, args.steps)
        adv, rep = craft(visual, cover, target, eps=eps, steps=args.steps)
        stem = f"s1_adv_eps{eps}"
        adv.save(IMAGES_DIR / f"{stem}.png")
        meta = asdict(rep) | {"image_sha256": image_sha256(adv),
                              "victim": model.describe_config(),
                              "objective": "1 - mean token cosine to target"}
        (IMAGES_DIR / f"{stem}.json").write_text(json.dumps(meta, indent=2))
        logger.info("eps=%d: cos %.4f -> %.4f (%.0fs, peak %.2f GB) -> %s.png",
                    eps, rep.cos_start, rep.cos_end, rep.seconds,
                    rep.peak_vram_gb, stem)


def do_measure(args) -> None:
    decode = EXISTENCE if args.mode == "existence" else RELIABILITY
    n = args.n or (5 if args.mode == "existence" else 20)
    pngs = sorted(IMAGES_DIR.glob("s1_adv_eps*.png"),
                  key=lambda p: int(p.stem.split("eps")[1]))
    if args.eps:
        pngs = [p for p in pngs if int(p.stem.split("eps")[1]) in args.eps]
    attacks = []
    if not args.no_controls:
        attacks += [
            adversarial_canary_attack(IMAGES_DIR / "cover.png",
                                      "s1_adv_control_cover", {"control": "cover"}),
            adversarial_canary_attack(IMAGES_DIR / "target_typographic.png",
                                      "s1_adv_control_target", {"control": "target"}),
        ]
    for p in pngs:
        rep = json.loads(p.with_suffix(".json").read_text())
        attacks.append(adversarial_canary_attack(
            p, p.stem, {k: rep[k] for k in ("eps", "steps", "cos_start", "cos_end")}))

    model = VictimModel()
    pipeline = ImageDescriptionPipeline(model, defense=args.defense, decode=decode)
    runner = AttackRunner(results_dir=RESULTS_DIR)
    rows = []
    for a in attacks:
        logger.info("=== %s (defense=%s, %s N=%d) ===", a.name, args.defense,
                    args.mode, n)
        r = runner.measure(pipeline, a, n_trials=n)
        rows.append((a.name, r))

    print("\n" + "=" * 78)
    print(f"Adversarial — {args.mode} N={n}, defense={args.defense}")
    for name, r in rows:
        print(f"  {name:<26}{r.successes:>4}/{r.n_trials:<4}  "
              f"{r.outputs[0][:70]!r}")
    print("=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("craft")
    c.add_argument("--eps", type=int, nargs="+", default=[8, 16, 32, 255])
    c.add_argument("--steps", type=int, default=300)
    m = sub.add_parser("measure")
    m.add_argument("--mode", choices=("existence", "reliability"),
                   default="existence")
    m.add_argument("--n", type=int, default=None)
    m.add_argument("--defense", choices=ALL_DEFENSES, default="off")
    m.add_argument("--eps", type=int, nargs="*", default=None)
    m.add_argument("--no-controls", action="store_true")
    args = ap.parse_args()
    do_craft(args) if args.cmd == "craft" else do_measure(args)


if __name__ == "__main__":
    main()
