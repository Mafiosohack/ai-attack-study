#!/usr/bin/env python3
"""Phase-3 evaluation: every System 1 attack, plus benign controls, per defense.

The hardened rebuild is only judged honestly against the WHOLE attack matrix
(readable text, near-invisible text, small text, pixel perturbation) AND against
benign images it must let through. One defense per invocation keeps the defense
the only variable across the rows of a sweep.

Attack set:
  typographic control (40 px black)               — must be stopped
  perceptual cells (--perceptual, by name suffix)  — the hard-to-see variants
  every crafted adversarial PNG                    — the no-text variant
  benign controls (cafe sign, recipe, notice, cover) — success = FALSE BLOCK

Run from the repo root:
    .venv/Scripts/python.exe -m systems.system1_direct.experiments.run_defenses --defense ocr_filter
    .venv/Scripts/python.exe -m systems.system1_direct.experiments.run_defenses --defense hardened --mode reliability --n 20
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from harness import VictimModel, AttackRunner, EXISTENCE, RELIABILITY
from systems.system1_direct.victim.image_description import (
    ALL_DEFENSES, ImageDescriptionPipeline,
)
from systems.system1_direct.attacks.typographic import typographic_canary_attack
from systems.system1_direct.attacks.perceptual import (
    LADDER, perceptual_canary_attack,
)
from systems.system1_direct.attacks.adversarial import adversarial_canary_attack
from systems.system1_direct.attacks.benign_controls import benign_controls

SYSTEM_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = SYSTEM_DIR / "results"
IMAGES_DIR = SYSTEM_DIR / "attacks" / "adversarial_images"

DEFAULT_PERCEPTUAL = ("40px_g252", "12px_g0", "26px_g200")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_attacks(perceptual: list[str], adv_eps: list[int] | None,
                  skip_benign: bool):
    attacks = [typographic_canary_attack()]
    attacks += [perceptual_canary_attack(v) for v in LADDER
                if any(v.name.endswith(s) for s in perceptual)]
    for p in sorted(IMAGES_DIR.glob("s1_adv_eps*.png")):
        eps = int(p.stem.split("eps")[1])
        if adv_eps is not None and eps not in adv_eps:
            continue
        rep = json.loads(p.with_suffix(".json").read_text())
        attacks.append(adversarial_canary_attack(
            p, p.stem, {k: rep[k] for k in ("eps", "steps", "cos_end")}))
    if not skip_benign:
        attacks += benign_controls()
    return attacks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--defense", choices=ALL_DEFENSES, required=True)
    ap.add_argument("--mode", choices=("existence", "reliability"),
                    default="existence")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--perceptual", nargs="*", default=list(DEFAULT_PERCEPTUAL))
    ap.add_argument("--adv-eps", type=int, nargs="*", default=None,
                    help="only these adversarial budgets (default: all crafted)")
    ap.add_argument("--skip-benign", action="store_true")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="parallel sampled trials per GPU call (reliability only; "
                         "1 = sequential, the original path)")
    ap.add_argument("--benign-n", type=int, default=None,
                    help="trials for benign controls (default: same as --n). "
                         "Their oracle is 'was it blocked', decided by the greedy, "
                         "cached reader pass, so it is deterministic per image "
                         "and N=1 is exact.")
    args = ap.parse_args()

    decode = EXISTENCE if args.mode == "existence" else RELIABILITY
    n = args.n or (5 if args.mode == "existence" else 20)
    attacks = build_attacks(args.perceptual, args.adv_eps, args.skip_benign)

    model = VictimModel()
    runner = AttackRunner(results_dir=RESULTS_DIR)
    rows = []
    for a in attacks:
        # Fresh pipeline per attack: last_reader in the saved config then
        # belongs unambiguously to THIS attack's image.
        pipeline = ImageDescriptionPipeline(model, defense=args.defense,
                                            decode=decode)
        is_control = a.objective.startswith("control")
        n_a = (args.benign_n or n) if is_control else n
        logger.info("=== %s (defense=%s, %s N=%d) ===", a.name, args.defense,
                    args.mode, n_a)
        r = runner.measure(pipeline, a, n_trials=n_a,
                           batch_size=args.batch_size if decode.do_sample else 1)
        rows.append((a, r, pipeline.last_reader))

    print("\n" + "=" * 90)
    print(f"Defense sweep — defense={args.defense}, {args.mode} N={n}")
    print("  (attacks: rate = ASR; controls: rate = FALSE-BLOCK rate)")
    print("-" * 90)
    for a, r, reader in rows:
        kind = "control" if a.objective.startswith("control") else "attack"
        why = (f"blocked by {reader['patterns_hit']}" if reader and reader["blocked"]
               else (f"reader saw: {reader['transcript'][:40]!r}" if reader else ""))
        print(f"  {a.name:<32}{kind:<8}{r.successes:>4}/{r.n_trials:<4} {why}")
    print("=" * 90)


if __name__ == "__main__":
    main()
