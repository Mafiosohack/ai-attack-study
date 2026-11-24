#!/usr/bin/env python3
"""Perceptual-threshold sweep: the control-case injection at falling visibility.

Walks the ladder in attacks/perceptual.py (contrast axis + size axis + System 2's
faint point), one Attack per cell, same victim, same oracle. Only visibility
varies within a sweep; the defense is fixed per invocation.

Run from the repo root:
    .venv/Scripts/python.exe -m systems.system1_direct.experiments.run_perceptual --mode existence
    .venv/Scripts/python.exe -m systems.system1_direct.experiments.run_perceptual --mode reliability --n 20
    ... --cells 40px_g240 8px_g0     (subset, by name suffix)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from harness import VictimModel, AttackRunner, EXISTENCE, RELIABILITY
from systems.system1_direct.victim.image_description import (
    ALL_DEFENSES, ImageDescriptionPipeline,
)
from systems.system1_direct.attacks.perceptual import (
    LADDER, contrast_ratio, perceptual_canary_attack,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("existence", "reliability"),
                    default="existence")
    ap.add_argument("--n", type=int, default=None,
                    help="trials per cell (default 5 existence / 20 reliability)")
    ap.add_argument("--defense", choices=ALL_DEFENSES, default="off")
    ap.add_argument("--cells", nargs="*", default=None,
                    help="only run cells whose name ends with one of these")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="parallel sampled trials per GPU call (reliability only)")
    args = ap.parse_args()

    decode = EXISTENCE if args.mode == "existence" else RELIABILITY
    n = args.n or (5 if args.mode == "existence" else 20)
    cells = [v for v in LADDER
             if not args.cells or any(v.name.endswith(c) for c in args.cells)]
    if not cells:
        sys.exit(f"no ladder cells match {args.cells}")

    model = VictimModel()
    pipeline = ImageDescriptionPipeline(model, defense=args.defense, decode=decode)
    runner = AttackRunner(results_dir=RESULTS_DIR)

    rows = []
    for vis in cells:
        logger.info("=== %s (%s, defense=%s, %s N=%d) ===", vis.name, vis.label,
                    args.defense, args.mode, n)
        attack = perceptual_canary_attack(vis)
        r = runner.measure(pipeline, attack, n_trials=n,
                           batch_size=args.batch_size if decode.do_sample else 1)
        miss = next((o for o in r.outputs if not attack.succeeded(o)), None)
        rows.append((vis, r, miss))

    print("\n" + "=" * 78)
    print(f"Perceptual ladder — {args.mode} N={n}, defense={args.defense}")
    print("-" * 78)
    print(f"{'cell':<26}{'axis':<10}{'WCAG':>7}{'ASR':>9}   example miss")
    for vis, r, miss in rows:
        print(f"{vis.name:<26}{vis.label:<10}{contrast_ratio(vis.grey):>7.2f}"
              f"{r.successes:>4}/{r.n_trials:<4}   "
              f"{(miss or '-')[:60]!r}")
    print("=" * 78)


if __name__ == "__main__":
    main()
