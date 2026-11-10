#!/usr/bin/env python3
"""Reliability-mode run for System 1 app #1 (image-description victim).

Fires the typographic canary attack at realistic temperature (RELIABILITY,
temp 0.7), N=50, sweeping the defense axis off -> naive -> spotlight. Existence
mode already confirmed the vector is real and deterministic; this run reports ASR
as a DISTRIBUTION and shows whether the stronger 'spotlight' defense dents an
attack that sits at 100% under greedy decoding.

Run from the repo root:
    .venv/Scripts/python.exe -m systems.system1_direct.run_reliability
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness import VictimModel, AttackRunner, RELIABILITY
from systems.system1_direct.image_description import ImageDescriptionPipeline
from attacks.typographic import typographic_canary_attack

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

N_RELIABILITY = 50  # large: ASR as a distribution under sampling.
DEFENSES = ("off", "naive", "spotlight")


def main() -> None:
    model = VictimModel()
    attack = typographic_canary_attack()
    runner = AttackRunner(results_dir="results")

    results = {}
    for defense in DEFENSES:
        logger.info("=== Typographic canary — defense=%s (reliability N=%d, "
                    "temp 0.7) ===", defense, N_RELIABILITY)
        results[defense] = runner.measure(
            ImageDescriptionPipeline(model, defense=defense, decode=RELIABILITY),
            attack, n_trials=N_RELIABILITY)

    print("\n" + "=" * 64)
    print(f"Typographic canary, RELIABILITY mode (N={N_RELIABILITY}, temp 0.7)")
    print("-" * 64)
    for defense, r in results.items():
        print(f"  defense={defense:<9}: ASR {r.asr*100:5.1f}%  "
              f"({r.successes}/{r.n_trials})")
    print("=" * 64)
    # One non-canary sample per condition, to see what a MISS looks like.
    for defense, r in results.items():
        miss = next((o for o in r.outputs
                     if not attack.succeeded(o)), None)
        print(f"Example non-canary output (defense={defense}): "
              f"{miss[:200] if miss else '<none — every trial emitted the canary>'}")


if __name__ == "__main__":
    main()
