#!/usr/bin/env python3
"""Existence-mode run for System 1 app #1 (image-description victim).

Fires the typographic canary attack at the undefended victim, then the defended
one, both in existence mode (greedy, small N) — just enough to confirm the vector
works and is stable before committing to long reliability runs. This is the run
that reaches the §7 checkpoint: does the VictimPipeline abstraction feel natural?

Run from the repo root:
    .venv/Scripts/python.exe -m systems.system1_direct.experiments.run_existence
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Allow running as a plain script (python systems/system1_direct/run_existence.py)
# by putting the repo root on sys.path, not just `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from harness import VictimModel, AttackRunner, EXISTENCE
from systems.system1_direct.victim.image_description import ImageDescriptionPipeline
from systems.system1_direct.attacks.typographic import typographic_canary_attack

# Every run of this system lands in its own results/ folder, wherever it is run from.
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

N_EXISTENCE = 5  # small: confirm the vector is real and stable, not measure ASR.


def main() -> None:
    model = VictimModel()                 # loads the fixed 4-bit victim once
    attack = typographic_canary_attack()  # payload + canary oracle
    runner = AttackRunner(results_dir=RESULTS_DIR)

    results = {}
    for defense in ("off", "naive", "spotlight"):
        logger.info("=== Typographic canary — defense=%s (existence N=%d) ===",
                    defense, N_EXISTENCE)
        results[defense] = runner.measure(
            ImageDescriptionPipeline(model, defense=defense, decode=EXISTENCE),
            attack, n_trials=N_EXISTENCE)

    print("\n" + "=" * 60)
    print(f"Typographic canary control-case, existence mode (N={N_EXISTENCE})")
    for defense, r in results.items():
        print(f"  defense={defense:<9}: ASR {r.asr*100:5.0f}%  "
              f"({r.successes}/{r.n_trials})")
    print("=" * 60)
    for defense, r in results.items():
        print(f"Sample output (defense={defense}):")
        print("  " + (r.outputs[0][:300] if r.outputs else "<none>"))


if __name__ == "__main__":
    main()
