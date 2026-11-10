#!/usr/bin/env python3
"""Existence-mode run for System 2 (retrieval-based hiring assistant).

Greedy, small N — confirms, with provenance JSON, whether each indirect-injection
cell flips the decision at all and whether the clean baseline holds. Run from root:
    .venv/Scripts/python.exe -m systems.system2_indirect.run_existence
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness import VictimModel, EXISTENCE
from systems.system2_indirect.clip_embedder import ClipEmbedder
from systems.system2_indirect.attack_matrix import run_matrix, print_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

N_EXISTENCE = 5


def main() -> None:
    model = VictimModel()
    embedder = ClipEmbedder()
    results = run_matrix(model, embedder, EXISTENCE, n_trials=N_EXISTENCE)
    print_summary(results, f"System 2 hiring assistant — existence (N={N_EXISTENCE}, defense=off)")


if __name__ == "__main__":
    main()
