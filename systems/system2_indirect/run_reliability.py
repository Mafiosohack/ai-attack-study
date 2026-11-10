#!/usr/bin/env python3
"""Reliability-mode run for System 2 (retrieval-based hiring assistant).

Sampled (temp 0.7), large N — the honest ASR distribution. Greedy existence can
read 0 while sampling still flips occasionally; this is the number that goes in the
report. Run from root:
    .venv/Scripts/python.exe -m systems.system2_indirect.run_reliability
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness import VictimModel, RELIABILITY
from systems.system2_indirect.clip_embedder import ClipEmbedder
from systems.system2_indirect.attack_matrix import run_matrix, print_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

N_RELIABILITY = 50


def main() -> None:
    model = VictimModel()
    embedder = ClipEmbedder()
    results = run_matrix(model, embedder, RELIABILITY, n_trials=N_RELIABILITY)
    print_summary(results, f"System 2 hiring assistant — reliability (N={N_RELIABILITY}, defense=off)")


if __name__ == "__main__":
    main()
