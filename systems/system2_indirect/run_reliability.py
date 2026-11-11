#!/usr/bin/env python3
"""Reliability-mode run for System 2 (retrieval-based hiring assistant).

Sampled (temp 0.7), large N — the honest ASR distribution. Greedy existence can
read 0 while sampling still flips occasionally; this is the number that goes in the
report. Run from root:
    .venv/Scripts/python.exe -m systems.system2_indirect.run_reliability [--defense schema_extract]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness import VictimModel, RELIABILITY
from systems.system2_indirect.clip_embedder import ClipEmbedder
from systems.system2_indirect.attack_matrix import run_matrix, print_summary
from systems.system2_indirect.hiring_assistant import DEFENSES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

N_RELIABILITY = 50


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--defense", default="off", choices=sorted(DEFENSES))
    ap.add_argument("--n", type=int, default=N_RELIABILITY,
                    help="trials per cell (N=20 resolves the study's large "
                         "contrasts; N=50 tightens the 0%% upper bound)")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="cell names already measured at this defense")
    args = ap.parse_args()
    model = VictimModel()
    embedder = ClipEmbedder()
    results = run_matrix(model, embedder, RELIABILITY, n_trials=args.n,
                         defense=args.defense, skip=frozenset(args.skip))
    print_summary(results, f"System 2 hiring assistant — reliability "
                           f"(N={args.n}, defense={args.defense})")


if __name__ == "__main__":
    main()
