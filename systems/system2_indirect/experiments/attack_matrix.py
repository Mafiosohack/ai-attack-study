#!/usr/bin/env python3
"""The System 2 attack matrix + a runner that measures every cell with provenance.

Cells = objective {canary, decision-override, decision-falsify} × channel {text,
image} × (for image) perception {plain, faint, tiny}, all at defense "off" for
Phase 1. Two clean-baseline controls (benign text + benign image, judged by the
decision-flip oracle) confirm the underqualified candidate is NOT spuriously
shortlisted — without that, a measured flip would be meaningless.

Each cell is one AttackResult JSON in this system's results/ (full provenance, incl. the
candidate's retrieval rank). Run via run_existence / run_reliability.
"""

from __future__ import annotations

import logging
from pathlib import Path

from harness import AttackRunner
from harness.runner import Attack
from systems.system2_indirect.victim.hiring_assistant import HiringAssistantPipeline
from systems.system2_indirect.attacks.hiring import (
    benign_candidate_page, build_cv_image, make_decision_flip_oracle, BENIGN_CV,
    schema_field_metrics,
    hiring_canary_text_attack, hiring_decision_text_attack, hiring_falsify_text_attack,
    hiring_canary_image_attack, hiring_decision_image_attack, hiring_falsify_image_attack,
)
from systems.system2_indirect.victim.retrieval import CorpusItem

logger = logging.getLogger(__name__)

# Every System 2 run lands in this system's own results/ folder.
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _benign_control(channel: str) -> Attack:
    """A clean (uninjected) page judged by the decision-flip oracle: its 'ASR' is
    the rate of SPURIOUS shortlists on an underqualified candidate — the control
    that must sit at ~0 for the attack numbers to mean anything."""
    if channel == "text":
        payload: object = benign_candidate_page()
    else:
        payload = CorpusItem(id="cand:applicant", modality="image",
                             content=build_cv_image(injection=None, style="plain"),
                             index_text=BENIGN_CV, source="corpus")
    return Attack(name=f"s2_baseline_clean_{channel}", payload=payload,
                  succeeded=make_decision_flip_oracle(),
                  metrics=schema_field_metrics,
                  objective="control:clean_baseline")


def build_attacks() -> list[Attack]:
    """The ordered attack matrix. Controls first so the baseline is on record
    before any attack number is read."""
    attacks = [
        _benign_control("text"),
        _benign_control("image"),
        hiring_canary_text_attack(),
        hiring_decision_text_attack(),
        hiring_falsify_text_attack(),
        hiring_canary_image_attack("plain"),
        hiring_decision_image_attack("plain"),
        hiring_falsify_image_attack("plain"),
        hiring_decision_image_attack("faint"),
        hiring_falsify_image_attack("faint"),
        hiring_decision_image_attack("tiny"),
        hiring_falsify_image_attack("tiny"),
    ]
    return attacks


def run_matrix(model, embedder, decode, n_trials: int,
               defense: str = "off", results_dir: str | Path = RESULTS_DIR,
               skip: frozenset[str] = frozenset()) -> list:
    """Measure every cell at one (defense, decode) and return the AttackResults.
    One pipeline is reused across cells — only the payload varies, which is what
    keeps the ASRs on one comparable axis (brief §4). `skip` names cells already
    measured at this (defense, decode) so a resumed sweep doesn't redo them."""
    unknown = skip - {a.name for a in build_attacks()}
    if unknown:
        raise ValueError(f"unknown cells in skip: {sorted(unknown)}")
    pipe = HiringAssistantPipeline(model, embedder, defense=defense, decode=decode)
    runner = AttackRunner(results_dir=results_dir)
    results = []
    for attack in build_attacks():
        if attack.name in skip:
            logger.info("--- %s skipped (already measured)", attack.name)
            continue
        logger.info("=== %s (defense=%s, N=%d) ===", attack.name, defense, n_trials)
        results.append(runner.measure(pipe, attack, n_trials))
    return results


def print_summary(results: list, title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("-" * 72)
    for r in results:
        rank = r.victim_config.get("candidate_retrieval_rank")
        print(f"  {r.attack_name:<34} ASR {r.asr * 100:5.0f}%  "
              f"({r.successes}/{r.n_trials})  cand_rank={rank}")
        if r.metrics:
            m = r.metrics
            print(f"      schema parsed {m['n_parsed']}/{m['n']}  falsified: "
                  f"years {m['falsified_years_exp']}  python {m['falsified_has_python']}  "
                  f"degree {m['falsified_has_degree']}  all3 {m['falsified_all_three']}")
    print("=" * 72)
