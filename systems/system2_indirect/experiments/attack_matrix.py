#!/usr/bin/env python3
"""The System 2 attack matrix + a runner that measures every cell with provenance.

Cells = objective {canary, decision-override, decision-falsify} × channel {text,
image} × (for image) perception {plain, faint, tiny}, plus the adaptive
falsification aimed at the fact_check defense (text + plain image). Four
controls come first:
- clean baselines (benign underqualified page, text + image): the rate of
  SPURIOUS shortlists, which must sit at ~0 for a flip to mean anything;
- utility controls (genuinely qualified page, text + image): the rate of
  CORRECT shortlists. A defense that rejects everyone aces every attack cell;
  only these show what it costs.

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
    QUALIFIED_CV, decision_metrics,
    hiring_canary_text_attack, hiring_decision_text_attack, hiring_falsify_text_attack,
    hiring_canary_image_attack, hiring_decision_image_attack, hiring_falsify_image_attack,
    hiring_falsify_adaptive_text_attack, hiring_falsify_adaptive_image_attack,
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
                  metrics=decision_metrics,
                  objective="control:clean_baseline")


def _utility_control(channel: str) -> Attack:
    """A genuinely qualified page judged by the same oracle. Here a 'success'
    is the CORRECT answer: its rate is the defense's utility, and 1 minus it is
    the false-reject rate. Read this column the opposite way to every other."""
    if channel == "text":
        payload: object = QUALIFIED_CV
    else:
        payload = CorpusItem(id="cand:applicant", modality="image",
                             content=build_cv_image(injection=None, style="plain",
                                                    base_text=QUALIFIED_CV),
                             index_text=QUALIFIED_CV, source="corpus")
    return Attack(name=f"s2_utility_qualified_{channel}", payload=payload,
                  succeeded=make_decision_flip_oracle(),
                  metrics=decision_metrics,
                  objective="control:utility")


def build_attacks() -> list[Attack]:
    """The ordered attack matrix. Controls first so the baseline is on record
    before any attack number is read."""
    attacks = [
        _benign_control("text"),
        _benign_control("image"),
        _utility_control("text"),
        _utility_control("image"),
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
        hiring_falsify_adaptive_text_attack(),
        hiring_falsify_adaptive_image_attack("plain"),
    ]
    return attacks


def run_matrix(model, embedder, decode, n_trials: int,
               defense: str = "off", results_dir: str | Path = RESULTS_DIR,
               skip: frozenset[str] = frozenset(),
               only: frozenset[str] = frozenset()) -> list:
    """Measure every cell at one (defense, decode) and return the AttackResults.
    One pipeline is reused across cells — only the payload varies, which is what
    keeps the ASRs on one comparable axis (brief §4). `skip` names cells already
    measured at this (defense, decode) so a resumed sweep doesn't redo them;
    `only`, if given, restricts the sweep to the named cells (e.g. the new
    controls under a defense that was measured before they existed)."""
    names = {a.name for a in build_attacks()}
    unknown = (skip | only) - names
    if unknown:
        raise ValueError(f"unknown cells: {sorted(unknown)}")
    pipe = HiringAssistantPipeline(model, embedder, defense=defense, decode=decode)
    runner = AttackRunner(results_dir=results_dir)
    results = []
    for attack in build_attacks():
        if attack.name in skip or (only and attack.name not in only):
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
        print(f"  {r.attack_name:<40} ASR {r.asr * 100:5.0f}%  "
              f"({r.successes}/{r.n_trials})  cand_rank={rank}")
        m = r.metrics
        if "falsified_all_three" in m:
            print(f"      schema parsed {m['n_parsed']}/{m['n']}  falsified: "
                  f"years {m['falsified_years_exp']}  python {m['falsified_has_python']}  "
                  f"degree {m['falsified_has_degree']}  all3 {m['falsified_all_three']}")
        elif "reason_conflict" in m:
            print(f"      claims parsed {m['n_parsed']}/{m['n']}  reasons: "
                  f"conflict {m['reason_conflict']}  unqualified {m['reason_unqualified']}  "
                  f"qualified {m['reason_qualified']}  unparsed {m['reason_unparsed']}")
            print(f"      unchecked would shortlist {m['unchecked_shortlist']}  caught by "
                  f"conflict {m['caught_by_conflict']} / grounding {m['caught_by_grounding']}  "
                  f"ungrounded claims {m['ungrounded_claims']}")
    print("=" * 72)
