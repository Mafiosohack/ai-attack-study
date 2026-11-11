#!/usr/bin/env python3
"""Attack + AttackRunner: fire an attack N times and compute ASR.

An ATTACK is a payload PLUS a success oracle. Success means different things for
different attacks (canary emission, a tool action, a bypassed refusal), so the
oracle travels with the attack and the runner stays agnostic to attack semantics.

ASR over N trials is the metric — not binary pass/fail. It is the honest way to
report a stochastic target and what makes 'cost raised, not immune' visible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from harness.pipeline import VictimPipeline
from harness.results import AttackResult

logger = logging.getLogger(__name__)


@dataclass
class Attack:
    """A payload and the oracle that decides whether it worked.

    `succeeded` returns True iff this attack achieved its objective for a given
    model output. Keeping the oracle beside the payload prevents the runner from
    needing to know anything about attack semantics.
    """
    name: str
    payload: object
    succeeded: Callable[[str], bool]
    objective: str = ""  # "exfiltration" | "jailbreak" | ... (for the report)
    # Optional secondary measurements over all N outputs (e.g. per-field
    # falsification counts), saved beside ASR. Never affects `successes`.
    metrics: Optional[Callable[[list[str]], dict]] = None


class AttackRunner:
    """Fires an attack against a victim N times and computes ASR.

    The runner is deliberately mode-agnostic: whether a run is existence
    (greedy, small N) or reliability (sampled, large N) is decided entirely by
    how the pipeline is configured. The runner just counts.
    """

    def __init__(self, results_dir: str | Path = "results") -> None:
        self.results_dir = Path(results_dir)

    def measure(self, pipeline: VictimPipeline, attack: Attack,
                n_trials: int) -> AttackResult:
        if n_trials < 1:
            raise ValueError("n_trials must be >= 1")

        successes = 0
        outputs: list[str] = []
        for i in range(n_trials):
            try:
                output = pipeline.run(attack.payload)
            except Exception:
                # A crashed trial is a failed trial, not a crashed experiment.
                # Log it and keep going so one bad run doesn't void the ASR.
                logger.exception("Trial %d/%d raised; counting as failure.",
                                 i + 1, n_trials)
                output = ""
            outputs.append(output)
            if attack.succeeded(output):
                successes += 1
            logger.info("  trial %d/%d: %s", i + 1, n_trials,
                        "HIT" if attack.succeeded(output) else "miss")

        result = AttackResult(
            attack_name=attack.name,
            objective=attack.objective,
            n_trials=n_trials,
            successes=successes,
            outputs=outputs,
            victim_config=pipeline.describe_config(),
            metrics=attack.metrics(outputs) if attack.metrics else {},
        )
        saved = result.save(self.results_dir)
        logger.info("%s: ASR %.0f%% (%d/%d) -> %s",
                    attack.name, result.asr * 100, successes, n_trials, saved.name)
        return result
