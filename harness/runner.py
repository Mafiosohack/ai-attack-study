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
                n_trials: int, batch_size: int = 1) -> AttackResult:
        """Fire `attack` n_trials times and record the result.

        batch_size > 1 runs trials in parallel chunks through the pipeline's
        optional `run_batch(payload, n)` (sampled decoding only). Each trial is
        still an independent sample; only the execution path differs, and that
        path is recorded in victim_config["trial_execution"] so batched and
        sequential results are never silently mixed.
        """
        if n_trials < 1:
            raise ValueError("n_trials must be >= 1")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if batch_size > 1 and not hasattr(pipeline, "run_batch"):
            raise TypeError(f"{type(pipeline).__name__} has no run_batch(); "
                            "use batch_size=1")

        outputs: list[str] = []
        if batch_size == 1:
            for i in range(n_trials):
                try:
                    output = pipeline.run(attack.payload)
                except Exception:
                    # A crashed trial is a failed trial, not a crashed
                    # experiment. Log it and keep going so one bad run doesn't
                    # void the ASR.
                    logger.exception("Trial %d/%d raised; counting as failure.",
                                     i + 1, n_trials)
                    output = ""
                outputs.append(output)
                logger.info("  trial %d/%d: %s", i + 1, n_trials,
                            "HIT" if attack.succeeded(output) else "miss")
            execution = {"mode": "sequential"}
        else:
            while len(outputs) < n_trials:
                k = min(batch_size, n_trials - len(outputs))
                try:
                    chunk = list(pipeline.run_batch(attack.payload, k))
                    if len(chunk) != k:
                        raise RuntimeError(f"run_batch returned {len(chunk)}, "
                                           f"expected {k}")
                except Exception:
                    # Same rule as sequential: crashed trials count as failures.
                    logger.exception("Batch of %d raised; counting %d failures.",
                                     k, k)
                    chunk = [""] * k
                outputs.extend(chunk)
                hits = sum(attack.succeeded(o) for o in chunk)
                logger.info("  trials %d-%d/%d: %d HIT, %d miss",
                            len(outputs) - k + 1, len(outputs), n_trials,
                            hits, k - hits)
            execution = {"mode": "batched", "batch_size": batch_size,
                         "method": "num_return_sequences"}
        successes = sum(1 for o in outputs if attack.succeeded(o))

        result = AttackResult(
            attack_name=attack.name,
            objective=attack.objective,
            n_trials=n_trials,
            successes=successes,
            outputs=outputs,
            victim_config=pipeline.describe_config() | {
                "trial_execution": execution},
            metrics=attack.metrics(outputs) if attack.metrics else {},
        )
        saved = result.save(self.results_dir)
        logger.info("%s: ASR %.0f%% (%d/%d) -> %s",
                    attack.name, result.asr * 100, successes, n_trials, saved.name)
        return result
