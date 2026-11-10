#!/usr/bin/env python3
"""
Harness core — the measurement instrument for the multimodal injection series.

Design rationale (do not casually refactor these two ideas apart):

1. A VICTIM is a pipeline you attack from the outside. Each system (direct,
   indirect, agentic) is a different pipeline behind the SAME interface, so
   the runner measures all three identically and their ASRs stay comparable.

2. An ATTACK is a payload PLUS a success oracle. Success is judged
   per-attack (exfil looks for a leaked secret; jailbreak looks for
   prohibited content), so the oracle travels with the attack, not the runner.

The runner fires each attack N times and reports Attack Success Rate. ASR is
not a workaround for nondeterminism — it is the honest metric for attack
reliability, and it is what makes "cost raised, not immune" visible in numbers.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class VictimPipeline(Protocol):
    """One attackable pipeline. System 1/2/3 each implement this.

    Holding this interface fixed is what lets a single runner produce
    comparable ASRs across every system in the series.
    """

    def run(self, attack_payload: object) -> str:
        """Feed the attack into the pipeline, return the model's final output.

        `attack_payload` is deliberately untyped here: System 1 passes an
        image, System 2 a retrieved document, System 3 an agent task. The
        concrete pipeline knows what it expects.
        """
        ...

    def describe_config(self) -> dict:
        """Snapshot of the FIXED victim config for this measurement:
        model id + revision, quantization, decode settings, which defenses
        are active. Recorded with every result so a run is reproducible and
        so 'defense off vs on' is an auditable one-variable change.
        """
        ...


@dataclass
class Attack:
    """A payload and the oracle that decides whether it worked.

    Keeping the oracle beside the payload prevents the runner from having to
    know anything about attack semantics — it just calls succeeded(output).
    """
    name: str
    payload: object
    # Returns True if this specific attack achieved its objective for a given
    # model output. Example: lambda out: "SECRET_FLAG_9f2a" in out
    succeeded: Callable[[str], bool]
    objective: str = ""  # "exfiltration" | "jailbreak" | ... (for the report)


@dataclass
class AttackResult:
    """Structured, serializable outcome of one attack measured over N trials."""
    attack_name: str
    objective: str
    n_trials: int
    successes: int
    outputs: list[str] = field(default_factory=list)
    victim_config: dict = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def asr(self) -> float:
        """Attack success rate in [0, 1]. The headline number."""
        return self.successes / self.n_trials if self.n_trials else 0.0

    def save(self, results_dir: str | Path) -> Path:
        """Persist as JSON. The results/ dir IS the provenance record — every
        claim in the eventual report should trace to a file here."""
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        stamp = self.timestamp.replace(":", "-")
        path = results_dir / f"{self.attack_name}_{stamp}.json"
        path.write_text(json.dumps(asdict(self) | {"asr": self.asr}, indent=2))
        return path


class AttackRunner:
    """Fires an attack against a victim N times and computes ASR.

    Two intended measurement modes, chosen by how the victim pipeline is
    configured (the runner stays agnostic; the pipeline owns decode settings):

      - Existence:  temp 0 / greedy / batch 1  -> near-deterministic yes/no.
                    Use N small (e.g. 5) just to confirm stability.
      - Reliability: realistic temp (e.g. 0.7) -> distribution.
                    Use N large (e.g. 50) so the ASR is meaningful.
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

        result = AttackResult(
            attack_name=attack.name,
            objective=attack.objective,
            n_trials=n_trials,
            successes=successes,
            outputs=outputs,
            victim_config=pipeline.describe_config(),
        )
        saved = result.save(self.results_dir)
        logger.info("%s: ASR %.0f%% (%d/%d) -> %s",
                    attack.name, result.asr * 100, successes, n_trials, saved.name)
        return result


# Sketch of how a concrete system will plug in (System 1 will flesh this out):
#
#   class DirectUploadPipeline:
#       def __init__(self, model, defenses_on: bool): ...
#       def run(self, attack_payload):            # attack_payload = a crafted image
#           return self.model.infer(SYSTEM_PROMPT, attack_payload)
#       def describe_config(self):
#           return {"system": "direct", "defenses_on": self.defenses_on, ...}
#
#   runner = AttackRunner()
#   runner.measure(DirectUploadPipeline(model, defenses_on=False),
#                  typographic_attack, n_trials=50)
