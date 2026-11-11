#!/usr/bin/env python3
"""AttackResult — the serializable outcome of one attack measured over N trials.

Each system's results/ directory IS the provenance record: every claim in the eventual
report must trace to a JSON file written here. That is why victim_config travels
inside each result — a number with no config attached is not evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class AttackResult:
    """Structured outcome of one attack over N trials. ASR is derived, not stored,
    so it can never drift out of sync with successes/n_trials."""
    attack_name: str
    objective: str
    n_trials: int
    successes: int
    outputs: list[str] = field(default_factory=list)
    victim_config: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)  # secondary, attack-defined
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def asr(self) -> float:
        """Attack success rate in [0, 1] — the headline number."""
        return self.successes / self.n_trials if self.n_trials else 0.0

    def save(self, results_dir: str | Path) -> Path:
        """Persist as JSON under results_dir (each system's results/). Filename embeds attack name + UTC
        timestamp so runs never overwrite each other and stay sortable."""
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        stamp = self.timestamp.replace(":", "-")
        path = results_dir / f"{self.attack_name}_{stamp}.json"
        path.write_text(json.dumps(asdict(self) | {"asr": self.asr}, indent=2))
        return path
