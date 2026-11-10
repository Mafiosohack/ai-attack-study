#!/usr/bin/env python3
"""The victim-side contract: one attackable pipeline behind one interface.

Each system (direct, indirect, agentic) — and, within System 1, each app variant
(image-description, visual-QA, screenshot-agent) — implements VictimPipeline. That
fixed interface is what lets a single runner produce comparable ASRs across every
victim in the series.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class VictimPipeline(Protocol):
    """One attackable pipeline.

    `run` takes the attack payload the concrete pipeline expects (System 1 an
    image, System 2 a retrieved document, System 3 an agent task) and returns
    the model's final output as a string for the oracle to judge.
    """

    def run(self, attack_payload: object) -> str:
        ...

    def describe_config(self) -> dict:
        """Snapshot of the FIXED victim config for this measurement: model id +
        revision, quantization, decode settings, and which defenses are active.
        Recorded with every result so 'defense off vs on' is an auditable
        one-variable change and every run is reproducible.
        """
        ...
