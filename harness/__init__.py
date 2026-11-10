"""Measurement harness for the multimodal injection series.

Re-exports the load-bearing pieces so systems import from one place:

    from harness import VictimModel, DecodeConfig, EXISTENCE, RELIABILITY
    from harness import Attack, AttackRunner, AttackResult, VictimPipeline
"""

from harness.model import (
    VictimModel, DecodeConfig, EXISTENCE, RELIABILITY,
    MODEL_ID, REVISION, MIN_PIXELS, MAX_PIXELS,
)
from harness.pipeline import VictimPipeline
from harness.results import AttackResult
from harness.runner import Attack, AttackRunner

__all__ = [
    "VictimModel", "DecodeConfig", "EXISTENCE", "RELIABILITY",
    "MODEL_ID", "REVISION", "MIN_PIXELS", "MAX_PIXELS",
    "VictimPipeline", "AttackResult", "Attack", "AttackRunner",
]
