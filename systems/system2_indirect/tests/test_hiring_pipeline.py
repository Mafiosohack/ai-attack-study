#!/usr/bin/env python3
"""Wiring tests for the hiring-assistant victim pipeline, with fakes for the
heavy models (no CUDA, no CLIP download).

They prove the measurement-relevant plumbing: the candidate's (poisoned) page is
retrieved and actually lands in the model's context, the active defense's system
prompt is the one used, and the config snapshot records the poison's retrieval
rank — the 'did it get fetched' axis that must travel with every result.
"""

from __future__ import annotations

import unittest

import numpy as np

from harness.model import EXISTENCE
from systems.system2_indirect.victim.hiring_assistant import (
    HiringAssistantPipeline, DEFENSES, CANDIDATE_ID,
)


class FakeEmbedder:
    """Maps the candidate page + query to one direction and distractors to
    orthogonal ones, so the candidate page always retrieves top-1."""

    def embed_text(self, text: str) -> np.ndarray:
        if "Jordan Lee" in text or "evaluate the candidate" in text.lower():
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        # Distractors: push them onto other axes (low similarity to the query).
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)

    def embed_image(self, image: object) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


class RecordingModel:
    """Stands in for VictimModel: records the messages it was handed and returns
    a canned verdict so the pipeline's output path is observable."""

    def __init__(self, reply: str = "DECISION: REJECT") -> None:
        self.reply = reply
        self.last_messages = None

    def generate_messages(self, messages, decode) -> str:
        self.last_messages = messages
        return self.reply

    def describe_config(self) -> dict:
        return {"model_id": "fake", "revision": "fake"}


class HiringPipelineWiringTest(unittest.TestCase):
    def _user_blocks(self, messages):
        return next(m["content"] for m in messages if m["role"] == "user")

    def test_candidate_page_is_retrieved_into_context(self) -> None:
        model = RecordingModel()
        pipe = HiringAssistantPipeline(
            model=model, embedder=FakeEmbedder(), defense="off",
            decode=EXISTENCE, k=3)

        pipe.run("Jordan Lee — junior developer, 1 year experience.")

        blocks = self._user_blocks(model.last_messages)
        text_joined = " ".join(b.get("text", "") for b in blocks)
        self.assertIn("junior developer", text_joined)

    def test_off_defense_uses_its_system_prompt(self) -> None:
        model = RecordingModel()
        pipe = HiringAssistantPipeline(
            model=model, embedder=FakeEmbedder(), defense="off",
            decode=EXISTENCE, k=3)

        pipe.run("Jordan Lee — junior developer, 1 year experience.")

        system = next(m["content"] for m in model.last_messages
                      if m["role"] == "system")
        self.assertEqual(system, DEFENSES["off"])

    def test_config_records_poison_retrieval_rank(self) -> None:
        model = RecordingModel()
        pipe = HiringAssistantPipeline(
            model=model, embedder=FakeEmbedder(), defense="off",
            decode=EXISTENCE, k=3)

        pipe.run("Jordan Lee — junior developer, 1 year experience.")
        cfg = pipe.describe_config()

        # The candidate page is the top hit under the fake embedder. (Named for
        # the candidate, not "poison": the same rank is recorded on the clean
        # baseline run, where the page carries no injection.)
        self.assertEqual(cfg["candidate_retrieval_rank"], 1)
        self.assertEqual(cfg["defense"], "off")
        self.assertEqual(cfg["system"], "system2_indirect")


if __name__ == "__main__":
    unittest.main(verbosity=2)
