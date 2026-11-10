#!/usr/bin/env python3
"""Characterization tests for the victim model's message assembly.

These lock the message STRUCTURE so the System-2 refactor (adding a general
multi-block path) cannot silently change what System 1 sends to the model — if it
did, System 1 vs System 2 ASRs would stop being comparable, which is the whole
point of the shared harness (brief §3). They are pure: no CUDA, no model load.
"""

from __future__ import annotations

import unittest

from harness.model import build_single_image_messages, build_context_messages


class SingleImageMessageTest(unittest.TestCase):
    def test_single_image_message_matches_system1_structure(self) -> None:
        # This is exactly the structure System 1 has always sent: a system turn,
        # then a user turn whose content is [image block, text block].
        image = object()  # stand-in; structure is what matters, not the pixels
        messages = build_single_image_messages(
            system_prompt="SYS", image=image, user_text="Describe this image.")

        self.assertEqual(messages, [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image."},
            ]},
        ])


class ContextMessageTest(unittest.TestCase):
    def test_context_message_interleaves_retrieved_blocks_then_question(self) -> None:
        # System 2 stuffs retrieved items (mixed text + image) into the user turn,
        # followed by the user's benign question. Order is preserved so retrieval
        # rank is reflected in context order.
        img = object()
        blocks = [
            {"type": "text", "text": "doc one"},
            {"type": "image", "image": img},
        ]
        messages = build_context_messages(
            system_prompt="SYS", context_blocks=blocks, user_text="Who is X?")

        self.assertEqual(messages, [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": [
                {"type": "text", "text": "doc one"},
                {"type": "image", "image": img},
                {"type": "text", "text": "Who is X?"},
            ]},
        ])


if __name__ == "__main__":
    unittest.main(verbosity=2)
