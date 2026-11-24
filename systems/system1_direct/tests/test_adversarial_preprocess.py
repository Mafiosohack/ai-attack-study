"""The adversarial attack's gradient path is only valid if our torch
preprocessing is byte-for-byte what the victim's processor produces. CPU-only:
loads the pinned processor, no model weights, no GPU."""

import unittest

import torch
from qwen_vl_utils import process_vision_info

from harness.model import MAX_PIXELS, MIN_PIXELS, MODEL_ID, REVISION
from systems.system1_direct.attacks.adversarial import (
    CANVAS, build_cover_image, build_target_image, grid_thw, pil_to_tensor,
    to_pixel_values,
)


class PreprocessMatchesProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoProcessor
        cls.proc = AutoProcessor.from_pretrained(
            MODEL_ID, revision=REVISION, min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS)

    def _reference(self, img):
        # Exactly the victim's path: qwen_vl_utils resize, then the processor.
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}]}]
        images, _ = process_vision_info(msgs)
        self.assertEqual(images[0].size, CANVAS, "canvas was resized")
        out = self.proc.image_processor(images=images, return_tensors="pt")
        return out["pixel_values"].float(), out["image_grid_thw"]

    def test_cover_and_target_match(self):
        for img in (build_cover_image(), build_target_image()):
            ref, thw = self._reference(img)
            ours = to_pixel_values(pil_to_tensor(img))
            self.assertTrue(torch.equal(thw, grid_thw()))
            self.assertEqual(ours.shape, ref.shape)
            self.assertLess((ours - ref).abs().max().item(), 1e-5)


if __name__ == "__main__":
    unittest.main()
