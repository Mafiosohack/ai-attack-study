#!/usr/bin/env python3
"""The ONE shared victim-model loader for the whole series.

Every system (direct, indirect, agentic) and the gradient attack load the model
through THIS class, so model behaviour is byte-for-byte identical everywhere and
ASRs stay comparable across systems. Do not introduce a second inference path
(Ollama/llama.cpp): they can't do gradients and would split behaviour.

Config here is the fixed victim baseline that passed the load gate on an RTX 3050
6GB: Qwen2.5-VL-3B-Instruct, 4-bit NF4 double-quant, fp16 compute, image tokens
capped at the model's native default. The revision is pinned so weights can't
shift mid-series and silently break cross-system ASR comparability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"  # HF main @ 2025-04-06

# Image-token caps. The 3B in 4-bit peaks ~2.6GB at this resolution, well under
# the 6GB ceiling, so we run the model's native default rather than an
# artificially blurred victim — OCR / typographic / pixel-space attacks need the
# detail to be a realistic test. Raised from the smoke test's 256*28*28 cap after
# the load-gate checkpoint showed ~3.7GB of VRAM sitting idle.
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28


@dataclass(frozen=True)
class DecodeConfig:
    """Decode settings for one measurement. The pipeline owns these, not the
    runner — existence mode passes greedy, reliability mode passes a temperature.

    Kept as a frozen dataclass so a config can't be mutated mid-run (which would
    make the ASR unattributable) and so it serializes cleanly into provenance.
    """
    do_sample: bool = False          # False = greedy = the existence-mode floor.
    temperature: float = 1.0         # Only consulted when do_sample is True.
    top_p: float = 1.0
    max_new_tokens: int = 128

    def describe(self) -> dict:
        mode = "reliability(sampled)" if self.do_sample else "existence(greedy)"
        d = {"decode_mode": mode, "max_new_tokens": self.max_new_tokens}
        if self.do_sample:
            d |= {"temperature": self.temperature, "top_p": self.top_p}
        return d


# The two canonical modes, so callers don't hand-roll decode settings.
EXISTENCE = DecodeConfig(do_sample=False)                 # near-deterministic yes/no
RELIABILITY = DecodeConfig(do_sample=True, temperature=0.7)  # ASR as a distribution


class VictimModel:
    """Loads the quantized VLM once and runs image+text inference.

    Load is eager (in __init__) so a broken environment fails at construction,
    not deep inside the first trial where it would be logged as an attack
    'failure' and silently poison the ASR.
    """

    def __init__(self, model_id: str = MODEL_ID, revision: str = REVISION,
                 min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA not available — torch is likely a CPU-only build. "
                "The 4-bit victim requires the cu124 torch. Fix the env first.")

        self.model_id = model_id
        self.revision = revision
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        logger.info("Loading processor %s @ %s ...", model_id, revision[:12])
        self.processor = AutoProcessor.from_pretrained(
            model_id, revision=revision,
            min_pixels=min_pixels, max_pixels=max_pixels,
        )
        logger.info("Loading model in 4-bit NF4 (pulls weights through RAM first)...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, revision=revision,
            quantization_config=quant_config,
            device_map="cuda",
            torch_dtype=torch.float16,
        )
        self.model.eval()
        logger.info("Victim model loaded. VRAM: %.2f GB",
                    torch.cuda.memory_allocated() / (1024 ** 3))

    def generate(self, system_prompt: str, image: Image.Image, user_text: str,
                 decode: DecodeConfig) -> str:
        """Run one system+image+text turn and return the model's decoded reply.

        This is the single choke point every attack passes through, so the
        prompt structure (system / image / user) is identical across all
        systems and only the *content* varies per attack.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ]},
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to("cuda")

        gen_kwargs = {"max_new_tokens": decode.max_new_tokens,
                      "do_sample": decode.do_sample}
        if decode.do_sample:
            gen_kwargs |= {"temperature": decode.temperature, "top_p": decode.top_p}

        with torch.inference_mode():
            generated = self.model.generate(**inputs, **gen_kwargs)

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        answer = self.processor.batch_decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]
        return answer.strip()

    def describe_config(self) -> dict:
        """Provenance snapshot of the fixed model config, recorded with results."""
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "quantization": "4bit-nf4-double-quant-fp16compute",
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
        }
