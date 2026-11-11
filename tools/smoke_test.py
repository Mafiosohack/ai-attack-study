#!/usr/bin/env python3
"""
Tool: Qwen2.5-VL-3B load-and-inference smoke test
Purpose: Confirm the model loads in 4-bit and runs one image inference
         entirely inside 6GB of VRAM BEFORE any harness is built on top.
Usage:   python tools/smoke_test.py tools/sample_images/test_image.png

This is a GATE, not a feature. If it fails or spills into shared GPU
memory, stop and fix the environment — do not build the series on an
unverified foundation.
"""

import sys
import logging

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Pin the exact checkpoint. Fill REVISION with the commit hash from the
# HuggingFace repo (Files > History) so weights can't shift mid-series and
# silently break cross-system ASR comparability. Leaving it None tracks
# the moving main branch — acceptable only for this throwaway smoke test.
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"  # HF main @ 2025-04-06

# Cap image tokens. A VLM turns one image into hundreds-to-thousands of
# tokens; on 6GB, an uncapped high-res image is the most likely cause of
# an out-of-memory error even when the weights fit. The 3B in 4-bit only
# uses ~2.3GB peak at 256*28*28, leaving ~3.7GB idle — so we spend that
# headroom on the model's NATIVE default resolution (1280*28*28 max). This
# matters for System 1: OCR / typographic / pixel-space attacks need fine
# image detail to be a realistic victim, not an artificially blurred one.
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28

VRAM_BUDGET_GB = 6.0  # RTX 3050 6GB Laptop — hard ceiling, must not spill.


def vram_gb() -> float:
    """Currently allocated VRAM in GB (torch's view, not driver total)."""
    return torch.cuda.memory_allocated() / (1024 ** 3)


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python tools/smoke_test.py tools/sample_images/test_image.png")
        sys.exit(1)
    image_path = sys.argv[1]

    # Fail loudly if CUDA isn't actually wired up. A CPU-only torch build
    # is the single most common silent failure — it "works" but ignores
    # the GPU entirely, and 4-bit quantization then errors or crawls.
    if not torch.cuda.is_available():
        logger.error("CUDA not available. PyTorch is likely a CPU-only build. "
                     "Reinstall torch from the cu124 index.")
        sys.exit(1)
    logger.info("CUDA device: %s", torch.cuda.get_device_name(0))

    # 4-bit NF4 with double quantization: the configuration that fits a 3B
    # into ~2GB of weights. fp16 compute because the 3050 (Ampere) has no
    # bf16 advantage here and fp16 is universally supported.
    from transformers import BitsAndBytesConfig
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    logger.info("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, revision=REVISION,
        min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
    )

    logger.info("Loading model in 4-bit (this pulls weights through system RAM "
                "first — close your browser and other apps if it stalls)...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=REVISION,
        quantization_config=quant_config,
        device_map="cuda",
        torch_dtype=torch.float16,
    )
    logger.info("Model loaded. VRAM after load: %.2f GB", vram_gb())

    # One benign inference. If the model can describe an image, the vision
    # encoder, projector, and language head are all wired correctly.
    image = Image.open(image_path).convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this image in one sentence."},
        ],
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to("cuda")

    logger.info("Running inference...")
    # do_sample=False = greedy decoding. Deterministic per exact input on
    # fixed hardware at batch size 1 — the reproducibility floor the whole
    # series' "does this vector work" measurement will stand on.
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=64, do_sample=False)

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    answer = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    logger.info("Model output: %s", answer.strip())
    logger.info("Peak VRAM during run: %.2f GB", peak)

    print("\n" + "=" * 60)
    if peak < VRAM_BUDGET_GB:
        print(f"[+] GATE PASSED — ran inside {peak:.2f} GB (< {VRAM_BUDGET_GB} GB budget).")
        print("    Foundation verified. Now watch Task Manager during a run:")
        print("    Dedicated GPU memory should stay under 6GB and Shared should")
        print("    stay at 0. If Shared climbs, the run is broken, not 'using more'.")
    else:
        print(f"[!] OVER BUDGET — peaked at {peak:.2f} GB (>= {VRAM_BUDGET_GB} GB).")
        print("    Lower MAX_PIXELS first, then reconsider model size. Do NOT")
        print("    proceed to the harness until this runs cleanly.")
    print("=" * 60)


if __name__ == "__main__":
    main()
