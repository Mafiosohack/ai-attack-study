#!/usr/bin/env python3
"""Interactive REPL against the local victim model — for poking at it by hand.

NOT part of the study protocol. Measurements still go through the harness
(`harness/model.py`), which owns the fixed baseline config; this script imports
that same loader so what you see here is the same weights/quantization the
attacks run against, but it keeps multi-turn history and streams tokens, which
the harness's single-turn choke point deliberately does not.

Usage (from the project root, venv python):
    ./.venv/Scripts/python.exe tools/victim_chat/chat.py
    ./.venv/Scripts/python.exe tools/victim_chat/chat.py --image tools/sample_images/test_image.png

In-chat commands:
    /image <path>   attach an image to your NEXT message
    /system <text>  replace the system prompt and reset history
    /reset          clear the conversation
    /exit           quit
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from threading import Thread

import torch
from PIL import Image
from transformers import TextIteratorStreamer
from qwen_vl_utils import process_vision_info

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from harness.model import VictimModel  # noqa: E402

DEFAULT_SYSTEM = "You are a helpful assistant."


def build_inputs(vm: VictimModel, messages: list[dict]):
    text = vm.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    return vm.processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to("cuda")


def stream_reply(vm: VictimModel, messages: list[dict], args) -> str:
    """Generate on a worker thread so tokens can be printed as they arrive."""
    inputs = build_inputs(vm, messages)
    streamer = TextIteratorStreamer(
        vm.processor.tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = dict(**inputs, streamer=streamer,
                      max_new_tokens=args.max_new_tokens,
                      do_sample=args.temperature > 0)
    if gen_kwargs["do_sample"]:
        gen_kwargs |= {"temperature": args.temperature, "top_p": args.top_p}

    def _run():
        with torch.inference_mode():
            vm.model.generate(**gen_kwargs)

    thread = Thread(target=_run, daemon=True)
    thread.start()

    chunks = []
    for chunk in streamer:
        chunks.append(chunk)
        print(chunk, end="", flush=True)
    thread.join()
    print()
    return "".join(chunks).strip()


def load_image(path_str: str) -> Image.Image | None:
    path = Path(path_str.strip().strip('"').strip("'"))
    if not path.exists():
        print(f"  [no such file: {path}]")
        return None
    img = Image.open(path).convert("RGB")
    print(f"  [attached {path.name} ({img.width}x{img.height}) to your next message]")
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description="Chat with the local Qwen2.5-VL victim model.")
    ap.add_argument("--image", help="image attached to the first message")
    ap.add_argument("--system", default=DEFAULT_SYSTEM, help="system prompt")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="0 = greedy/deterministic (default 0.7)")
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("Loading the victim model (4-bit NF4, ~30s cold)...")
    vm = VictimModel()
    print(f"Ready. VRAM {torch.cuda.memory_allocated() / 1024**3:.2f} GB. "
          f"temp={args.temperature} max_new_tokens={args.max_new_tokens}")
    print("Commands: /image <path>  /system <text>  /reset  /exit\n")

    system_prompt = args.system
    history: list[dict] = []
    pending_image = load_image(args.image) if args.image else None

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not user:
            continue

        if user in ("/exit", "/quit"):
            return 0
        if user == "/reset":
            history, pending_image = [], None
            print("  [history cleared]")
            continue
        if user.startswith("/system "):
            system_prompt = user[len("/system "):]
            history, pending_image = [], None
            print("  [system prompt set, history cleared]")
            continue
        if user.startswith("/image "):
            pending_image = load_image(user[len("/image "):])
            continue
        if user.startswith("/"):
            print("  [unknown command]")
            continue

        content: list[dict] = []
        if pending_image is not None:
            content.append({"type": "image", "image": pending_image})
            pending_image = None
        content.append({"type": "text", "text": user})
        history.append({"role": "user", "content": content})

        messages = [{"role": "system", "content": system_prompt}] + history
        print("model> ", end="", flush=True)
        try:
            reply = stream_reply(vm, messages, args)
        except KeyboardInterrupt:
            print("\n  [interrupted — dropping that turn]")
            history.pop()
            continue
        history.append({"role": "assistant", "content": reply})

    return 0


if __name__ == "__main__":
    sys.exit(main())
