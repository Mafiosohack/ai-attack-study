#!/usr/bin/env python3
"""CLIP embedder — the real shared-space encoder for System 2 retrieval.

One CLIP model embeds BOTH text and images into a single joint space, which is
what lets a text query retrieve an image (true multimodal RAG). It runs on CPU on
purpose: embedding a tiny corpus + one query per trial is not latency-critical, so
keeping it off the GPU leaves the whole 6GB VRAM ceiling for the quantized Qwen
victim (brief §0 — VRAM is a hard ceiling).

The model id + resolved revision are exposed via describe() so every retrieval is
reproducible provenance (brief §4), exactly like the victim model's config.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ViT-B/32: small (~600MB), standard, and its joint text-image space is more than
# enough to make a semantically-relevant poison out-rank benign items on a small
# corpus. Pinned revision keeps retrieval ranks stable across the series.
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"  # HF main @ pin time


class ClipEmbedder:
    """Embeds text and images into CLIP's shared space. Model loads lazily on
    first use so constructing the object is free and the ~600MB download only
    happens when retrieval actually runs."""

    def __init__(self, model_id: str = CLIP_MODEL_ID,
                 revision: str = CLIP_REVISION) -> None:
        self.model_id = model_id
        self.revision = revision
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # Imported lazily so merely importing this module (e.g. in unit tests that
        # inject a fake embedder) never pulls in transformers' CLIP stack.
        from transformers import CLIPModel, CLIPProcessor
        logger.info("Loading CLIP %s @ %s on CPU ...", self.model_id,
                    self.revision[:12])
        self._model = CLIPModel.from_pretrained(
            self.model_id, revision=self.revision).eval()
        self._processor = CLIPProcessor.from_pretrained(
            self.model_id, revision=self.revision)

    def embed_text(self, text: str) -> np.ndarray:
        self._ensure_loaded()
        inputs = self._processor(text=[text], return_tensors="pt",
                                 padding=True, truncation=True)
        # NB: in this transformers version get_text_features returns the
        # UN-projected text-model output, which is (a) sequence-shaped and (b) not
        # in the joint text-image space. We apply text_projection to the pooled
        # output ourselves so text and image land in the SAME 512-d space (that
        # shared space is what lets a text query retrieve an image). no_grad, not
        # inference_mode: the latter makes the projection Linear raise.
        with torch.no_grad():
            pooled = self._model.text_model(**inputs).pooler_output
            emb = self._model.text_projection(pooled)
        return emb[0].cpu().numpy()

    def embed_image(self, image: Image.Image) -> np.ndarray:
        self._ensure_loaded()
        inputs = self._processor(images=[image], return_tensors="pt")
        with torch.no_grad():
            pooled = self._model.vision_model(**inputs).pooler_output
            emb = self._model.visual_projection(pooled)
        return emb[0].cpu().numpy()

    def describe(self) -> dict:
        """Provenance snapshot, recorded with every System 2 result."""
        return {"embedder_model": self.model_id, "embedder_revision": self.revision}
