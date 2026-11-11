#!/usr/bin/env python3
"""Semantic retriever for System 2 (indirect injection via retrieval).

A small FIXED mixed-modality corpus (text snippets + images) is embedded into one
shared space by a CLIP-style embedder; a benign query retrieves the top-k nearest
items, which are then stuffed into the victim VLM's context. One corpus item is
the attacker's POISON — authored to rank for a plausible benign query, so it is
retrieved and ingested without the user ever seeing it.

Design choices (brief-driven):
- Real SEMANTIC retrieval (cosine over CLIP embeddings), not keyword: the indirect
  threat IS that a meaning-relevant-looking poison gets pulled in.
- Brute-force exact cosine over a tiny corpus — no FAISS/Chroma (scale we do not
  have), no extra dependency, fully deterministic.
- For a fixed query + fixed corpus the ranking is identical every trial, so
  retrieval contributes NO per-trial variance to ASR (brief §4). The two stages —
  "does the poison get retrieved?" and "once retrieved, does it inject?" — are
  measured as separate axes, never conflated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class CorpusItem:
    """One retrievable item.

    `content` is what enters the model's context — a str for text items, a
    PIL.Image for image items. `index_text` is what RETRIEVAL embeds: for text it
    defaults to the content; for an image it is the OCR'd / transcribed visible
    text (document-RAG style). Indexing images by text — rather than a cross-modal
    image embedding — sidesteps CLIP's modality gap and its inability to read words
    inside a picture, while the real image still rides into context so the VLM
    reads any faint injection from pixels. `is_poison`/`source` are provenance the
    pipeline records and the phase-3 defenses key off of."""
    id: str
    modality: str           # "text" | "image"
    content: object
    index_text: str = ""
    is_poison: bool = False
    source: str = ""        # provenance label, e.g. "corpus" | "attacker-web"

    def __post_init__(self) -> None:
        if self.index_text:
            return
        if self.modality == "text" and isinstance(self.content, str):
            self.index_text = self.content
        else:
            raise ValueError(
                f"item {self.id!r} ({self.modality}) needs an explicit index_text "
                "(e.g. the image's OCR'd text) — only text items default it to "
                "their content.")


@runtime_checkable
class Embedder(Protocol):
    """Embeds text and images into ONE shared vector space (so a text query can
    retrieve an image). CLIP is the real implementation; tests inject a fake."""

    def embed_text(self, text: str) -> np.ndarray: ...
    def embed_image(self, image: object) -> np.ndarray: ...


def _unit(v: np.ndarray) -> np.ndarray:
    """L2-normalize so a dot product IS cosine similarity. A zero vector is left
    as-is (its similarity to everything is 0), which keeps ranking well-defined."""
    v = np.asarray(v, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0.0 else v


class Retriever:
    """Ranks a fixed corpus against a query by exact cosine similarity.

    Item embeddings are precomputed once at construction (the corpus is fixed), so
    each `retrieve` call only embeds the query and does one matrix-vector product.
    """

    def __init__(self, corpus: list[CorpusItem], embedder: Embedder) -> None:
        self.corpus = list(corpus)
        self.embedder = embedder
        self._matrix = np.vstack([self._embed_item(it) for it in self.corpus]) \
            if self.corpus else np.empty((0, 0), dtype=np.float32)

    def _embed_item(self, item: CorpusItem) -> np.ndarray:
        # Everything is ranked in the TEXT space via its index_text — images
        # included (document-RAG). So a text query and every item live in one
        # space with no modality gap. The image's pixels are used only later, in
        # context; never for ranking.
        return _unit(self.embedder.embed_text(item.index_text))

    def _scores(self, query: str) -> np.ndarray:
        q = _unit(self.embedder.embed_text(query))
        return self._matrix @ q

    def retrieve(self, query: str, k: int) -> list[CorpusItem]:
        """Top-k corpus items by cosine similarity, most-similar first.

        Ties break by corpus order (stable), so the ranking is fully determined by
        (query, corpus) — no per-trial variance (brief §4)."""
        if not self.corpus:
            return []
        scores = self._scores(query)
        # argsort is ascending; negate for descending. kind="stable" makes ties
        # fall back to corpus order so the result is deterministic.
        order = np.argsort(-scores, kind="stable")
        return [self.corpus[i] for i in order[:k]]

    def rank_of(self, item_id: str, query: str) -> int:
        """1-based rank of `item_id` across the WHOLE corpus for `query`.

        This is the retrieval-stage metric: where the poison lands for a benign
        query. rank 1 = top hit; rank <= k means it enters the model's context at
        top-k. Reported once per measurement (it is fixed for a fixed corpus), so
        'did the poison get retrieved?' stays separate from injection-ASR."""
        if not self.corpus:
            raise ValueError("empty corpus has no ranks")
        order = np.argsort(-self._scores(query), kind="stable")
        for rank, idx in enumerate(order, start=1):
            if self.corpus[idx].id == item_id:
                return rank
        raise KeyError(f"no corpus item with id {item_id!r}")
