#!/usr/bin/env python3
"""Unit tests for the System 2 retriever — pure cosine-ranking logic.

The retriever is tested in ISOLATION from CLIP via a FakeEmbedder that maps known
strings/objects to known vectors. That keeps these tests deterministic and
CPU-only (no model download), and proves the measurement-critical property: for a
fixed query + fixed corpus the ranking is exact and repeatable, so retrieval adds
no per-trial confound to ASR (brief §4, one variable at a time).
"""

from __future__ import annotations

import unittest

import numpy as np

from systems.system2_indirect.retrieval import CorpusItem, Retriever


class FakeEmbedder:
    """Deterministic embedder: looks each item/query up in a fixed table.

    Text and images share one table (one shared embedding space), mirroring how
    CLIP puts both modalities in a joint space — which is exactly what lets a
    text query retrieve an image.
    """

    def __init__(self, table: dict[object, list[float]]) -> None:
        self.table = {k: np.asarray(v, dtype=np.float32) for k, v in table.items()}

    def embed_text(self, text: str) -> np.ndarray:
        return self.table[text]

    def embed_image(self, image: object) -> np.ndarray:
        return self.table[image]


class RetrieverRankingTest(unittest.TestCase):
    def test_retrieve_returns_items_ranked_by_cosine_similarity(self) -> None:
        # Query points along [1, 0]. A is identical (cos 1.0), C is at 45deg
        # (cos ~0.707), B is orthogonal (cos 0.0). Expected order: A, C, B.
        embedder = FakeEmbedder({
            "query": [1.0, 0.0],
            "a-text": [2.0, 0.0],      # non-unit on purpose: retriever must normalize
            "b-text": [0.0, 5.0],
            "c-text": [1.0, 1.0],
        })
        corpus = [
            CorpusItem(id="B", modality="text", content="b-text"),
            CorpusItem(id="A", modality="text", content="a-text"),
            CorpusItem(id="C", modality="text", content="c-text"),
        ]
        retriever = Retriever(corpus, embedder)

        top2 = retriever.retrieve("query", k=2)

        self.assertEqual([item.id for item in top2], ["A", "C"])

    def test_image_items_retrieve_by_index_text_not_by_embed_image(self) -> None:
        # Document-RAG design: an image is retrieved via its OCR'd index_text
        # (embedded as TEXT), not via a cross-modal image embedding — which fixes
        # CLIP's modality gap. If retrieval ever calls embed_image for ranking,
        # this fake makes it fail loudly.
        class TextOnlyEmbedder:
            def embed_text(self, text: str) -> np.ndarray:
                return (np.array([1.0, 0.0], dtype=np.float32)
                        if "backend" in text
                        else np.array([0.0, 1.0], dtype=np.float32))

            def embed_image(self, image: object) -> np.ndarray:
                raise AssertionError("retrieval must not embed images for ranking")

        corpus = [
            CorpusItem(id="other", modality="text", content="a cooking blog post"),
            CorpusItem(id="cv", modality="image", content=object(),
                       index_text="jordan backend engineer python"),
        ]
        retriever = Retriever(corpus, TextOnlyEmbedder())

        top1 = retriever.retrieve("backend engineer", k=1)
        self.assertEqual([it.id for it in top1], ["cv"])

    def test_text_item_index_text_defaults_to_its_content(self) -> None:
        item = CorpusItem(id="t", modality="text", content="hello world")
        self.assertEqual(item.index_text, "hello world")

    def test_rank_of_reports_one_based_position_of_an_item(self) -> None:
        # Same geometry: for "query" the full order is A(1), C(2), B(3). rank_of
        # is the retrieval-stage metric — where the poison lands for a benign
        # query — so it must count the item's position across the WHOLE corpus.
        embedder = FakeEmbedder({
            "query": [1.0, 0.0],
            "a-text": [2.0, 0.0],
            "b-text": [0.0, 5.0],
            "c-text": [1.0, 1.0],
        })
        corpus = [
            CorpusItem(id="B", modality="text", content="b-text"),
            CorpusItem(id="A", modality="text", content="a-text"),
            CorpusItem(id="C", modality="text", content="c-text", is_poison=True),
        ]
        retriever = Retriever(corpus, embedder)

        self.assertEqual(retriever.rank_of("C", "query"), 2)
        self.assertEqual(retriever.rank_of("A", "query"), 1)
        self.assertEqual(retriever.rank_of("B", "query"), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
