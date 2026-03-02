"""
nesy_core/retrieval.py

Generalised KNN retrieval engine for in-context learning.

Ported from Alfworld/utils.py — the data format is generalised so it
works with any dataset that provides (text, label, metadata) triples.

Default embedding model: all-MiniLM-L12-v2 (fast, good quality).
Swap via the `embedder_name` constructor argument.

Usage:
    engine = RetrievalEngine(demo_data_path="./data/demos.json")
    prompt_context = engine.search_demo(query_text, k=3)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


# ---------------------------------------------------------------------------
# Demo record schema
# ---------------------------------------------------------------------------
# Each record in the demo JSON file should have at least:
#   "episode"  : str   — the trajectory / context text
#   "positive" : bool  — whether the episode succeeded
#
# Optional fields used by CLEVR adapter:
#   "question"  : str  — the NL question
#   "answer"    : str  — ground-truth answer
#   "program"   : list — functional program (for reference)


class RetrievalEngine:
    """
    KNN retrieval over a demo dataset using sentence embeddings.

    Parameters
    ----------
    demo_data_path : str | Path
        Path to a JSON file containing demo records.
        Records must have at minimum: {"episode": str, "positive": bool}.
    embedder_name : str
        SentenceTransformer model name.
    """

    def __init__(
        self,
        demo_data_path: str | Path = "./data/demo/demos.json",
        embedder_name: str = "all-MiniLM-L12-v2",
    ):
        self.demo_set = pd.read_json(demo_data_path)
        self.embedder = SentenceTransformer(embedder_name)

    # ── Core KNN ──────────────────────────────────────────────────────────

    def knn_retrieval(
        self,
        query: str,
        k: int = 3,
        field: str = "episode",
    ) -> list[tuple[int, Any, str, float]]:
        """
        Find the k demo records closest to `query` by cosine similarity.

        Args:
            query: The query string (observation, question, etc.).
            k:     Number of neighbours to return.
            field: Which record field to embed for comparison.

        Returns:
            Sorted list of (index, positive, episode_text, distance) tuples.
            Distance is negated cosine similarity (lower = closer).
        """
        query_emb = self.embedder.encode(query)
        top_k: list[tuple[int, Any, str, float]] = []

        for _, row in self.demo_set.iterrows():
            item_emb = self.embedder.encode(str(row[field]))
            dist = float(-cos_sim(query_emb, item_emb)[0][0])

            entry = (0, row.get("positive", True), str(row[field]), dist)

            if len(top_k) < k:
                top_k.append(entry)
                top_k.sort(key=lambda x: x[-1])
            elif dist < top_k[-1][-1]:
                top_k[-1] = entry
                top_k.sort(key=lambda x: x[-1])

        return top_k

    # ── Formatted demo prompt ─────────────────────────────────────────────

    def search_demo(
        self,
        query: str,
        k: int = 3,
        field: str = "episode",
        success_label: str = "success",
        fail_label: str = "fail",
    ) -> str:
        """
        Retrieve k nearest demos and format them as an in-context prompt block.

        Args:
            query:         The query string.
            k:             Number of examples to retrieve.
            field:         Demo field to embed.
            success_label: Tag appended to positive examples.
            fail_label:    Tag appended to negative examples.

        Returns:
            A formatted string of retrieved episodes ready to prepend to a prompt.
        """
        retrieved = self.knn_retrieval(query, k=k, field=field)
        parts = []
        for _, positive, episode, _ in retrieved:
            label = success_label if positive else fail_label
            parts.append(f"\n{episode} ({label})")
        return "".join(parts)

    # ── Batch embedding (for offline indexing) ────────────────────────────

    def embed_all(self, field: str = "episode") -> list[Any]:
        """
        Pre-compute embeddings for all records.
        Useful for large datasets where per-query encoding is too slow.

        Returns a list of embedding tensors aligned with self.demo_set rows.

        TODO: add FAISS/annoy index support for large-scale retrieval.
        """
        texts = self.demo_set[field].astype(str).tolist()
        return self.embedder.encode(texts, show_progress_bar=True)

    # ── Convenience loaders ───────────────────────────────────────────────

    @classmethod
    def from_list(
        cls,
        records: list[dict],
        embedder_name: str = "all-MiniLM-L12-v2",
        tmp_path: str = "/tmp/nesy_demos.json",
    ) -> "RetrievalEngine":
        """
        Build a RetrievalEngine from an in-memory list of record dicts.
        Writes a temporary JSON file to satisfy the pandas reader.
        """
        Path(tmp_path).write_text(json.dumps(records))
        engine = cls.__new__(cls)
        engine.demo_set = pd.DataFrame(records)
        engine.embedder = SentenceTransformer(embedder_name)
        return engine
