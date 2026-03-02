"""
nesy_core/datasets/base.py

Abstract dataset interface for NeSy + LLM pipelines.

Every dataset adapter should:
1. Subclass NeSyDataset
2. Implement __len__, __getitem__, and to_asp_facts()
3. Optionally implement get_demo_records() for retrieval-augmented prompting

This keeps dataset-specific loading/parsing isolated from the core pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator


class NeSyDataset(ABC):
    """
    Abstract base for datasets used in LLM + NeSy experiments.

    Each item yielded should contain enough information for the pipeline to:
        - Build an ASP program (facts from the scene/state)
        - Express a goal or query
        - Evaluate the result against a ground-truth answer

    Minimal item schema (dict):
        {
            "id":          str | int,   # unique identifier
            "question":    str,         # NL query / goal instruction
            "answer":      Any,         # ground-truth answer for evaluation
            "asp_facts":   list[str],   # pre-computed ASP fact strings
            "metadata":    dict,        # domain-specific extras
        }
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return total number of items."""

    @abstractmethod
    def __getitem__(self, idx: int) -> dict:
        """Return a single item by index."""

    @abstractmethod
    def to_asp_facts(self, item: dict) -> list[str]:
        """
        Convert a dataset item into ASP fact strings.

        This is called by the pipeline to build the initial state of
        the ASP program for each question/episode.
        """

    def __iter__(self) -> Iterator[dict]:
        for i in range(len(self)):
            yield self[i]

    def get_demo_records(self) -> list[dict]:
        """
        Return a list of demo records suitable for the RetrievalEngine.

        Each record should have:
            {"episode": str, "positive": bool, ...}

        Override in subclasses that support few-shot retrieval.
        Default returns empty list (no retrieval).
        """
        return []

    def split(self, train_frac: float = 0.8, seed: int = 42) -> tuple["NeSyDataset", "NeSyDataset"]:
        """
        Simple random train/val split.

        TODO: implement once concrete subclasses are in place.
        """
        raise NotImplementedError("Override split() in your dataset subclass.")
