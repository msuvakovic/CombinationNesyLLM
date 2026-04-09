"""
nesy_core/datasets/ambig_qa.py

AmbigQA dataset adapter for LLM + NeSy pipelines.

AmbigQA (Ambiguous Questions in Open-Domain QA) is a QA benchmark built on
NaturalQuestions, where questions are disambiguated into multiple well-formed
sub-questions, each with unambiguous answers.

Dataset structure (HuggingFace: sewon/ambig_qa, config: "full"):
    Each item has:
        id            — unique string ID
        question      — original (potentially ambiguous) NL question
        annotations   — dict with:
            type      — "singleAnswer" | "multipleQAs"
            answer    — list of acceptable answers (for singleAnswer)
            qaPairs   — list of {question: [...], answer: [[...], ...]}
                        (for multipleQAs)
        nq_answer     — original NaturalQuestions answer strings
        nq_doc_title  — source Wikipedia article title

Usage:
    from datasets import load_dataset
    from nesy_core.datasets.ambig_qa import AmbigQADataset

    hf_ds = load_dataset("sewon/ambig_qa", "full")
    dataset = AmbigQADataset(hf_ds["train"])
    item = dataset[0]
    print(item["question"])       # "When did the Simpsons first air?"
    print(item["answer"])         # list of acceptable answer strings
    print(item["asp_facts"][:3])  # ["question_type(multipleQAs).", ...]
"""

from __future__ import annotations

from typing import Any

from .base import NeSyDataset


def _sanitize(text: str) -> str:
    """Escape characters that would break an ASP atom."""
    return (
        text.lower()
        .replace('"', "")
        .replace("'", "")
        .replace(",", "")
        .replace(".", "")
        .replace("(", "")
        .replace(")", "")
        .replace("\n", " ")
        .strip()
        .replace(" ", "_")
    )


class AmbigQADataset(NeSyDataset):
    """
    AmbigQA dataset loader and ASP fact generator.

    Parameters
    ----------
    hf_split : datasets.Dataset
        A HuggingFace dataset split, e.g. load_dataset("sewon/ambig_qa",
        "full")["train"].
    max_items : int | None
        Limit dataset size (useful for quick experiments).
    """

    def __init__(self, hf_split, max_items: int | None = None):
        self._data = hf_split
        if max_items is not None:
            self._data = self._data.select(range(min(max_items, len(self._data))))

    # ── NeSyDataset interface ─────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict:
        raw = self._data[idx]
        annotations = raw.get("annotations", {})
        ann_type = (annotations.get("type") or ["singleAnswer"])[0]

        # Collect all acceptable answers
        if ann_type == "singleAnswer":
            raw_answers = annotations.get("answer") or []
            # answer is a list-of-lists: [["David Morse"]] — flatten one level
            answers: list[str] = []
            for entry in raw_answers:
                if isinstance(entry, list):
                    answers.extend(str(a) for a in entry)
                else:
                    answers.append(str(entry))
        else:
            # multipleQAs — flatten all answers from qaPairs
            qa_pairs = annotations.get("qaPairs") or []
            answers = []
            for pair in qa_pairs:
                for ans_group in pair.get("answer", []):
                    if isinstance(ans_group, list):
                        answers.extend(ans_group)
                    else:
                        answers.append(str(ans_group))

        # Build clarifying sub-questions (only for multipleQAs)
        qa_pairs_out: list[dict] = []
        if ann_type == "multipleQAs":
            for pair in (annotations.get("qaPairs") or []):
                sub_qs = pair.get("question", [])
                sub_as = pair.get("answer", [])
                qa_pairs_out.append({"questions": sub_qs, "answers": sub_as})

        item = {
            "id":            raw.get("id", str(idx)),
            "question":      raw.get("question", ""),
            "answer":        answers,
            "annotation_type": ann_type,
            "qa_pairs":      qa_pairs_out,
            "nq_answer":     raw.get("nq_answer", []),
            "nq_doc_title":  raw.get("nq_doc_title", ""),
            "metadata": {
                "viewed_doc_titles": raw.get("viewed_doc_titles", []),
            },
        }
        item["asp_facts"] = self.to_asp_facts(item)
        return item

    def to_asp_facts(self, item: dict) -> list[str]:
        """
        Convert an AmbigQA item into ASP fact strings.

        Emitted facts
        -------------
        question_type(<type>).
            "singleAnswer" or "multipleQAs"

        nq_doc(<sanitized_title>).
            Source Wikipedia article (when available).

        acceptable_answer(<idx>, <sanitized_answer>).
            One fact per acceptable answer string.

        sub_question(<pair_idx>, <q_idx>, <sanitized_question>).
            One fact per clarifying sub-question (multipleQAs only).

        sub_answer(<pair_idx>, <ans_idx>, <sanitized_answer>).
            One fact per sub-answer group (multipleQAs only).
        """
        facts: list[str] = []

        ann_type = item.get("annotation_type", "singleAnswer")
        facts.append(f'question_type("{ann_type}").')

        doc_title = item.get("nq_doc_title", "")
        if doc_title:
            facts.append(f'nq_doc("{_sanitize(doc_title)}").')

        for i, ans in enumerate(item.get("answer", [])):
            facts.append(f'acceptable_answer({i}, "{_sanitize(str(ans))}").')

        for pi, pair in enumerate(item.get("qa_pairs", [])):
            for qi, sq in enumerate(pair.get("questions", [])):
                facts.append(f'sub_question({pi}, {qi}, "{_sanitize(str(sq))}").')
            for ai, ans_group in enumerate(pair.get("answers", [])):
                if isinstance(ans_group, list):
                    for ans in ans_group:
                        facts.append(f'sub_answer({pi}, {ai}, "{_sanitize(str(ans))}").')
                else:
                    facts.append(f'sub_answer({pi}, {ai}, "{_sanitize(str(ans_group))}").')

        return facts

    # ── Retrieval support ─────────────────────────────────────────────────

    def get_demo_records(self) -> list[dict]:
        """
        Build RetrievalEngine-compatible records from this dataset.

        Returns records in the format expected by RetrievalEngine:
            {"episode": "<question>", "positive": True, "answer": "<answer>"}
        """
        records = []
        for item in self:
            answers = item.get("answer", [])
            records.append({
                "episode":  item["question"],
                "positive": True,
                "answer":   answers[0] if answers else "",
                "all_answers": answers,
                "annotation_type": item["annotation_type"],
            })
        return records

    # ── Evaluation ────────────────────────────────────────────────────────

    @staticmethod
    def evaluate(
        predictions: list[str],
        ground_truths: list[list[str]],
    ) -> dict[str, float]:
        """
        Compute accuracy: prediction matches any acceptable answer (exact match,
        case-insensitive).

        Args:
            predictions:   List of predicted answer strings.
            ground_truths: List of acceptable-answer lists (one per item).

        Returns:
            {"accuracy": float, "correct": int, "total": int}
        """
        assert len(predictions) == len(ground_truths)
        correct = 0
        for pred, gts in zip(predictions, ground_truths):
            pred_norm = pred.strip().lower()
            if any(pred_norm == g.strip().lower() for g in gts):
                correct += 1
        total = len(predictions)
        return {
            "accuracy": correct / total if total > 0 else 0.0,
            "correct":  correct,
            "total":    total,
        }

    # ── Convenience factory ───────────────────────────────────────────────

    @classmethod
    def from_hf(
        cls,
        split: str = "train",
        max_items: int | None = None,
    ) -> "AmbigQADataset":
        """
        Load directly from HuggingFace Hub.

        Args:
            split:     "train" or "validation".
            max_items: Optional cap on dataset size.

        Example:
            ds = AmbigQADataset.from_hf("train", max_items=500)
        """
        from datasets import load_dataset  # type: ignore
        hf_ds = load_dataset("sewon/ambig_qa", "full")
        return cls(hf_ds[split], max_items=max_items)
