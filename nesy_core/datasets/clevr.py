"""
nesy_core/datasets/clevr.py

CLEVR dataset adapter for LLM + NeSy pipelines.

CLEVR (Compositional Language and Elementary Visual Reasoning) is a
diagnostic VQA benchmark with:
    - Synthetic scenes of 3D geometric objects
    - Compositional NL questions requiring multi-step reasoning
    - Exact answers (counts, attribute values, yes/no)
    - Functional programs (optional, useful as supervision signal)

Dataset files (download from https://cs.stanford.edu/people/jcjohns/clevr/):
    CLEVR_v1.0/
        questions/
            CLEVR_train_questions.json
            CLEVR_val_questions.json
        scenes/
            CLEVR_train_scenes.json
            CLEVR_val_scenes.json
        images/   (not required for text-only NeSy pipelines)

Usage:
    dataset = CLEVRDataset(
        questions_path="CLEVR_v1.0/questions/CLEVR_val_questions.json",
        scenes_path="CLEVR_v1.0/scenes/CLEVR_val_scenes.json",
    )
    item = dataset[0]
    print(item["question"])      # "How many red cubes are there?"
    print(item["answer"])        # "2"
    print(item["asp_facts"][:3]) # ["object(obj_0).", "shape(obj_0, cube).", ...]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import NeSyDataset
from ..semantic_parser import CLEVRSemanticParser


# CLEVR question families mapped to ASP query types
QUESTION_FAMILY_TO_TYPE = {
    "count":           "count",
    "exist":           "exist",
    "query_color":     "query",
    "query_shape":     "query",
    "query_size":      "query",
    "query_material":  "query",
    "compare_integer": "compare",
    "compare_attribute": "compare",
}

# Base ASP theory for CLEVR static QA
# Objects have attributes; spatial relations are given as facts.
# The goal clause is injected per question.
CLEVR_BASE_ASP = """\
% ── CLEVR base theory ────────────────────────────────────────────────────────
% Object / attribute domains — populated by scene facts
% (no actions or time steps needed for static QA)

% Count query support
answer_count(N) :- N = #count{ O : object(O), query_object(O) }.

% Attribute query support
answer_attr(V) :- query_object(O), {attr}(O, V).  % 'attr' replaced at runtime

% Exist query support
answer_yes  :- query_object(_).
answer_no   :- not query_object(_).

% Comparison query support
answer_more  :- answer_count_a(Na), answer_count_b(Nb), Na > Nb.
answer_equal :- answer_count_a(Na), answer_count_b(Nb), Na == Nb.
answer_fewer :- answer_count_a(Na), answer_count_b(Nb), Na < Nb.

#show answer_count/1.
#show answer_attr/1.
#show answer_yes/0.
#show answer_no/0.
#show answer_more/0.
#show answer_equal/0.
#show answer_fewer/0.
"""


class CLEVRDataset(NeSyDataset):
    """
    CLEVR dataset loader and ASP fact generator.

    Parameters
    ----------
    questions_path : str | Path
        Path to CLEVR_*_questions.json
    scenes_path : str | Path
        Path to CLEVR_*_scenes.json
    max_items : int | None
        Limit dataset size (useful for quick experiments).
    split : str
        "train" | "val" | "test"
    """

    def __init__(
        self,
        questions_path: str | Path,
        scenes_path: str | Path,
        max_items: int | None = None,
        split: str = "val",
    ):
        self.split = split
        self.parser = CLEVRSemanticParser()

        # Load questions
        with open(questions_path, "r") as f:
            raw = json.load(f)
        self.questions: list[dict] = raw["questions"]

        # Load scenes, indexed by image filename for O(1) lookup
        with open(scenes_path, "r") as f:
            raw = json.load(f)
        self.scenes: dict[str, dict] = {
            s["image_filename"]: s for s in raw["scenes"]
        }

        if max_items is not None:
            self.questions = self.questions[:max_items]

    # ── NeSyDataset interface ─────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int) -> dict:
        q = self.questions[idx]
        scene = self.scenes.get(q["image_filename"], {})
        asp_facts = self.to_asp_facts({"scene": scene})

        return {
            "id":             q.get("question_index", idx),
            "image_filename": q.get("image_filename", ""),
            "question":       q.get("question", ""),
            "answer":         str(q.get("answer", "")),
            "question_type":  QUESTION_FAMILY_TO_TYPE.get(
                                  q.get("question_family_index", ""), "unknown"
                              ),
            "question_family": q.get("question_family_index", ""),
            "program":        q.get("program", []),   # functional program (optional)
            "asp_facts":      asp_facts,
            "scene":          scene,
            "metadata": {
                "split":      self.split,
                "image_index": q.get("image_index", -1),
            },
        }

    def to_asp_facts(self, item: dict) -> list[str]:
        """
        Convert a CLEVR scene dict into ASP facts.

        The scene dict should be one entry from CLEVR's scenes JSON.
        """
        scene = item.get("scene", {})
        if not scene:
            return []
        return self.parser.scene_to_facts(scene)

    # ── CLEVR-specific helpers ────────────────────────────────────────────

    def build_asp_program(self, item: dict, goal_asp: str = "") -> str:
        """
        Assemble the full ASP program for a single CLEVR question.

        Args:
            item:     A dataset item from __getitem__().
            goal_asp: ASP goal clause (from LLM or question_to_goal()).
                      If empty, a stub is used.

        Returns:
            Complete ASP program string ready for gen_answer_set().
        """
        facts = "\n".join(item["asp_facts"])
        if not goal_asp:
            goal_asp = self.parser.question_to_goal(item["question"])

        return f"{CLEVR_BASE_ASP}\n% Scene facts\n{facts}\n\n% Goal\n{goal_asp}\n"

    def get_question_types(self) -> dict[str, int]:
        """Return a count of each question type in the dataset."""
        counts: dict[str, int] = {}
        for q in self.questions:
            family = q.get("question_family_index", "unknown")
            qtype = QUESTION_FAMILY_TO_TYPE.get(family, "unknown")
            counts[qtype] = counts.get(qtype, 0) + 1
        return counts

    def filter_by_type(self, question_type: str) -> "CLEVRDataset":
        """
        Return a view of the dataset filtered to a single question type.

        TODO: return a proper subset view rather than mutating in place.
        """
        target_family = {v: k for k, v in QUESTION_FAMILY_TO_TYPE.items()}.get(
            question_type, question_type
        )
        filtered = [
            q for q in self.questions
            if QUESTION_FAMILY_TO_TYPE.get(q.get("question_family_index", ""), "") == question_type
        ]
        clone = object.__new__(CLEVRDataset)
        clone.split = self.split
        clone.parser = self.parser
        clone.questions = filtered
        clone.scenes = self.scenes
        return clone

    def get_demo_records(self) -> list[dict]:
        """
        Build RetrievalEngine-compatible records from this dataset.

        Returns records in the format:
            {"episode": "<question>", "positive": True, "answer": "<answer>"}
        """
        records = []
        for q in self.questions:
            records.append({
                "episode":  q.get("question", ""),
                "positive": True,   # CLEVR questions always have a correct answer
                "answer":   str(q.get("answer", "")),
                "image_filename": q.get("image_filename", ""),
            })
        return records

    # ── Evaluation ────────────────────────────────────────────────────────

    @staticmethod
    def evaluate(predictions: list[str], ground_truths: list[str]) -> dict[str, float]:
        """
        Compute exact-match accuracy for CLEVR answers.

        Args:
            predictions:   List of predicted answer strings.
            ground_truths: List of ground-truth answer strings.

        Returns:
            {"accuracy": float, "correct": int, "total": int}
        """
        assert len(predictions) == len(ground_truths)
        correct = sum(
            p.strip().lower() == g.strip().lower()
            for p, g in zip(predictions, ground_truths)
        )
        total = len(predictions)
        return {
            "accuracy": correct / total if total > 0 else 0.0,
            "correct":  correct,
            "total":    total,
        }

    @staticmethod
    def extract_answer_from_asp(answer_sets: list[list[str]]) -> str:
        """
        Pull the CLEVR answer out of Clingo answer sets.

        Looks for atoms: answer_count(N), answer_attr(V),
                         answer_yes, answer_no,
                         answer_more, answer_equal, answer_fewer.

        Returns the answer as a string, or "" if nothing found.
        """
        if not answer_sets:
            return ""

        atoms = answer_sets[0]  # take first model
        for atom in atoms:
            if atom.startswith("answer_count("):
                return atom[len("answer_count("):-1]
            elif atom.startswith("answer_attr("):
                return atom[len("answer_attr("):-1]
            elif atom == "answer_yes":
                return "yes"
            elif atom == "answer_no":
                return "no"
            elif atom == "answer_more":
                return "more"
            elif atom == "answer_equal":
                return "equal"
            elif atom == "answer_fewer":
                return "fewer"
        return ""
