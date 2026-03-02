"""
nesy_core/semantic_parser.py

Abstract base class and utilities for Natural Language → ASP conversion.

Ported and generalised from Alfworld/utils.py.

To add a new domain:
1. Subclass SemanticParser
2. Implement act_to_facts() and obs_to_facts()
3. Optionally override parse_act_obs() if your trajectory format differs

The CLEVR parser is included here as a concrete example.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class SemanticParser(ABC):
    """
    Convert natural-language trajectories to ASP fact strings.

    Each domain (Alfworld, CLEVR, etc.) implements its own subclass.
    """

    def parse_act_obs(self, text: str) -> list[str]:
        """
        Parse a full 'Act N: ... / Obs N: ...' block into ASP facts.

        This default implementation splits on 'Act'/'Obs' prefixes.
        Override if your trajectory format differs.
        """
        lines = text.strip().split("\n")
        act_info = None
        obs_info = None

        for line in lines:
            if line.startswith("Act"):
                act_info = self._parse_act_line(line)
            elif line.startswith("Obs"):
                obs_info = self._parse_obs_line(line)

        if obs_info is None:
            return []

        step, observation = obs_info
        if "Nothing happens" in observation:
            return [f"action(no_op, {step - 1})."]

        facts: list[str] = []
        if act_info:
            facts.extend(self.act_to_facts(act_info))
        if obs_info:
            facts.extend(self.obs_to_facts(obs_info))
        return facts

    @abstractmethod
    def act_to_facts(self, act_info: tuple[int, str]) -> list[str]:
        """
        Convert a (step, action_string) pair to ASP fact strings.
        e.g. (0, "take apple 1 from table 1")
             -> ["action(pick_up(apple_1, table_1), 0)."]
        """

    @abstractmethod
    def obs_to_facts(self, obs_info: tuple[int, str]) -> list[str]:
        """
        Convert a (step, observation_string) pair to ASP fact strings.
        e.g. (1, "You are at table 1. On table 1 you see apple 1.")
             -> ["robot_at(table_1, 1).", "at(apple_1, table_1, 1)."]
        """

    # ── Internal helpers ───────────────────────────────────────────────────

    def _parse_act_line(self, line: str) -> tuple[int, str] | None:
        m = re.match(r"Act (\d+): (.+)", line)
        if m:
            return int(m.group(1)), m.group(2)
        return None

    def _parse_obs_line(self, line: str) -> tuple[int, str] | None:
        m = re.match(r"Obs (\d+): (.+)", line)
        if m:
            return int(m.group(1)), m.group(2)
        return None


# ---------------------------------------------------------------------------
# Alfworld concrete parser (ported from Alfworld/utils.py)
# ---------------------------------------------------------------------------

class AlfworldSemanticParser(SemanticParser):
    """
    NL → ASP for the ALFWorld embodied navigation domain.

    Action predicates: pick_up, put_down, go_to, heat, cool, clean, use, open, close
    State predicates:  robot_at, at, holding, is_heated, is_cooled, is_cleaned, is_opened
    """

    def act_to_facts(self, act_info: tuple[int, str]) -> list[str]:
        step, action = act_info
        facts: list[str] = []

        if "take" in action or "pick up" in action:
            m = re.search(r"(take|pick up) (\w+ \d+) from (\w+ \d+)", action)
            if m:
                obj, loc = m.group(2), m.group(3)
                facts.append(
                    f"action(pick_up({_slug(obj)}, {_slug(loc)}), {step})."
                )

        elif "go to" in action:
            m = re.search(r"go to (\w+ \d+)", action)
            if m:
                loc = m.group(1)
                facts.append(f"action(go_to({_slug(loc)}), {step}).")
                facts.append(f"location({_slug(loc)}).")

        elif "place" in action or "put" in action:
            m = re.search(r"(place|put) (\w+ \d+) (?:in|on|in/on) (\w+ \d+)", action)
            if m:
                obj, loc = m.group(2), m.group(3)
                facts.append(
                    f"action(put_down({_slug(obj)}, {_slug(loc)}), {step})."
                )

        elif "cool" in action:
            m = re.search(r"cool (\w+ \d+) (?:in|at|with) (\w+ \d+)", action)
            if m:
                obj, loc = m.group(1), m.group(2)
                facts.append(f"action(cool({_slug(obj)}, {_slug(loc)}), {step}).")

        elif "clean" in action:
            m = re.search(r"clean (\w+ \d+) (?:in|at|with) (\w+ \d+)", action)
            if m:
                obj, loc = m.group(1), m.group(2)
                facts.append(f"action(clean({_slug(obj)}, {_slug(loc)}), {step}).")

        elif "heat" in action:
            m = re.search(r"heat (\w+ \d+) (?:in|at|with) (\w+ \d+)", action)
            if m:
                obj, loc = m.group(1), m.group(2)
                facts.append(f"action(heat({_slug(obj)}, {_slug(loc)}), {step}).")

        elif "use" in action:
            m = re.search(r"use (\w+ \d+)", action)
            if m:
                facts.append(f"action(use({_slug(m.group(1))}), {step}).")

        elif "open" in action:
            m = re.search(r"open (\w+ \d+)", action)
            if m:
                facts.append(f"action(open({_slug(m.group(1))}), {step}).")

        elif "close" in action:
            m = re.search(r"close (\w+ \d+)", action)
            if m:
                facts.append(f"action(close({_slug(m.group(1))}), {step}).")

        return facts

    def obs_to_facts(self, obs_info: tuple[int, str]) -> list[str]:
        step, observation = obs_info
        facts: list[str] = []

        # Robot location
        m = re.search(r"(arrive at|you are at) (\w+ \d+)", observation)
        if m:
            loc = m.group(2)
            facts.append(f"robot_at({_slug(loc)}, {step}).")

        # Objects on a receptacle: "On the table 1, ..."
        rec_m = re.search(r"On the (\w+ \d+),", observation)
        if rec_m:
            receptacle = rec_m.group(1)
            for obj in re.findall(r"a (\w+ \d+)", observation):
                facts += [
                    f"location({_slug(receptacle)}).",
                    f"at({_slug(obj)}, {_slug(receptacle)}, {step}).",
                ]
        else:
            for obj, loc in re.findall(
                r"(\w+ \d+) (?:is in|is on|in/on|,? you see) (\w+ \d+)", observation
            ):
                facts += [
                    f"location({_slug(loc)}).",
                    f"at({_slug(obj)}, {_slug(loc)}, {step}).",
                ]

        # Holding
        m = re.search(
            r"(?:pick up|put) the (\w+ \d+) (?:from|in|on|in/on) the (\w+ \d+)",
            observation,
        )
        if m:
            obj, loc = m.group(1), m.group(2)
            if "pick up" in observation:
                facts.append(f"holding({_slug(obj)}, {step}).")
            else:
                facts.append(f"at({_slug(obj)}, {_slug(loc)}, {step}).")

        # Object state changes
        for action_word, state_pred in {
            "heat": "is_heated",
            "cool": "is_cooled",
            "clean": "is_cleaned",
        }.items():
            if action_word in observation:
                m = re.search(
                    f"{action_word} the (\w+ \d+) using the (\w+ \d+)", observation
                )
                if m:
                    obj = m.group(1)
                    facts.append(f"{state_pred}({_slug(obj)}, {step}).")

        # Opened
        m = re.search(r"(\w+ \d+) is open", observation)
        if m:
            facts.append(f"is_opened({_slug(m.group(1))}, {step}).")

        return facts


# ---------------------------------------------------------------------------
# CLEVR concrete parser
# ---------------------------------------------------------------------------

class CLEVRSemanticParser(SemanticParser):
    """
    Convert CLEVR scene-graph JSON entries into ASP facts.

    CLEVR objects have: shape, color, size, material, 3D position,
    and pairwise spatial relations (left, right, front, behind).

    ASP representation:
        object(obj_0).
        shape(obj_0, cube).
        color(obj_0, red).
        size(obj_0, large).
        material(obj_0, rubber).
        left_of(obj_0, obj_1).   % relation facts
        ...

    This parser handles CLEVR scene dicts directly.
    The act_to_facts / obs_to_facts interface is adapted for QA:
        act_info  -> (step, question_string)
        obs_info  -> (step, answer_string)
    """

    # Attribute predicates recognised in CLEVR scenes
    ATTR_PREDICATES = ("shape", "color", "size", "material")
    RELATION_PREDICATES = ("left", "right", "front", "behind")

    def scene_to_facts(self, scene: dict) -> list[str]:
        """
        Convert a CLEVR scene dict to ASP facts.

        Args:
            scene: One entry from CLEVR's scenes JSON, e.g.:
                   {
                     "objects": [
                       {"shape": "cube", "color": "red", "size": "large",
                        "material": "rubber", "3d_coords": [x, y, z]},
                       ...
                     ],
                     "relationships": {
                       "left": [[1,2],[],[0],...],
                       ...
                     }
                   }

        Returns:
            List of ASP fact strings.
        """
        facts: list[str] = []
        objects = scene.get("objects", [])

        for i, obj in enumerate(objects):
            obj_id = f"obj_{i}"
            facts.append(f"object({obj_id}).")
            for attr in self.ATTR_PREDICATES:
                val = obj.get(attr, "")
                if val:
                    facts.append(f"{attr}({obj_id}, {val}).")

            # 3D coordinates as separate predicates (useful for spatial reasoning)
            coords = obj.get("3d_coords", [])
            if len(coords) == 3:
                x, y, z = [round(c, 2) for c in coords]
                facts.append(f"position({obj_id}, {x}, {y}, {z}).")

        # Pairwise spatial relations
        relationships = scene.get("relationships", {})
        for rel_name in self.RELATION_PREDICATES:
            rel_lists = relationships.get(rel_name, [])
            for i, neighbours in enumerate(rel_lists):
                src = f"obj_{i}"
                for j in neighbours:
                    tgt = f"obj_{j}"
                    facts.append(f"{rel_name}_of({src}, {tgt}).")

        return facts

    def question_to_goal(self, question: str) -> str:
        """
        Produce a rough ASP goal stub for a CLEVR question.

        In practice you will want the LLM to do this translation using
        the 'goal' prompt template.  This method provides a simple regex
        fallback for the most common CLEVR question types.

        Question families and their goal patterns:
            count    -> answer(N) :- #count{O : object(O), ...} = N.
            exist    -> answer(yes) :- object(O), ...
            query_*  -> answer(V)  :- attr(O, V), ...
            compare_* -> answer(yes/no) :- ...

        TODO: replace with LLM-based translation for full coverage.
        """
        q = question.lower()

        if q.startswith("how many"):
            # e.g. "How many red cubes are there?"
            return "% TODO: count query — use LLM goal parser\nanswer(N) :- #count{O : object(O)} = N."

        elif q.startswith("is there"):
            return "% TODO: exist query — use LLM goal parser\nanswer(yes) :- object(_)."

        elif q.startswith("what") and ("color" in q or "shape" in q or "size" in q or "material" in q):
            attr = next(
                (a for a in self.ATTR_PREDICATES if a in q), "shape"
            )
            return f"% TODO: query_{attr} — use LLM goal parser\nanswer(V) :- {attr}(_, V)."

        else:
            return "% TODO: compare/other query — use LLM goal parser\nanswer(yes)."

    # ── SemanticParser interface (adapted for QA) ──────────────────────────

    def act_to_facts(self, act_info: tuple[int, str]) -> list[str]:
        """
        For CLEVR (a static QA task) there are no actions — return empty.
        Override if you extend CLEVR to an interactive setting.
        """
        return []

    def obs_to_facts(self, obs_info: tuple[int, str]) -> list[str]:
        """
        For CLEVR, 'observations' are scene descriptions.
        In practice use scene_to_facts() directly with the scene dict.
        """
        return []

    def parse_act_obs(self, text: str) -> list[str]:
        """
        Not meaningful for static CLEVR QA.
        Call scene_to_facts(scene_dict) instead.
        """
        return []


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _slug(s: str) -> str:
    """Replace spaces with underscores (for ASP atom names)."""
    return s.replace(" ", "_")
