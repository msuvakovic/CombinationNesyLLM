"""
nesy_core/pipeline.py

Generalized LLM + NeSy Pipeline.

Ported and decoupled from Alfworld/our_pipeline.py.
Domain-specific behaviour is injected via subclassing or prompt configuration.
"""

import os
import pickle

from .datasets.ambig_qa import AmbigQADataset

from abc import ABC, abstractmethod
from typing import Anya
from .asp_utils import (
    sanitize_asp,
    keep_only_parseable_rules,
    gen_answer_set,
)
from .llm_utils import call_llm

MAX_VERIFICATION_TRIAL = 3


# ---------------------------------------------------------------------------
# Rule post-processing helpers (domain-agnostic)
# ---------------------------------------------------------------------------

def extract_rule(text: str) -> str:
    """Strip markdown fences and normalise backslashes from LLM rule output."""
    cleaned = text.replace("```", " ").replace("**", "")
    cleaned = cleaned.replace("\\", "!").strip()
    return cleaned + "\n"


def rule_filtering(text: str) -> str:
    """Keep only lines that look like ASP rules (contain ':-') and drop 'take'."""
    lines = text.replace("`", "").split("\n")
    lines = [l for l in lines if ":-" in l]
    lines = [l for l in lines if "take" not in l]
    return "\n".join(lines)


def load_generalized_rules(load_path: str) -> str:
    with open(load_path, "r") as f:
        return f.read()


def save_generalized_rules(rules: str, save_path: str) -> None:
    with open(save_path, "w") as f:
        f.write(rules)


# ---------------------------------------------------------------------------
# Core Pipeline
# ---------------------------------------------------------------------------





class Pipeline(ABC):
    """
    Base class for LLM + NeSy (ASP) pipelines.

    Subclass this for each domain (ALFWorld, CLEVR, etc.) and implement:
        - parse_observation()
        - parse_goal()
        - get_target_predicates()  (for ILP generalisation)

    Everything else — LLM dispatch, ASP solving, caching, ILP induction,
    the symbolic-feedback correction loop — lives here and is reusable.
    """

    def __init__(self, args: dict = {}):
        # ── ASP state ─────────────────────────────────────────────────────
        self.asp_program: str = ""       # base domain theory (fixed)
        self.init_state: str = ""        # initial facts for the current episode
        self.dynamic_facts: str = ""     # facts added during an episode
        self.adapted_rules: str = ""     # rules adapted from experience
        self.goal_state: str = ""        # current goal expressed in ASP

        # ── Conversation / episode tracking ───────────────────────────────
        self.adaptation_chat_history: str = ""
        self.history: str = ""
        self.step_num: int = 0

        # ── LLM config ────────────────────────────────────────────────────
        self.engine: str = "claude-sonnet-4-6"
        self.temperature: float = 0.0
        self.max_tokens: int = 1024
        self.method: str = "ours"

        # ── Prompt store  {kind -> prompt string} ─────────────────────────
        self.prompt: dict[str, str] = {}

        # ── Cache  {kind -> {key -> response}} ────────────────────────────
        self.path_cache: dict[str, str] = {}
        self.cache: dict[str, dict] = {}

        # ── Solver config ─────────────────────────────────────────────────
        self.clingo_seed: int = 1

        # ── Misc ──────────────────────────────────────────────────────────
        self.rule_save_path: str = ""
        self.count: int = 0
        self.total_cost: float = 0.0

        # Apply any kwargs passed in as args dict
        for k, v in args.items():
            setattr(self, k, v)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def clean(self) -> None:
        """Reset per-episode state. Call between episodes."""
        self.init_state = ""
        self.dynamic_facts = ""
        self.adapted_rules = ""
        self.goal_state = ""
        self.adaptation_chat_history = ""
        self.step_num = 0
        self.history = ""

    # ── Prompt management ─────────────────────────────────────────────────

    def load_prompt(self, kind_to_path: dict[str, str]) -> None:
        """Load prompt files into self.prompt keyed by kind string."""
        for kind, path in kind_to_path.items():
            with open(path, "r", encoding="utf-8") as f:
                self.prompt[kind] = f.read()

    # ── Cache management ──────────────────────────────────────────────────

    def load_cache(self) -> None:
        for kind, path in self.path_cache.items():
            if os.path.isfile(path):
                with open(path, "rb") as f:
                    self.cache[kind] = pickle.load(f)
            else:
                self.cache[kind] = {}

    def save_cache(self) -> None:
        for kind, path in self.path_cache.items():
            with open(path, "wb") as f:
                pickle.dump(self.cache[kind], f, protocol=pickle.HIGHEST_PROTOCOL)
            if self.count % 100 == 0:
                with open(path + str(self.count), "wb") as f:
                    pickle.dump(self.cache[kind], f, protocol=pickle.HIGHEST_PROTOCOL)

    # ── LLM dispatch ──────────────────────────────────────────────────────

    def _generate_llm_response(
        self,
        prompt: str,
        instruction: str = "",
        stop: list[str] = [],
    ) -> str | None:
        """
        Send a prompt to the configured LLM engine and return the response.

        Supports: claude-*, gpt-*, gemini-*, llama-*
        Cost tracking is done for OpenAI models.
        """
        full_prompt = prompt + instruction
        try:
            response, cost = call_llm(
                prompt=full_prompt,
                engine=self.engine,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=stop,
            )
            self.total_cost += cost
            self.count += 1
            return response
        except Exception as e:
            print(f"[Pipeline] LLM error: {e}")
            return None

    def _generate_thought(self, prompt: str) -> str:
        """Optionally prepend a CoT 'Thought:' step (when method contains 'cot')."""
        if "cot" in self.method:
            thought = self._generate_llm_response(prompt, "\nThought:")
            if thought:
                print(f"\nThought: {thought}")
                return thought
        return ""

    # ── Goal / fact / rule generation ────────────────────────────────────

    def gen_predict_state_response(self, instruction: str, kind: str = "goal") -> str:
        """
        NL instruction → ASP goal clause.

        The default wraps the LLM output in `goal(T) :- ...`.
        Override post_process_goal() for domain-specific fixups.
        """
        prompt = f"{self.prompt[kind]}{instruction.strip()}"
        response = self._generate_llm_response(prompt, "\nSemantic Parse:")
        goal_asp = f"goal(T) :- {response}"
        return self.post_process_goal(goal_asp)

    def post_process_goal(self, goal_asp: str) -> str:
        """
        Hook for domain-specific goal fixups (string replacements, etc.).
        Override in subclasses.
        """
        return goal_asp

    def gen_adapt_fact_response(
        self, observations: str, kind: str = "adapt_fact"
    ) -> tuple[list, str]:
        """
        Trajectory observations → new ASP facts.

        Returns ([], added_facts_str) — the empty list is kept for API
        compatibility with the Alfworld version.
        """
        prompt = f"{self.prompt[kind]}\n\nEnvironment Trajectory:\n{observations}"
        thought = self._generate_thought(prompt)
        if thought:
            prompt += f"\nThought: {thought}"
        response = self._generate_llm_response(prompt, "\nSemantic Parse:\n")
        added_facts = response.split("Semantic Parse:")[-1]
        print(f"\n{'New Facts':=^40}\n{added_facts}\n{'':=^40}")
        return [], added_facts

    def gen_adapt_rule_response(
        self, observations: str, kind: str = "adapt_rule"
    ) -> str:
        """
        Trajectory observations → updated/refined ASP rules.

        When Clingo finds 0 stable models the caller should increment its
        counter and call this again; the chat history accumulates the
        '0 Stable models' feedback automatically.

        TODO: plug in HI-score feedback (see get_HI_score) for smarter
        rule selection based on true/false positive rates.
        """
        prompt = f"{self.prompt[kind]}\n\nEnvironment Trajectory:\n{observations}"

        if self.adaptation_chat_history == "":
            self.adaptation_chat_history = prompt
        else:
            self.adaptation_chat_history += (
                "\nFeedback: The Program you created has 0 Stable models. "
                "Identify what went wrong, improve it, and create it again.\n"
            )

        raw = self._generate_llm_response(self.adaptation_chat_history) or ""
        raw = raw.replace("```asp", "").replace("```", "").replace("prolog", "")
        print(f"\n{'New Rules':=^40}\n{raw}\n{'':=^40}")
        return rule_filtering(raw)

    # ── ILP generalisation via LLM ────────────────────────────────────────

    def gen_general_fact_response(self, example: dict, kind: str = "general_fact") -> str:
        """Trajectory + target predicate → LLM-generated fact set."""
        prompt = (
            f"{self.prompt[kind]}\n"
            f"External Trajectory: {example['traj']}\n"
            f"Target Predicate: {example['target_predicate']}"
        )
        thought = self._generate_thought(prompt)
        if thought:
            prompt += f"\nThought:{thought}"
        return self._generate_llm_response(prompt, "\nFact set: ") or ""

    def gen_ilp_bk_response(self, examples: dict, kind: str = "general_bk") -> str:
        """Positive/negative examples → background knowledge (ASP)."""
        prompt = f"{self.prompt[kind]}\nPositive/Negative Examples:\n{examples['pn_e']}"
        return self._generate_llm_response(prompt, "Make Background knowledge.\n") or ""

    def gen_ilp_rule_response(self, examples: dict, kind: str = "general_rule") -> str:
        """
        Examples + BK + target predicate → ASP rules.

        Handles both 'effect' (positive rules) and 'precondition'/'openable'
        (constraint rules) action types.
        """
        action_type = examples["action_type"]
        pn_e = examples["pn_e"]
        bk = examples["bk"]
        target_predicate = examples["target_predicate"]
        rule_label = ""

        if "effect" in action_type:
            rule_label = "Positive Rules"
        elif "precondition" in action_type or "openable" in action_type:
            rule_label = "Constraint Rules"
        else:
            return ""

        prompt = (
            f"{self.prompt[kind]}\n"
            f"Positive/Negative Examples:\n{pn_e}\n\n"
            f"Background Knowledge:\n{bk}\n\n"
            f"Target Predicate:\n{target_predicate}\n\n"
        ).replace("XXXXX", rule_label)

        thought = self._generate_thought(prompt)
        if thought:
            prompt += f"\nThought:{thought}"

        response = self._generate_llm_response(prompt, f"Make {rule_label}.\n") or ""
        return extract_rule(response)

    def generalize_external_traj(self, save: bool = True) -> str:
        """
        Full ILP induction pipeline over an external trajectory dataset.

        Expects self.get_target_predicates() to return a dict mapping
        action_type -> target_predicate, and dataset files at
        ./data/demo/{action}_external_data.json.

        Override get_external_data_path() to point elsewhere.
        """
        import json

        rule_sets = ""
        for action_type, target_predicate in self.get_target_predicates().items():
            action = action_type.split("_")[0]
            path = self.get_external_data_path(action)
            with open(path, "r") as f:
                dataset = json.load(f)

            p_examples, n_examples = [], []
            external_trajs = [
                (d["episode"], d["positive"])
                for d in dataset
                if d["action_type"] == action_type
            ]
            for traj, positive in external_trajs:
                example = {
                    "action_type": action_type,
                    "traj": traj,
                    "target_predicate": target_predicate,
                }
                pn_e = self.gen_general_fact_response(example)
                (p_examples if positive == "true" else n_examples).append(
                    pn_e.split("\n")[0]
                )

            pn_str = "".join(f"Positive: {e}\n" for e in p_examples)
            pn_str += "".join(f"Negative: {e}\n" for e in n_examples)

            examples = {
                "action_type": action_type,
                "pn_e": pn_str,
                "target_predicate": target_predicate,
            }
            bk = self.gen_ilp_bk_response(examples)
            print(f"#### BK: {action_type} ####\n{bk}")

            examples["bk"] = bk
            rule_set = self.gen_ilp_rule_response(examples)
            print(f"#### ILP rule set: {action_type} ####\n{rule_set}")
            rule_sets += rule_set + "\n"

        rules = sanitize_asp(rule_sets)
        rules = keep_only_parseable_rules(rules)
        if save:
            save_generalized_rules(rules, self.rule_save_path)
        return rules

    # ── Symbolic reasoning ────────────────────────────────────────────────

    def eval_single_plan(
        self, example: dict, opt: bool = False, to_print: bool = False
    ) -> list[list[str]]:
        """
        Process one step of a LLM+ASP episode.

        'example' is a dict with some subset of these keys:
            'goal'        -> NL instruction, triggers goal parsing
            'adapt_fact'  -> NL trajectory, triggers fact update
            'adapt_rule'  -> (traj1, traj2), triggers rule refinement loop

        Returns the list of answer sets from Clingo.
        """
        cnt = 0
        answer_sets: list[list[str]] = []

        for kind in example:
            if kind == "goal":
                self.goal_state = self.gen_predict_state_response(example[kind], kind) + "\n"
                print(f"\n{'goal':=^40}\n{self.goal_state}\n{'':=^40}")
                program = (
                    self.asp_program
                    + self.init_state + "\n"
                    + self.goal_state + "\n\n"
                    + self.adapted_rules + "\n"
                )
                answer_sets = gen_answer_set(program, self.clingo_seed, opt=opt)

            elif kind == "adapt_fact":
                new_facts = "\n".join(self.parse_observation(example[kind]))
                self.dynamic_facts += "\n" + new_facts
                if to_print:
                    print(f"\n{'new facts':=^40}\n{new_facts}\n{'':=^40}")
                cur_state = self.init_state + "\n" + self.dynamic_facts
                program = (
                    self.asp_program + "\n"
                    + cur_state + "\n"
                    + self.goal_state + "\n\n"
                    + self.adapted_rules + "\n"
                )
                answer_sets = gen_answer_set(program, self.clingo_seed, opt=opt)

            elif kind == "adapt_rule":
                traj1, traj2 = example[kind]
                new_facts = "\n".join(self.parse_observation(traj1))
                self.dynamic_facts += "\n" + new_facts
                if to_print:
                    print(f"\n{'new facts':=^40}\n{new_facts}\n{'':=^40}")
                cur_state = self.init_state + "\n" + self.dynamic_facts

                while len(answer_sets) == 0 and cnt < MAX_VERIFICATION_TRIAL:
                    self.adapted_rules = self.gen_adapt_rule_response(traj2, kind)
                    program = (
                        self.asp_program
                        + cur_state + "\n"
                        + self.goal_state + "\n\n"
                        + self.adapted_rules + "\n"
                    )
                    answer_sets = gen_answer_set(program, self.clingo_seed, opt=opt)
                    cnt += 1

                self.adaptation_chat_history = ""

        return answer_sets

    # ── HI-score utility (for advanced rule refinement) ───────────────────

    def get_HI_score(
        self, tp: int, tn: int, fp: int, fn: int, alpha: float = 0.5
    ) -> float:
        """
        Harmonic Informativeness score.
        TPR weighted against FPR — use to guide rule refinement prompts.

        TODO: wire into gen_adapt_rule_response() for feedback-driven
        rule selection.
        """
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        return alpha * tpr + (1 - alpha) * fpr

    # ── Abstract interface  ───────────────────────────────────────────────

    @abstractmethod
    def parse_observation(self, text: str) -> list[str]:
        """
        Convert a natural-language observation string into a list of ASP fact strings.
        e.g. "Act 0: take apple from table\nObs 1: You picked up apple 1."
             -> ["action(pick_up(apple_1, table_1), 0).", "holding(apple_1, 1)."]
        """

    @abstractmethod
    def get_target_predicates(self) -> dict[str, str]:
        """
        Return {action_type -> target_predicate} for ILP generalisation.
        Used by generalize_external_traj().
        """

    def get_external_data_path(self, action: str) -> str:
        """Override to point to domain-specific trajectory data."""
        return f"./data/demo/{action}_external_data.json"



class PipelineQA(Pipeline):
    @abstractmethod
    def extract_answer( self, answer_sets):
        if(not (answer_sets)):
           return "" 
        for x in answer_sets[0]:
            if x.startswith("acceptable_answer("):
                return x.split('"')[1]
        return ""
        
    def run(self, item: dict):
        # 1. facts → init_state
        self.init_state = "\n".join(item["asp_facts"])
        # 2. LLM → goal_state (ASP query)

        self.goal_state = self.gen_predict_state_response(item["question"])
        
        # 3. clingo → answer_sets
        program = self.asp_program + "\n" + self.init_state + "\n" + self.init_state = "\n" = self.goal_state
        answer_sets = gen_answer_set

        # 4. parse answer_sets → prediction
        return self.extract_answer(answer_sets)






class AmbigQADirectPipeline(PipelineQA):
    def run(self, item: dict) -> str:
        # skip ASP entirely, just ask the LLM
        response = self._generate_llm_response(item["question"])
        return response.strip()
    
    def extract_answer(self, answer_sets):
        return # not used



    




        

    