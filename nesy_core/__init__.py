"""
nesy_core — Generalised LLM + Neuro-Symbolic pipeline toolkit.

Extracted and decoupled from the ALFWorld NeSy+LLM codebase.
Supports any domain that can be expressed in Answer Set Programming.

Quick start
-----------
from nesy_core import Pipeline, CLEVRDataset
from nesy_core.asp_utils import gen_answer_set, sanitize_asp
from nesy_core.retrieval import RetrievalEngine
from nesy_core.semantic_parser import CLEVRSemanticParser, AlfworldSemanticParser
from nesy_core.prompts import PromptStore

Modules
-------
pipeline        Core LLM+ASP Pipeline base class
llm_utils       Multi-provider LLM dispatch (Claude, GPT, Gemini, LLaMA)
asp_utils       Clingo wrapper: sanitize, validate, solve ASP programs
retrieval       KNN retrieval engine for few-shot in-context learning
semantic_parser NL → ASP conversion (Alfworld + CLEVR parsers included)
datasets        Dataset adapters (base class + CLEVR)
prompts         Prompt file loading and templating utilities
"""

from .pipeline import Pipeline
from .llm_utils import call_llm
from .asp_utils import sanitize_asp, keep_only_parseable_rules, gen_answer_set
from .retrieval import RetrievalEngine
from .semantic_parser import (
    SemanticParser,
    AlfworldSemanticParser,
    CLEVRSemanticParser,
)
from .datasets import NeSyDataset, CLEVRDataset
from .prompts import PromptStore

__all__ = [
    "Pipeline",
    "call_llm",
    "sanitize_asp",
    "keep_only_parseable_rules",
    "gen_answer_set",
    "RetrievalEngine",
    "SemanticParser",
    "AlfworldSemanticParser",
    "CLEVRSemanticParser",
    "NeSyDataset",
    "CLEVRDataset",
    "PromptStore",
]
