"""
nesy_core/llm_utils.py

Multi-provider LLM dispatch with LangChain SQLite response caching.

Supported engines (engine string -> provider):
    claude-*       -> Anthropic (via langchain-anthropic)
    gpt-*          -> OpenAI    (via langchain-community)
    gemini-*       -> Google    (via langchain-google-genai)
    llama-* / meta-* -> Replicate (via replicate SDK)

All functions return (response_str, cost_float).
Cost is only tracked for OpenAI; others return 0.0.

Usage:
    from nesy_core.llm_utils import call_llm

    response, cost = call_llm(
        prompt="Your prompt here",
        engine="claude-sonnet-4-6",
        temperature=0.0,
        max_tokens=1024,
    )
"""

import os
import time
from typing import Optional

# ── LangChain caching (shared across all providers) ───────────────────────────
from langchain.globals import set_llm_cache
from langchain.cache import SQLiteCache

set_llm_cache(SQLiteCache(database_path=".langchain_nesy.db"))

# ── API keys (pulled from environment) ────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY", "")
GOOGLE_API_KEY    = os.environ.get("GOOGLE_API_KEY", "")
REPLICATE_API_KEY = os.environ.get("REPLICATE_API_TOKEN", "")


# ── Provider implementations ──────────────────────────────────────────────────

def _claude_llm(
    prompt: str,
    model_name: str = "claude-sonnet-4-6",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    stop: list[str] = [],
) -> tuple[str, float]:
    """Anthropic Claude via langchain-anthropic."""
    from langchain_anthropic import ChatAnthropic

    llm = ChatAnthropic(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        anthropic_api_key=ANTHROPIC_API_KEY,
        stop=stop or None,
    )
    response = llm.invoke(prompt).content
    # Anthropic doesn't expose cost via LangChain callback yet
    return response, 0.0


def _openai_llm(
    prompt: str,
    model_name: str = "gpt-4o",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    stop: list[str] = [],
) -> tuple[str, float]:
    """OpenAI GPT via langchain-community with cost tracking."""
    from langchain_community.chat_models import ChatOpenAI
    from langchain_community.callbacks import get_openai_callback

    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_key=OPENAI_API_KEY,
        model_kwargs={"stop": stop} if stop else {},
    )
    with get_openai_callback() as cb:
        response = llm.invoke(prompt).content
    return response, cb.total_cost


def _google_llm(
    prompt: str,
    model_name: str = "gemini-1.5-flash",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    stop: list[str] = [],
) -> tuple[str, float]:
    """Google Gemini via langchain-google-genai with retry."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    for attempt in range(10):
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                max_output_tokens=max_tokens,
                google_api_key=GOOGLE_API_KEY,
            )
            response = llm.invoke(prompt, stop=stop).content
            return response, 0.0
        except Exception as e:
            print(f"[google_llm] attempt {attempt + 1} failed: {e}")
            time.sleep(2)
    return "", 0.0


def _meta_llm(
    prompt: str,
    model_name: str = "meta/meta-llama-3-70b",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    stop: list[str] = [],
) -> tuple[str, float]:
    """Meta LLaMA via Replicate streaming API."""
    import replicate

    result = ""
    for event in replicate.stream(
        model_name,
        input={
            "prompt": prompt,
            "temperature": temperature,
            "min_tokens": max_tokens,
            "presence_penalty": 1.15,
        },
    ):
        result += str(event)
    return result, 0.0


# ── Unified dispatcher ────────────────────────────────────────────────────────

def call_llm(
    prompt: str,
    engine: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    stop: list[str] = [],
) -> tuple[str, float]:
    """
    Route a prompt to the correct provider based on the engine string.

    Returns (response, cost).  Cost is 0.0 for all providers except OpenAI.

    Examples:
        call_llm(prompt, "claude-sonnet-4-6")
        call_llm(prompt, "gpt-4o")
        call_llm(prompt, "gemini-1.5-flash")
        call_llm(prompt, "meta/meta-llama-3-70b")
    """
    kwargs = dict(
        prompt=prompt,
        model_name=engine,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop,
    )

    if engine.startswith("claude"):
        return _claude_llm(**kwargs)
    elif engine.startswith("gpt"):
        return _openai_llm(**kwargs)
    elif engine.startswith("gemini"):
        return _google_llm(**kwargs)
    elif "llama" in engine or "meta" in engine:
        return _meta_llm(**kwargs)
    else:
        raise ValueError(
            f"Unknown engine '{engine}'. "
            "Prefix must be one of: claude-, gpt-, gemini-, llama-/meta-"
        )
