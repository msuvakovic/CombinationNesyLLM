"""
nesy_core/prompts/loader.py

Lightweight prompt loading and templating helpers.

Prompt files are plain .txt files.  Template variables use {VARNAME}
syntax so Python's str.format_map() can be applied.

Usage:
    store = PromptStore("./nesy_core/prompts/clevr")
    store.load_all()

    prompt = store["goal"]                         # raw prompt string
    prompt = store.render("goal", question=q_str)  # with variable substitution
"""

from __future__ import annotations

from pathlib import Path


def load_prompt(path: str | Path) -> str:
    """Load a single prompt file and return its contents as a string."""
    return Path(path).read_text(encoding="utf-8")


def load_prompt_dir(directory: str | Path) -> dict[str, str]:
    """
    Load all .txt files in a directory as a {stem -> content} mapping.

    e.g. prompts/goal.txt  ->  {"goal": "<file contents>"}
    """
    prompts: dict[str, str] = {}
    for p in Path(directory).glob("*.txt"):
        prompts[p.stem] = p.read_text(encoding="utf-8")
    return prompts


class PromptStore:
    """
    A keyed store of prompt templates for a single domain.

    Parameters
    ----------
    prompt_dir : str | Path
        Directory containing .txt prompt files.

    Attributes
    ----------
    prompts : dict[str, str]
        Loaded prompt contents keyed by filename stem.
    """

    def __init__(self, prompt_dir: str | Path):
        self.prompt_dir = Path(prompt_dir)
        self.prompts: dict[str, str] = {}

    def load_all(self) -> "PromptStore":
        """Load every .txt file in prompt_dir into self.prompts."""
        self.prompts = load_prompt_dir(self.prompt_dir)
        return self

    def load(self, *kinds: str) -> "PromptStore":
        """Load only the specified prompt files (by stem name)."""
        for kind in kinds:
            path = self.prompt_dir / f"{kind}.txt"
            self.prompts[kind] = load_prompt(path)
        return self

    def __getitem__(self, kind: str) -> str:
        if kind not in self.prompts:
            raise KeyError(
                f"Prompt '{kind}' not loaded. "
                f"Available: {list(self.prompts.keys())}"
            )
        return self.prompts[kind]

    def render(self, kind: str, **kwargs) -> str:
        """
        Return the prompt with {VARNAME} placeholders filled in.

        e.g.  store.render("goal", question="How many red cubes?")
        """
        template = self[kind]
        return template.format_map(kwargs)

    def inject_into_pipeline(self, pipeline) -> None:
        """
        Copy all loaded prompts into a Pipeline object's self.prompt dict.
        Convenience shortcut so you don't have to call pipeline.load_prompt().
        """
        pipeline.prompt.update(self.prompts)
