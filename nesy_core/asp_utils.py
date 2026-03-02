"""
nesy_core/asp_utils.py

Clingo / Answer Set Programming utilities.

Ported from Alfworld/our_pipeline.py — fully domain-agnostic.

Key functions:
    sanitize_asp()              Strip LLM noise from ASP text
    keep_only_parseable_rules() Filter out lines Clingo can't parse
    setup_and_ground()          Build a Clingo control object
    gen_answer_set()            Solve and return answer sets
"""

from typing import Optional
import clingo
from clingo.control import Control
from clingo.symbol import parse_term


# ---------------------------------------------------------------------------
# Clingo context helper
# ---------------------------------------------------------------------------

class _Context:
    """Provides gen_feature/1 to Clingo for string-based term parsing."""

    def gen_feature(self, x):
        ret = []
        for term in str(x.string).split(" "):
            ret.append(parse_term(term))
        return ret


# ---------------------------------------------------------------------------
# ASP text cleaning
# ---------------------------------------------------------------------------

def sanitize_asp(text: str) -> str:
    """
    Strip markdown code fences and junk from LLM-generated ASP.

    Keeps only lines that are:
        - Comments   (start with %)
        - Facts      (end with .)
        - Rules      (contain :-)

    Returns a newline-terminated string ready for Clingo.
    """
    if not text:
        return ""

    text = (
        text.replace("```asp", "")
            .replace("```", "")
            .replace("prolog", "")
    )

    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("semantic parse"):
            continue
        if ":-" in line or line.endswith(".") or line.startswith("%"):
            # Ensure every rule/fact is properly terminated
            if not line.startswith("%") and not line.endswith("."):
                line += "."
            lines.append(line)

    return "\n".join(lines) + "\n"


def keep_only_parseable_rules(rules: str) -> str:
    """
    Incrementally validate ASP rules with Clingo.

    Tries to ground each line; silently drops any that cause parse errors.
    This is the key safety net between LLM output and the solver.
    """
    good: list[str] = []
    for line in rules.splitlines():
        trial = "\n".join(good + [line]) + "\n"
        try:
            ctl = clingo.Control(["0", "--warn=none"])
            ctl.add("base", [], trial)
            ctl.ground([("base", [])])
            good.append(line)
        except Exception:
            pass  # drop unparseable line silently
    return "\n".join(good) + "\n"


# ---------------------------------------------------------------------------
# Clingo solve interface
# ---------------------------------------------------------------------------

def setup_and_ground(
    program: str,
    seed: int = 1,
    threads: int = 16,
    models: int = 10,
) -> Control:
    """
    Build and ground a Clingo control object for the given ASP program.

    Args:
        program:  Full ASP program string.
        seed:     Random seed for Clingo solver.
        threads:  Number of solver threads (parallelism).
        models:   Max answer sets to enumerate.

    Returns:
        A grounded Clingo Control object ready for .solve().
    """
    context = _Context()
    ctl = Control(
        ["0", "--warn=none", "--opt-mode=optN", f"--seed={seed}", "-t", str(threads)]
    )
    ctl.configuration.solve.models = models
    ctl.add("base", [], program)
    ctl.ground([("base", [])], context=context)
    return ctl


def gen_answer_set(
    program: str,
    seed: int = 1,
    opt: bool = False,
    threads: int = 16,
    models: int = 10,
) -> list[list[str]]:
    """
    Solve an ASP program and return all (or optimal) answer sets.

    Args:
        program: Full ASP program string (base theory + facts + rules + goal).
        seed:    Clingo random seed.
        opt:     If True, return only optimality-proven models
                 (requires weak constraints in the program).
        threads: Solver threads.
        models:  Max models to collect.

    Returns:
        List of answer sets, each answer set is a list of atom strings.
        Returns [] if the program is unsatisfiable or an error occurs.
    """
    collected: list[list[str]] = []

    try:
        ctl = setup_and_ground(program, seed=seed, threads=threads, models=models)
    except Exception as e:
        print(f"[asp_utils] Ground error: {e}")
        return []

    def on_model(model):
        atoms = model.symbols(atoms=True, shown=True)
        if opt and not model.optimality_proven:
            return
        collected.append([str(a) for a in atoms])

    ctl.solve(on_model=on_model)
    return collected


def check_satisfiable(program: str, seed: int = 1) -> bool:
    """Quick satisfiability check — returns True if at least one model exists."""
    return len(gen_answer_set(program, seed=seed, models=1)) > 0
