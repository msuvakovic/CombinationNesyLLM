"""
ROAD-R Shortcut Detection

A reasoning shortcut occurs when learned ASP rules satisfy logical requirements
via unintended predicate combinations — e.g., a rule that works in Phase 1 because
"car" co-occurs with "veh_lane" by chance, but fails when "bus" (a new agent type)
appears in "veh_lane" in Phase 2.

Detection Strategy
──────────────────
1. Concept Drift Score
   For each agent/action/location type, measure how often the learned rules
   fire correctly vs. what ground-truth annotations say.
   A label whose accuracy drops sharply at a phase boundary is a shortcut candidate.

2. Predicate Co-occurrence Bias
   If a rule fires predominantly when two specific labels co-occur (beyond what
   requirements mandate), it may be exploiting spurious correlation.
   Measured as: P(rule fires | label_A, label_B) >> P(rule fires | label_A alone)

3. Cross-Phase Generalisation Gap
   Compare requirement-violation rate on old-phase facts with new-phase rules
   vs. old-phase rules. A large gap means new rules broke old knowledge.

4. Constraint Coverage
   What fraction of the 243 requirements are "covered" (appear to be enforced)
   by the learned rules? Rules that cover very few constraints are likely shortcuts.
"""

import json
import re
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from ROAD.requirements import (
    CORE_REQUIREMENTS,
    ROAD_LABELS,
    count_violations,
    load_dimacs_requirements,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Concept Drift Score
# ─────────────────────────────────────────────────────────────────────────────

def concept_drift_score(
    phase_rules: Dict[int, str],
    phase_facts: Dict[int, str],
    dimacs_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    For each phase transition, compute the requirement violation rate when
    applying rules from phase N to facts from phase N+1.

    A high score at transition N→N+1 indicates shortcuts acquired in phase N
    that don't generalise.

    Args:
        phase_rules: {phase → learned ASP rules string}
        phase_facts: {phase → ASP facts string}
        dimacs_path: path to DIMACS requirements file

    Returns:
        {transition_label → violation_rate}
    """
    scores = {}
    phases = sorted(phase_rules.keys())

    for i in range(len(phases) - 1):
        p_src = phases[i]       # rules trained in this phase
        p_tgt = phases[i + 1]   # facts from the next phase

        rules = phase_rules[p_src]
        facts = phase_facts.get(p_tgt, '')

        if not facts.strip():
            continue

        n_viol = count_violations(facts + '\n' + rules, dimacs_path)
        label = f"phase{p_src}→phase{p_tgt}"
        scores[label] = n_viol

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# 2. Predicate Co-occurrence Bias
# ─────────────────────────────────────────────────────────────────────────────

def _parse_facts_to_labels(facts: str) -> Dict[str, Dict[str, set]]:
    """
    Parse ASP facts into a structure:
      tube_id → {frame_id → set of labels}
    """
    tube_map: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
    pattern = re.compile(
        r'(?:agent|action|loc)\(([^,]+),\s*([^,]+),\s*(\d+),\s*([^)]+)\)\.'
    )
    for match in pattern.finditer(facts):
        tube, label, frame, _ = match.groups()
        tube_map[tube.strip()][frame.strip()].add(label.strip())
    return tube_map


def cooccurrence_bias(
    facts: str,
    rules: str,
    label_a: str,
    label_b: str,
) -> float:
    """
    Measure P(label_b | label_a) in the given facts.
    Values close to 1.0 indicate label_b is nearly always present when label_a is,
    which is a potential shortcut trigger.

    Args:
        facts: ASP facts string
        rules: learned ASP rules (not used for counting, but kept for context)
        label_a: anchor label (e.g., 'car')
        label_b: potentially spuriously correlated label (e.g., 'veh_lane')

    Returns:
        co-occurrence probability P(label_b | label_a)
    """
    tube_map = _parse_facts_to_labels(facts)
    count_a = 0
    count_ab = 0

    for tube, frames in tube_map.items():
        for frame, labels in frames.items():
            if label_a in labels:
                count_a += 1
                if label_b in labels:
                    count_ab += 1

    if count_a == 0:
        return 0.0
    return count_ab / count_a


def top_cooccurrence_biases(
    facts: str,
    rules: str,
    top_k: int = 10,
) -> List[Tuple[str, str, float]]:
    """
    Find the top-k most biased label co-occurrences in the given facts.
    These are shortcut candidates.

    Returns list of (label_a, label_b, probability) sorted by probability descending.
    """
    labels = list(ROAD_LABELS.values())
    biases = []

    for i, la in enumerate(labels):
        for lb in labels[i + 1:]:
            p = cooccurrence_bias(facts, rules, la, lb)
            if p > 0.7:  # only report high-bias pairs
                biases.append((la, lb, round(p, 4)))

    biases.sort(key=lambda x: x[2], reverse=True)
    return biases[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Cross-Phase Generalisation Gap
# ─────────────────────────────────────────────────────────────────────────────

def generalisation_gap(
    phase_rules: Dict[int, str],
    phase_facts: Dict[int, str],
    dimacs_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    For each phase N > 1, compare:
      - violation rate of phase-N rules on phase-N facts (in-phase)
      - violation rate of phase-N rules on phase-1 facts (backward compat)
      - violation rate of phase-1 rules on phase-N facts (forward compat)

    A large forward-compat gap means the new labels triggered shortcuts.
    A large backward-compat gap means the new rules forgot old knowledge.

    Returns a dict with all gap metrics.
    """
    results = {}
    base_rules = phase_rules.get(1, '')
    base_facts = phase_facts.get(1, '')

    for phase in sorted(phase_rules.keys()):
        new_rules = phase_rules[phase]
        new_facts = phase_facts.get(phase, '')

        # In-phase: new rules on new facts
        in_phase = count_violations(new_facts + '\n' + new_rules, dimacs_path)

        if phase > 1 and base_facts and base_rules:
            # Backward: new rules on old facts
            backward = count_violations(base_facts + '\n' + new_rules, dimacs_path)
            # Forward: old rules on new facts
            forward = count_violations(new_facts + '\n' + base_rules, dimacs_path)

            results[f'phase{phase}_in_phase_violations']   = in_phase
            results[f'phase{phase}_backward_compat_gap']   = max(0, backward - in_phase)
            results[f'phase{phase}_forward_compat_gap']    = max(0, forward - in_phase)
        else:
            results[f'phase{phase}_in_phase_violations'] = in_phase

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. Constraint Coverage
# ─────────────────────────────────────────────────────────────────────────────

def _rules_mention_label(rules: str, label: str) -> bool:
    """Return True if the ASP rules string mentions the given label atom."""
    return label in rules


def constraint_coverage(
    rules: str,
    dimacs_path: Optional[str] = None,
) -> float:
    """
    Compute what fraction of requirement labels are explicitly referenced
    in the learned rules. Low coverage = rules are likely shortcuts that
    ignore important constraint structure.

    Returns fraction in [0, 1].
    """
    clauses = (
        load_dimacs_requirements(dimacs_path)
        if dimacs_path
        else CORE_REQUIREMENTS
    )

    # Collect all label indices mentioned in requirements
    req_labels = set()
    for clause in clauses:
        for lit in clause:
            req_labels.add(ROAD_LABELS.get(abs(lit), ''))

    req_labels.discard('')
    if not req_labels:
        return 0.0

    covered = sum(1 for label in req_labels if _rules_mention_label(rules, label))
    return round(covered / len(req_labels), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Unified Shortcut Report
# ─────────────────────────────────────────────────────────────────────────────

def shortcut_report(
    phase_rules: Dict[int, str],
    phase_facts: Dict[int, str],
    dimacs_path: Optional[str] = None,
) -> Dict:
    """
    Run all shortcut detection metrics and return a unified report.

    Args:
        phase_rules: {phase → learned ASP rules string}
        phase_facts: {phase → ASP facts string}
        dimacs_path: path to DIMACS requirements file (optional)

    Returns:
        report dict with all metrics
    """
    report = {}

    # 1. Concept drift across phase boundaries
    print("Computing concept drift scores...")
    report['concept_drift'] = concept_drift_score(phase_rules, phase_facts, dimacs_path)

    # 2. Generalisation gap
    print("Computing generalisation gaps...")
    report['generalisation_gap'] = generalisation_gap(phase_rules, phase_facts, dimacs_path)

    # 3. Constraint coverage per phase
    print("Computing constraint coverage...")
    report['constraint_coverage'] = {
        f'phase{p}': constraint_coverage(rules, dimacs_path)
        for p, rules in phase_rules.items()
    }

    # 4. Top co-occurrence biases (use phase-1 facts as reference)
    phase1_facts = phase_facts.get(1, '')
    phase1_rules = phase_rules.get(1, '')
    if phase1_facts:
        print("Computing co-occurrence biases...")
        report['top_cooccurrence_biases'] = top_cooccurrence_biases(
            phase1_facts, phase1_rules, top_k=10
        )
    else:
        report['top_cooccurrence_biases'] = []

    # 5. Shortcut verdict
    drift = report['concept_drift']
    gap   = report['generalisation_gap']
    cov   = report['constraint_coverage']

    # Heuristic: shortcut likely if any transition has high drift + low coverage
    max_drift = max(drift.values(), default=0)
    max_gap   = max(
        (v for k, v in gap.items() if 'gap' in k),
        default=0
    )
    min_coverage = min(cov.values(), default=1.0)

    report['shortcut_risk'] = {
        'max_drift_violations': max_drift,
        'max_generalisation_gap': max_gap,
        'min_constraint_coverage': min_coverage,
        'verdict': (
            'HIGH' if (max_drift > 5 or max_gap > 5 or min_coverage < 0.2)
            else 'MEDIUM' if (max_drift > 2 or max_gap > 2 or min_coverage < 0.4)
            else 'LOW'
        ),
    }

    return report


def print_shortcut_report(report: Dict):
    """Pretty-print the shortcut detection report."""
    print("\n" + "="*60)
    print("  SHIELD SHORTCUT DETECTION REPORT")
    print("="*60)

    print("\n[1] Concept Drift (violations when applying old rules to new facts):")
    for transition, count in report.get('concept_drift', {}).items():
        print(f"    {transition}: {count} violations")

    print("\n[2] Generalisation Gap:")
    for metric, value in report.get('generalisation_gap', {}).items():
        print(f"    {metric}: {value}")

    print("\n[3] Constraint Coverage (fraction of requirement labels in rules):")
    for phase, cov in report.get('constraint_coverage', {}).items():
        bar = '█' * int(cov * 20) + '░' * (20 - int(cov * 20))
        print(f"    {phase}: [{bar}] {cov:.1%}")

    print("\n[4] Top Co-occurrence Biases (shortcut candidates):")
    for la, lb, prob in report.get('top_cooccurrence_biases', [])[:5]:
        print(f"    P({lb} | {la}) = {prob:.2%}")

    risk = report.get('shortcut_risk', {})
    verdict = risk.get('verdict', 'UNKNOWN')
    print(f"\n[VERDICT] Shortcut Risk: {verdict}")
    print(f"  Max drift violations: {risk.get('max_drift_violations', 0)}")
    print(f"  Max generalisation gap: {risk.get('max_generalisation_gap', 0)}")
    print(f"  Min constraint coverage: {risk.get('min_constraint_coverage', 0):.1%}")
    print("="*60 + "\n")
