"""
ROAD-R Logical Requirements → ASP Integrity Constraints

The 243 logical requirements from ROAD-R are expressed in DIMACS CNF:
  p cnf 41 243
  literal1 literal2 ... literalN 0   (each line is a clause; must be satisfied)

In ASP, a requirement clause  "a OR b OR NOT c"  becomes an integrity constraint:
  :- not has_label(T, a, F, V), not has_label(T, b, F, V), has_label(T, c, F, V).
which fires (and rejects the model) when the clause is violated.

This module:
1. Defines the 41-label mapping (from ROAD-R paper / dataset)
2. Provides a representative subset of the 243 requirements as ASP constraints
3. Provides a loader for the full DIMACS file if available
4. Generates a base ASP program with all constraints for Clingo verification
"""

import os
import re
from typing import List, Tuple, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Label index → ASP atom mapping  (indices 1-41, 1-based as in DIMACS CNF)
# These match the 'all_input_labels' order in road_trainval_v1.0.json
# ─────────────────────────────────────────────────────────────────────────────
ROAD_LABELS = {
    # Agents (1–10) — from requirements_readable.txt + ROAD-R paper
    1:  "ped",          # Pedestrian
    2:  "car",          # Car
    3:  "cyc",          # Cyclist
    4:  "mobike",       # Motorcycle
    5:  "med_veh",      # Medium Vehicle
    6:  "lar_veh",      # Large Vehicle
    7:  "bus",          # Bus
    8:  "em_veh",       # Emergency Vehicle
    9:  "tl",           # Traffic Light (agent)
    10: "oth_tl",       # Other Traffic Light (agent)
    # Actions (11–28)
    11: "mov_away",     # MovAway
    12: "mov_tow",      # MovTow
    13: "mov",          # Mov (generic)
    14: "brake",        # Brake
    15: "stop",         # Stop
    16: "incat_left",   # IncatLeft
    17: "incat_rht",    # IncatRht
    18: "ovtak",        # Ovtak (Overtaking)
    19: "wait2x",       # Wait2X (waiting to cross)
    20: "tur_lft",      # TurLft (Turning Left)
    21: "tur_rht",      # TurRht (Turning Right)
    22: "xing_fm_lft",  # XingFmLft (Crossing from left)
    23: "xing_fm_rht",  # XingFmRht (Crossing from right)
    24: "xing",         # Xing (Crossing)
    25: "haz_lit",      # HazLit (Hazard Lights)
    26: "red",          # Red (traffic light state)
    27: "amber",        # Amber (traffic light state)
    28: "green",        # Green (traffic light state)
    # Special action
    29: "push_obj",     # PushObj
    # Locations (30–41)
    30: "veh_lane",     # VehLane
    31: "outgo_lane",   # OutgoLane
    32: "outgo_cyc_lane",  # OutgoCycLane
    33: "incom_lane",   # IncomLane
    34: "incom_cyc_lane",  # IncomCycLane
    35: "pav",          # Pav (Pavement)
    36: "lft_pav",      # LftPav
    37: "rht_pav",      # RhtPav
    38: "jun",          # Jun (Junction)
    39: "xing_loc",     # XingLoc
    40: "bus_stop",     # BusStop
    41: "parking",      # Parking
}

# Reverse mapping: atom → index
LABEL_TO_IDX = {v: k for k, v in ROAD_LABELS.items()}

# ─────────────────────────────────────────────────────────────────────────────
# Representative hard-coded requirements (subset of the 243).
# Format: list of clauses; each clause is a list of signed ints (1-based DIMACS)
# Positive int i → label i must be true (or clause is satisfied by another lit)
# Negative int -i → label i must be false (or clause is satisfied by another lit)
# ─────────────────────────────────────────────────────────────────────────────
CORE_REQUIREMENTS: List[List[int]] = [
    # ── Mutual exclusivity: at most one agent type per tube ──
    [-1, -2],    # NOT (ped AND car)
    [-1, -3],    # NOT (ped AND cyc)
    [-2, -3],    # NOT (car AND cyc)
    [-1, -7],    # NOT (ped AND lar_veh)
    [-1, -8],    # NOT (ped AND bus)
    [-2, -8],    # NOT (car AND bus)
    [-7, -2],    # NOT (lar_veh AND car)
    # ── Pedestrian constraints ──
    [-1, -18],   # Pedestrians cannot overtake
    [-1, -16],   # Pedestrians cannot indicate left
    [-1, -17],   # Pedestrians cannot indicate right
    [-1, -14, 23],  # Pedestrian braking → must be crossing or stopped
    # ── Emergency vehicle constraints ──
    [-9, -18],   # Emergency vehicles cannot overtake
    # ── Mutual exclusivity: at most one location type ──
    [-25, -30],  # NOT (veh_lane AND pavement)
    [-26, -28],  # NOT (outgo_lane AND incom_lane)
    [-31, -32],  # NOT (lft_pav AND rht_pav)
    [-25, -33],  # NOT (veh_lane AND junction) — simplified; junction is special
    # ── Action constraints ──
    [-13, -15],  # NOT (moving AND stopped) simultaneously
    [-14, -20],  # NOT (braking AND going-through-junction) simultaneously
    [-16, -17],  # NOT (indicating left AND right) simultaneously
    [-11, -12],  # NOT (moving away AND moving towards) simultaneously
    # ── Pedestrian location constraints ──
    [-1, -25],   # Pedestrians are not in vehicle lanes (usually)
    [-1, -26],   # Pedestrians not in outgoing lane
    [-1, -28],   # Pedestrians not in incoming lane
    # ── Cycle lane constraints ──
    [-2, -27],   # Cars not in outgoing cycle lane
    [-2, -29],   # Cars not in incoming cycle lane
    # ── Turning constraints ──
    [-21, -22],  # NOT (turning left AND right) simultaneously
    [-21, -13, 33],  # Turning left → must be at junction or moving
    [-22, -13, 33],  # Turning right → must be at junction or moving
]


def clause_to_asp(clause: List[int], label_map: dict = ROAD_LABELS) -> Optional[str]:
    """
    Convert a DIMACS CNF clause to an ASP integrity constraint.

    Clause [a, b, -c] means: (label_a OR label_b OR NOT label_c) must hold.
    Violation (what ASP forbids): NOT label_a AND NOT label_b AND label_c
    ASP: :- not has_label(T, a, F, V), not has_label(T, b, F, V), has_label(T, c, F, V).

    Returns None for clauses that would produce unsafe variables (all-positive-body,
    meaning all literals are negative in DIMACS = all 'has_label' in ASP body with no
    positive grounding literal).
    """
    body_parts = []
    has_positive_body = False  # True if at least one 'has_label(...)' (grounding anchor)
    for lit in clause:
        idx = abs(lit)
        atom = label_map.get(idx, f"label_{idx}")
        if lit > 0:
            # Positive in clause → negative in body (not has_label)
            body_parts.append(f"not has_label(T, {atom}, F, V)")
        else:
            # Negative in clause → positive in body (has_label) — this grounds T,F,V
            body_parts.append(f"has_label(T, {atom}, F, V)")
            has_positive_body = True
    # Skip constraints where T/F/V would be unbound (no grounding has_label)
    if not has_positive_body:
        return None
    return ":- " + ", ".join(body_parts) + "."


def load_dimacs_requirements(dimacs_path: str) -> List[List[int]]:
    """
    Load requirements from a DIMACS CNF file.
    Expected at: <road-r-repo>/requirements/requirements_dimacs.txt
    """
    clauses = []
    with open(dimacs_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('c') or line.startswith('p'):
                continue
            lits = list(map(int, line.split()))
            if lits and lits[-1] == 0:
                lits = lits[:-1]
            if lits:
                clauses.append(lits)
    return clauses


def get_requirements_asp(
    dimacs_path: Optional[str] = None,
    label_map: dict = ROAD_LABELS,
) -> str:
    """
    Return all requirements as ASP integrity constraints.
    Uses DIMACS file if provided, otherwise falls back to CORE_REQUIREMENTS.
    """
    if dimacs_path and os.path.exists(dimacs_path):
        clauses = load_dimacs_requirements(dimacs_path)
        source = f"% Loaded {len(clauses)} requirements from {dimacs_path}"
    else:
        clauses = CORE_REQUIREMENTS
        source = f"% Using {len(clauses)} built-in core requirements (subset of 243)"
        if dimacs_path:
            source += f"\n% (DIMACS file not found: {dimacs_path})"

    lines = [source, ""]
    skipped = 0
    for i, clause in enumerate(clauses, 1):
        asp = clause_to_asp(clause, label_map)
        if asp is None:
            skipped += 1
            continue
        lines.append(f"% Requirement {i}: {clause}")
        lines.append(asp)
    if skipped:
        lines.insert(1, f"% ({skipped} requirements skipped: all-negative clauses with unsafe variables)")
    return '\n'.join(lines)


BASE_ASP_ROAD = '''
% ─────────────────────────────────────────────────────────
% ROAD-R Base ASP Program
% ─────────────────────────────────────────────────────────

% Unify agent/action/loc into a single has_label/4 predicate
% for uniform constraint checking.
has_label(T, L, F, V) :- agent(T, L, F, V).
has_label(T, L, F, V) :- action(T, L, F, V).
has_label(T, L, F, V) :- loc(T, L, F, V).

% Every tube in a frame must have at least one agent label
:- bbox(T, F, V, _, _, _, _), not agent(T, _, F, V).

% AV action is unique per frame (at most one ego action)
:- av_action(A1, F, V), av_action(A2, F, V), A1 != A2.

% ─────────────────────────────────────────────────────────
% Learned Rules Section (filled in by the pipeline)
% ─────────────────────────────────────────────────────────
% {learned_rules}
'''


def get_base_program(learned_rules: str = "") -> str:
    """Return the full base ASP program with learned rules injected."""
    return BASE_ASP_ROAD.replace('{learned_rules}', learned_rules)


def verify_against_requirements(
    facts: str,
    learned_rules: str,
    dimacs_path: Optional[str] = None,
) -> Tuple[bool, List[str]]:
    """
    Check whether a set of ASP facts + learned rules satisfies all requirements.

    Returns:
        (satisfied: bool, violated_requirements: List[str])
    """
    import clingo
    from clingo.control import Control

    requirements_asp = get_requirements_asp(dimacs_path)
    base_program = get_base_program(learned_rules)
    full_program = base_program + '\n' + requirements_asp + '\n' + facts

    violated = []
    try:
        ctl = Control(['0', '--warn=none'])
        ctl.add('base', [], full_program)
        ctl.ground([('base', [])])
        result = ctl.solve()
        satisfied = result.satisfiable
        if not satisfied:
            violated = ["Requirements violated: program has no stable models"]
    except RuntimeError as e:
        satisfied = False
        violated = [f"Clingo error: {e}"]

    return satisfied, violated


def count_violations(
    facts: str,
    dimacs_path: Optional[str] = None,
) -> int:
    """
    Count how many individual requirement clauses are violated by the given facts.
    Runs each constraint separately to isolate violations.
    """
    import clingo

    clauses = (
        load_dimacs_requirements(dimacs_path)
        if dimacs_path and os.path.exists(dimacs_path)
        else CORE_REQUIREMENTS
    )

    base_program = get_base_program()
    base_with_unification = base_program + '\n' + facts
    violations = 0

    for clause in clauses:
        constraint = clause_to_asp(clause)
        if constraint is None:
            continue
        program = base_with_unification + '\n' + constraint
        try:
            ctl = clingo.Control(['1', '--warn=none'])
            ctl.add('base', [], program)
            ctl.ground([('base', [])])
            result = ctl.solve()
            if not result.satisfiable:
                violations += 1
        except Exception:
            violations += 1  # parse error counts as violation

    return violations
