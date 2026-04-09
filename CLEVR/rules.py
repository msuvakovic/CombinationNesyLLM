"""
CLEVR-Hans ASP Rules and Shortcut Definitions

Ground-truth class concepts expressed as ASP integrity constraints.
These are used to:
1. Validate learned rules against ground truth
2. Define what a shortcut looks like (using confounded attribute instead of true concept)
3. Seed the base ASP program
"""

from typing import Dict, List, Optional, Tuple


# ── Base ASP program ─────────────────────────────────────────────────────────
BASE_ASP_CLEVR = """
% ── CLEVR-Hans Base ASP Program ──────────────────────────────────────────────

% Attribute domains
shape(cube). shape(sphere). shape(cylinder).
size(large). size(small).
material(metal). material(rubber).
color(gray). color(red). color(blue). color(green).
color(brown). color(purple). color(cyan). color(yellow).

% Convenience predicates
has_shape(S, O, Sh)   :- obj(S, O, Sh, _, _, _).
has_size(S, O, Sz)    :- obj(S, O, _, Sz, _, _).
has_material(S, O, M) :- obj(S, O, _, _, M, _).
has_color(S, O, C)    :- obj(S, O, _, _, _, C).

% An object matches a partial description
matches(S, O, Sh, Sz, M, C) :-
    obj(S, O, Sh, Sz, M, C),
    shape(Sh), size(Sz), material(M), color(C).

% Two distinct objects in same scene
two_objects(S, O1, O2) :- obj(S, O1, _, _, _, _), obj(S, O2, _, _, _, _), O1 != O2.

% ── Learned Rules Section (filled in by pipeline) ────────────────────────────
% {learned_rules}
"""

# ── Ground-truth class definitions as ASP rules ──────────────────────────────
# These express what it MEANS to be in each class.
# In a correct learned program, these (or equivalent) should appear.

GT_CLASS_RULES = {
    0: """
% Class 0: large cube AND large cylinder exist in scene
class_pred(S, c0) :-
    scene(S),
    obj(S, O1, cube, large, _, _),
    obj(S, O2, cylinder, large, _, _),
    O1 != O2.
""",
    1: """
% Class 1: small metal cube AND small sphere exist in scene
class_pred(S, c1) :-
    scene(S),
    obj(S, O1, cube, small, metal, _),
    obj(S, O2, sphere, small, _, _),
    O1 != O2.
""",
    2: """
% Class 2: large blue sphere AND small yellow sphere
class_pred(S, c2) :-
    scene(S),
    obj(S, O1, sphere, large, _, blue),
    obj(S, O2, sphere, small, _, yellow),
    O1 != O2.
""",
}

# ── Shortcut definitions ─────────────────────────────────────────────────────
# These are the WRONG rules that exploit confounding in train/val.
# If a model learns these instead of GT_CLASS_RULES, it has a shortcut.

SHORTCUT_RULES = {
    0: """
% SHORTCUT for Class 0: uses color=gray (confounded) instead of shape=cube
% A model with this shortcut will fail on test where cube color is randomized
class_pred(S, c0) :-
    scene(S),
    obj(S, O1, _, large, _, gray),   % <-- should be cube, not gray
    obj(S, O2, cylinder, large, _, _),
    O1 != O2.
""",
    1: """
% SHORTCUT for Class 1: uses material=metal for sphere (confounded)
% A model with this shortcut will fail when sphere is rubber on test
class_pred(S, c1) :-
    scene(S),
    obj(S, O1, cube, small, metal, _),
    obj(S, O2, sphere, small, metal, _),  % <-- should not require metal
    O1 != O2.
""",
}

# ── ASP integrity constraints (what must hold for valid classifications) ──────
CLASSIFICATION_CONSTRAINTS = """
% Every scene must be classified as exactly one class
:- scene(S), not class_pred(S, c0), not class_pred(S, c1), not class_pred(S, c2).

% A scene classified as c0 must have the right objects
:- class(S, c0), not class_pred(S, c0).
:- class(S, c1), not class_pred(S, c1).
:- class(S, c2), not class_pred(S, c2).
"""


def get_base_program(learned_rules: str = "") -> str:
    return BASE_ASP_CLEVR.replace('{learned_rules}', learned_rules)


def get_gt_rules(classes: List[int]) -> str:
    """Return ground-truth ASP rules for the given class IDs."""
    return '\n'.join(GT_CLASS_RULES[c] for c in classes if c in GT_CLASS_RULES)


def measure_shortcut_score(
    learned_rules: str,
    gt_facts_correct: str,
    confounded_facts: str,
) -> Dict:
    """
    Measure whether learned rules exploit shortcuts.

    Strategy:
    - Apply learned rules to gt_facts_correct (test set, no confounding)
    - Apply learned rules to confounded_facts (train set, with confounding)
    - If accuracy drops significantly on test vs train → shortcut detected

    Since we work with ASP (not neural predictions), we check:
    - Does the learned program correctly classify confounded examples?
    - Does it still classify correctly when confound is removed?

    Returns dict with scores.
    """
    import clingo

    def count_correct(facts: str, rules: str, classes: List[int]) -> Tuple[int, int]:
        """Count correctly classified scenes."""
        base = get_base_program(rules)
        gt_rules = get_gt_rules(classes)
        program = base + '\n' + gt_rules + '\n' + CLASSIFICATION_CONSTRAINTS + '\n' + facts

        correct = 0
        total = 0
        try:
            ctl = clingo.Control(['0', '--warn=none'])
            ctl.add('base', [], program)
            ctl.ground([('base', [])])

            models = []
            ctl.solve(on_model=lambda m: models.append(
                [str(a) for a in m.symbols(shown=True)]
            ))

            # Count class/2 facts that match ground truth
            for model in models[:1]:
                for atom in model:
                    if atom.startswith('class_pred('):
                        total += 1
                        # Check if it matches class(S, C) in facts
                        sid, cid = atom.replace('class_pred(', '').rstrip(')').split(', ')
                        if f'class({sid}, {cid}).' in facts:
                            correct += 1
        except Exception:
            pass
        return correct, total

    classes = [0, 1, 2]
    conf_correct, conf_total = count_correct(confounded_facts, learned_rules, classes)
    test_correct, test_total = count_correct(gt_facts_correct, learned_rules, classes)

    conf_acc = conf_correct / max(conf_total, 1)
    test_acc = test_correct / max(test_total, 1)
    shortcut_gap = conf_acc - test_acc

    return {
        'confounded_accuracy': round(conf_acc, 4),
        'test_accuracy': round(test_acc, 4),
        'shortcut_gap': round(shortcut_gap, 4),
        'shortcut_detected': shortcut_gap > 0.15,
    }


def attribute_usage_in_rules(rules: str) -> Dict[str, bool]:
    """
    Detect which CLEVR attributes appear in the learned rules.
    Used to identify if confounded attributes (gray, metal) are over-used.
    """
    attrs = {
        'color_gray':    'gray' in rules,
        'color_blue':    'blue' in rules,
        'color_yellow':  'yellow' in rules,
        'material_metal': 'metal' in rules,
        'shape_cube':    'cube' in rules,
        'shape_sphere':  'sphere' in rules,
        'shape_cylinder':'cylinder' in rules,
        'size_large':    'large' in rules,
        'size_small':    'small' in rules,
        'spatial_left':  'left_of' in rules or 'in_left_half' in rules,
        'spatial_right': 'right_of' in rules or 'in_right_half' in rules,
    }
    # Confounded attributes are gray (class 0) and metal (class 1)
    confounded_used = attrs['color_gray'] or attrs['material_metal']
    correct_attrs_used = (
        attrs['shape_cube'] and attrs['shape_cylinder'] or  # class 0
        attrs['material_metal'] and attrs['shape_sphere'] or  # class 1
        attrs['color_blue'] and attrs['color_yellow']  # class 2
    )
    attrs['confounded_attr_used'] = confounded_used
    attrs['correct_attrs_used'] = correct_attrs_used
    return attrs
