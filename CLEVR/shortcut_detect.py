"""
CLEVR-Hans Shortcut Detection

CLEVR-Hans is uniquely suited for shortcut detection because:
1. Ground-truth scene graphs are available → we know the true concept
2. Confounding is explicit and documented:
   - Class 0 train: large cube is ALWAYS gray (should only require: large + cube)
   - Class 1 train: small sphere is ALWAYS metal (should only require: small + sphere)
   - Class 2: no confounding (control class)
3. Test set removes confounding → train/test accuracy gap = shortcut score

Metrics:
  1. Shortcut Gap    = train_accuracy - test_accuracy per phase
  2. Concept Purity  = fraction of correct attributes used vs total attributes in rules
  3. Confound Usage  = whether confounded attrs (gray, metal-sphere) appear in rules
  4. Backward Compat = do phase-N rules still classify phase-1 scenes correctly?
  5. Cross-Phase Drift = accuracy on old classes when new classes introduced
"""

import re
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


# ── Known confounding in CLEVR-Hans3 ─────────────────────────────────────────
CONFOUNDINGS = {
    0: {
        'description': 'Large cube always gray in train/val',
        'confounded_attribute': 'color',
        'confounded_value': 'gray',
        'confounded_object': 'large cube',
        'true_rule': 'large cube AND large cylinder (color irrelevant)',
    },
    1: {
        'description': 'Small sphere always metal in train/val',
        'confounded_attribute': 'material',
        'confounded_value': 'metal',
        'confounded_object': 'small sphere',
        'true_rule': 'small metal cube AND small sphere (material of sphere irrelevant)',
    },
    2: {
        'description': 'No confounding — control class',
        'confounded_attribute': None,
        'confounded_value': None,
        'confounded_object': None,
        'true_rule': 'large blue sphere AND small yellow sphere',
    },
}

# ── Correct attribute sets per class ─────────────────────────────────────────
CORRECT_ATTRIBUTES = {
    0: {'cube', 'cylinder', 'large'},          # color/material NOT required
    1: {'cube', 'sphere', 'small', 'metal'},   # metal required for CUBE only
    2: {'sphere', 'large', 'small', 'blue', 'yellow'},
}

CONFOUNDED_ATTRIBUTES = {
    0: {'gray'},          # gray is confounded for class 0
    1: set(),             # no extra confound (metal IS required for cube in class 1)
    2: set(),
}


def shortcut_gap(
    phase_train_acc: Dict[int, float],
    phase_test_acc: Dict[int, float],
) -> Dict:
    """
    Compute shortcut gap per phase.
    High gap = model exploits train confounding, fails on unconfounded test.
    """
    gaps = {}
    for phase in sorted(phase_train_acc.keys()):
        train = phase_train_acc.get(phase, 0.0)
        test  = phase_test_acc.get(phase, 0.0)
        gap   = train - test
        gaps[f'phase{phase}'] = {
            'train_acc': round(train, 4),
            'test_acc':  round(test, 4),
            'gap':       round(gap, 4),
            'shortcut':  gap > 0.15,
        }
    return gaps


def concept_purity(rules: str, target_classes: List[int]) -> float:
    """
    Measure what fraction of attributes in the rules are CORRECT
    (appear in CORRECT_ATTRIBUTES) vs confounded.

    Returns value in [0, 1]: 1.0 = fully correct, 0.0 = all confounded.
    """
    all_correct = set()
    all_confounded = set()
    for c in target_classes:
        all_correct.update(CORRECT_ATTRIBUTES.get(c, set()))
        all_confounded.update(CONFOUNDED_ATTRIBUTES.get(c, set()))

    correct_count   = sum(1 for a in all_correct if a in rules)
    confounded_count = sum(1 for a in all_confounded if a in rules)

    total = correct_count + confounded_count
    if total == 0:
        return 0.0
    return round(correct_count / total, 4)


def confound_attribution(rules: str) -> Dict[int, bool]:
    """
    For each class, check if the rules contain the confounded attribute.
    True = shortcut present for that class.
    """
    return {
        0: 'gray' in rules and 'cube' in rules,    # using gray for cube identification
        1: ('sphere' in rules and 'metal' in rules  # using metal for sphere (not just cube)
            and re.search(r'sphere.*metal|metal.*sphere', rules) is not None),
        2: False,  # class 2 has no confounding
    }


def backward_compatibility(
    phase_rules: Dict[int, str],
    phase_facts: Dict[int, str],
    compute_accuracy_fn,
) -> Dict:
    """
    Check whether phase-N rules still correctly classify phase-1 scenes.
    A drop indicates catastrophic forgetting of earlier concepts.
    """
    results = {}
    p1_facts = phase_facts.get(1, '')
    p1_rules = phase_rules.get(1, '')
    p1_classes = [0]

    if not p1_facts:
        return results

    baseline = compute_accuracy_fn(p1_facts, p1_rules, p1_classes)

    for phase in sorted(phase_rules.keys()):
        if phase == 1:
            continue
        new_rules = phase_rules[phase]
        acc = compute_accuracy_fn(p1_facts, new_rules, p1_classes)
        results[f'phase{phase}_on_phase1'] = {
            'accuracy': round(acc, 4),
            'drop':     round(baseline - acc, 4),
            'forgot':   (baseline - acc) > 0.15,
        }

    return results


def cross_phase_drift(
    phase_rules: Dict[int, str],
    phase_facts: Dict[int, str],
    compute_accuracy_fn,
) -> Dict:
    """
    Apply rules from phase N to facts from phase N+1.
    High drop = rules didn't generalise to new classes.
    """
    results = {}
    phases = sorted(phase_rules.keys())

    for i in range(len(phases) - 1):
        src = phases[i]
        tgt = phases[i + 1]
        src_rules = phase_rules[src]
        tgt_facts  = phase_facts.get(tgt, '')
        tgt_classes = [0, 1] if tgt == 2 else [0, 1, 2]

        if not tgt_facts:
            continue

        acc = compute_accuracy_fn(tgt_facts, src_rules, tgt_classes[:src])
        results[f'phase{src}rules_on_phase{tgt}facts'] = {
            'accuracy': round(acc, 4),
            'note': f'Old rules applied to {len(tgt_classes)} classes, expected {src} classes',
        }

    return results


def shortcut_report(
    phase_rules: Dict[int, str],
    phase_train_acc: Dict[int, float],
    phase_test_acc: Dict[int, float],
    phase_facts: Dict[int, str],
    compute_accuracy_fn,
) -> Dict:
    """
    Full SHIELD shortcut detection report for CLEVR-Hans.
    """
    report = {}

    # 1. Shortcut gap
    print("Computing shortcut gaps...")
    report['shortcut_gap'] = shortcut_gap(phase_train_acc, phase_test_acc)

    # 2. Concept purity per phase
    print("Computing concept purity...")
    report['concept_purity'] = {}
    for phase, rules in phase_rules.items():
        classes = list(range(phase))
        report['concept_purity'][f'phase{phase}'] = concept_purity(rules, classes)

    # 3. Confound attribution
    print("Computing confound attribution...")
    report['confound_attribution'] = {}
    for phase, rules in phase_rules.items():
        report['confound_attribution'][f'phase{phase}'] = confound_attribution(rules)

    # 4. Backward compatibility
    print("Computing backward compatibility...")
    report['backward_compatibility'] = backward_compatibility(
        phase_rules, phase_facts, compute_accuracy_fn
    )

    # 5. Cross-phase drift
    print("Computing cross-phase drift...")
    report['cross_phase_drift'] = cross_phase_drift(
        phase_rules, phase_facts, compute_accuracy_fn
    )

    # 6. Overall verdict
    max_gap = max(
        (v['gap'] for v in report['shortcut_gap'].values()),
        default=0
    )
    min_purity = min(report['concept_purity'].values(), default=1.0)
    any_confounded = any(
        any(v.values()) for v in report['confound_attribution'].values()
    )

    report['verdict'] = {
        'max_shortcut_gap': round(max_gap, 4),
        'min_concept_purity': round(min_purity, 4),
        'confounded_attr_used': any_confounded,
        'risk': (
            'HIGH'   if max_gap > 0.15 or (any_confounded and min_purity < 0.5)
            else 'MEDIUM' if max_gap > 0.05 or min_purity < 0.7
            else 'LOW'
        ),
    }

    return report


def print_shortcut_report(report: Dict):
    print("\n" + "="*60)
    print("  SHIELD SHORTCUT DETECTION REPORT — CLEVR-Hans")
    print("="*60)

    print("\n[1] Shortcut Gap (train acc - test acc per phase):")
    for phase, vals in report.get('shortcut_gap', {}).items():
        flag = " ← SHORTCUT" if vals['shortcut'] else ""
        print(f"    {phase}: train={vals['train_acc']:.1%} test={vals['test_acc']:.1%} "
              f"gap={vals['gap']:+.1%}{flag}")

    print("\n[2] Concept Purity (fraction of correct attributes in rules):")
    for phase, purity in report.get('concept_purity', {}).items():
        bar = '█' * int(purity * 20) + '░' * (20 - int(purity * 20))
        print(f"    {phase}: [{bar}] {purity:.1%}")

    print("\n[3] Confounded Attribute Usage:")
    for phase, attrs in report.get('confound_attribution', {}).items():
        flags = [f"class{c}={'YES ←shortcut' if v else 'no'}" for c, v in attrs.items()]
        print(f"    {phase}: {', '.join(flags)}")

    print("\n[4] Backward Compatibility (new rules on old class scenes):")
    for key, vals in report.get('backward_compatibility', {}).items():
        flag = " ← FORGOT" if vals['forgot'] else ""
        print(f"    {key}: acc={vals['accuracy']:.1%} drop={vals['drop']:+.1%}{flag}")

    print("\n[5] Cross-Phase Drift:")
    for key, vals in report.get('cross_phase_drift', {}).items():
        print(f"    {key}: acc={vals['accuracy']:.1%}")

    v = report.get('verdict', {})
    print(f"\n[VERDICT] Shortcut Risk: {v.get('risk', 'UNKNOWN')}")
    print(f"  Max shortcut gap:    {v.get('max_shortcut_gap', 0):+.1%}")
    print(f"  Min concept purity:  {v.get('min_concept_purity', 0):.1%}")
    print(f"  Confounded attr used: {v.get('confounded_attr_used', False)}")
    print("="*60 + "\n")
