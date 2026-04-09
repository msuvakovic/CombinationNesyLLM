"""
CLEVR-Hans SHIELD Pipeline
Continual learning over CLEVR-Hans3 classes using LLM + ASP.

Phase 1: Learn rules for Class 0 (large cube + large cylinder)
Phase 2: Learn rules for Class 0+1 (add: small metal cube + small sphere)
Phase 3: Learn rules for Class 0+1+2 (add: large blue sphere + small yellow sphere)

Key difference from ROAD-R: CLEVR has ground-truth scene graphs, so we can
PRECISELY measure whether the model learned the true concept or a shortcut.
"""

import os
import csv
import json
import clingo
from clingo.control import Control
from typing import Dict, List, Optional, Tuple

from CLEVR.data_loader import CLEVRHansLoader, PHASE_CLASSES
from CLEVR.rules import (
    get_base_program, get_gt_rules, CLASSIFICATION_CONSTRAINTS,
    measure_shortcut_score, attribute_usage_in_rules,
)
from Alfworld.llm_utils import google_llm, openai_llm, grok_llm

MAX_REFINEMENT_TRIALS = 3


def _llm_call(prompt: str, engine: str, max_tokens: int = 1024) -> str:
    try:
        if 'gemini' in engine:
            return google_llm(prompt=prompt, model_name=engine, max_tokens=max_tokens)
        elif 'gpt' in engine:
            response, _ = openai_llm(prompt=prompt, model_name=engine, max_tokens=max_tokens)
            return response
        elif 'grok' in engine:
            response, _ = grok_llm(prompt=prompt, model_name=engine, max_tokens=max_tokens)
            return response
    except Exception as e:
        print(f"  LLM error: {e}")
    return ""


def _extract_asp_rules(text: str) -> str:
    """Extract ASP rules from LLM output."""
    lines = text.replace('```asp', '').replace('```prolog', '').replace('```', '').split('\n')
    rules = []
    for line in lines:
        line = line.strip()
        if line.startswith(':-') or (line.endswith('.') and ':-' in line):
            rules.append(line)
        elif line.startswith('class_pred') and ':-' in line:
            rules.append(line)
    return '\n'.join(rules)


def _is_valid_asp(program: str) -> bool:
    try:
        ctl = Control(['0', '--warn=none'])
        ctl.add('base', [], program)
        ctl.ground([('base', [])])
        return True
    except RuntimeError:
        return False


class CLEVRPipeline:
    def __init__(
        self,
        data_dir: str,
        engine: str = 'grok-3-mini',
        results_dir: str = './CLEVR/results',
        max_scenes_per_phase: int = 100,
    ):
        self.loader = CLEVRHansLoader(data_dir)
        self.engine = engine
        self.results_dir = results_dir
        self.max_scenes = max_scenes_per_phase

        os.makedirs(results_dir, exist_ok=True)

        self.learned_rules: Dict[int, str] = {}
        self.phase_metrics: List[Dict] = []
        self._load_prompts()

    def _load_prompts(self):
        prompt_dir = os.path.join(os.path.dirname(__file__), 'prompts')
        self.prompts = {}
        for name, path in [
            ('general', os.path.join(prompt_dir, 'general', 'generalize.txt')),
            ('adapt',   os.path.join(prompt_dir, 'adapt', 'adapt.txt')),
        ]:
            if os.path.exists(path):
                with open(path) as f:
                    self.prompts[name] = f.read()
            else:
                self.prompts[name] = self._default_prompt(name)

    def _default_prompt(self, kind: str) -> str:
        if kind == 'general':
            return (
                "You are an expert in Answer Set Programming (ASP) and visual reasoning.\n"
                "Given CLEVR scene descriptions, generate ASP rules that classify scenes into classes.\n\n"
                "Available predicates:\n"
                "  obj(SceneId, ObjId, Shape, Size, Material, Color)\n"
                "  left_of(SceneId, ObjA, ObjB)  right_of(SceneId, ObjA, ObjB)\n"
                "  behind(SceneId, ObjA, ObjB)   front_of(SceneId, ObjA, ObjB)\n"
                "  in_left_half(SceneId, ObjId)   in_right_half(SceneId, ObjId)\n\n"
                "Values: shapes=cube/sphere/cylinder, sizes=large/small, "
                "materials=metal/rubber, colors=gray/red/blue/green/brown/purple/cyan/yellow\n\n"
                "Generate class_pred(SceneId, ClassId) rules. Output ONLY valid ASP rules, "
                "one per line, ending with a period.\n\n"
            )
        else:
            return (
                "You are an expert in ASP and visual reasoning.\n"
                "New classes have been introduced. Adapt the existing rules to cover them.\n\n"
                "Previous rules:\n{previous_rules}\n\n"
                "New class examples:\n{trajectory}\n\n"
                "Output ONLY valid ASP rules. Keep previous rules that are still correct.\n"
                "Adapted rules:\n"
            )

    # ── Phase execution ───────────────────────────────────────────────────────

    def run_phase(self, phase: int) -> Dict:
        print(f"\n{'='*60}\n  PHASE {phase} — Classes {PHASE_CLASSES[phase]}\n{'='*60}")

        train_facts = self.loader.get_phase_split(phase, 'train', self.max_scenes)
        sample_traj = self.loader.get_sample_trajectory(phase, n_scenes=6)
        print(f"  Train facts: {len(train_facts.splitlines())} lines")

        prev_rules = self.learned_rules.get(phase - 1, '')
        if phase == 1 or not prev_rules:
            print(f"  Generating rules with {self.engine}...")
            prompt = self.prompts['general'] + f"Scene examples:\n{sample_traj}\n\nASP Rules:\n"
            rules = _extract_asp_rules(_llm_call(prompt, self.engine))
        else:
            print(f"  Adapting rules for new classes with {self.engine}...")
            prompt = self.prompts['adapt'].replace(
                '{previous_rules}', prev_rules
            ).replace('{trajectory}', sample_traj)
            rules = _extract_asp_rules(_llm_call(prompt, self.engine))
            if not rules.strip():
                rules = prev_rules  # fallback: keep previous

        rules, n_trials, valid = self._refine_rules(train_facts, rules, phase)
        self.learned_rules[phase] = rules

        rule_path = os.path.join(self.results_dir, f'rules_phase{phase}_{self.engine}.txt')
        with open(rule_path, 'w') as f:
            f.write(rules)
        print(f"  Rules saved → {rule_path}")
        print(f"\n  Learned rules:\n{rules}")

        metrics = self._evaluate(phase, rules)
        metrics.update({
            'phase': phase,
            'classes': str(PHASE_CLASSES[phase]),
            'engine': self.engine,
            'refinement_trials': n_trials,
            'asp_valid': valid,
            'n_rules': len([l for l in rules.splitlines() if l.strip()]),
        })
        self.phase_metrics.append(metrics)
        print(f"\n  Metrics: {metrics}")
        return metrics

    def run_all_phases(self) -> List[Dict]:
        for phase in [1, 2, 3]:
            self.run_phase(phase)
        self._save_results()
        return self.phase_metrics

    # ── Refinement ───────────────────────────────────────────────────────────

    def _refine_rules(self, facts: str, rules: str, phase: int) -> Tuple[str, int, bool]:
        classes = PHASE_CLASSES.get(phase, [0])
        gt = get_gt_rules(classes)
        base = get_base_program(rules)

        for trial in range(1, MAX_REFINEMENT_TRIALS + 1):
            if not _is_valid_asp(base + '\n' + gt):
                print(f"  Trial {trial}: parse error, requesting fix...")
                prompt = (
                    f"Fix these ASP syntax errors:\n{rules}\n\n"
                    "Output corrected rules only:\n"
                )
                rules = _extract_asp_rules(_llm_call(prompt, self.engine))
                base = get_base_program(rules)
                continue

            # Check satisfiability on small sample
            sample = '\n'.join(facts.splitlines()[:100])
            prog = base + '\n' + gt + '\n' + sample
            try:
                ctl = Control(['1', '--warn=none'])
                ctl.add('base', [], prog)
                ctl.ground([('base', [])])
                result = ctl.solve()
                if result.satisfiable:
                    print(f"  Trial {trial}: rules are satisfiable ✓")
                    return rules, trial, True
                else:
                    print(f"  Trial {trial}: unsatisfiable, refining...")
            except RuntimeError as e:
                print(f"  Trial {trial}: error: {str(e)[:80]}")

            prompt = (
                f"The following ASP rules produce no valid models for CLEVR-Hans scenes:\n\n"
                f"{rules}\n\n"
                "Fix them so they correctly classify CLEVR scenes. Output rules only:\n"
            )
            rules = _extract_asp_rules(_llm_call(prompt, self.engine))
            base = get_base_program(rules)

        return rules, MAX_REFINEMENT_TRIALS, False

    # ── Evaluation ───────────────────────────────────────────────────────────

    def _evaluate(self, phase: int, rules: str) -> Dict:
        """
        Evaluate learned rules on val set.
        Key metrics:
          - val_accuracy: fraction of scenes correctly classified
          - shortcut_gap: train accuracy - test accuracy (shortcut indicator)
          - attr_usage: which attributes appear in rules
        """
        classes = PHASE_CLASSES.get(phase, [0])

        # Attribute usage analysis — does the model use confounded attributes?
        attr_usage = attribute_usage_in_rules(rules)

        # Val accuracy
        val_facts = self.loader.get_phase_split(phase, 'val', max_scenes=50)
        val_acc = self._compute_accuracy(val_facts, rules, classes)

        # Shortcut gap: compare train (confounded) vs test (unconfounded)
        train_facts = self.loader.get_phase_split(phase, 'train', max_scenes=50)
        test_facts  = self.loader.get_phase_split(phase, 'test', max_scenes=50)
        train_acc = self._compute_accuracy(train_facts, rules, classes)
        test_acc  = self._compute_accuracy(test_facts, rules, classes)

        return {
            'train_accuracy': round(train_acc, 4),
            'val_accuracy':   round(val_acc, 4),
            'test_accuracy':  round(test_acc, 4),
            'shortcut_gap':   round(train_acc - test_acc, 4),
            'shortcut_detected': (train_acc - test_acc) > 0.15,
            'confounded_attr_used': attr_usage['confounded_attr_used'],
            'correct_attrs_used':   attr_usage['correct_attrs_used'],
        }

    def _compute_accuracy(self, facts: str, rules: str, classes: List[int]) -> float:
        """Run ASP and measure classification accuracy."""
        if not facts.strip() or not rules.strip():
            return 0.0

        gt = get_gt_rules(classes)
        base = get_base_program(rules)
        prog = base + '\n' + gt + '\n' + facts

        try:
            ctl = Control(['0', '--warn=none'])
            ctl.add('base', [], prog)
            ctl.ground([('base', [])])

            predicted = {}
            ground_truth = {}

            # Extract class facts from facts string
            import re
            for m in re.finditer(r'class\((\w+),\s*(c\d+)\)', facts):
                ground_truth[m.group(1)] = m.group(2)

            def on_model(model):
                for atom in model.symbols(shown=True):
                    s = str(atom)
                    if s.startswith('class_pred('):
                        parts = s.replace('class_pred(','').rstrip(')').split(',')
                        if len(parts) == 2:
                            predicted[parts[0].strip()] = parts[1].strip()

            ctl.solve(on_model=on_model)

            if not ground_truth:
                return 0.0
            correct = sum(1 for sid, cid in ground_truth.items() if predicted.get(sid) == cid)
            return correct / len(ground_truth)
        except Exception:
            return 0.0

    # ── Results ──────────────────────────────────────────────────────────────

    def _save_results(self):
        if not self.phase_metrics:
            return
        csv_path = os.path.join(self.results_dir, f'clevr_results_{self.engine}.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(self.phase_metrics[0].keys()))
            writer.writeheader()
            writer.writerows(self.phase_metrics)
        print(f"\nResults saved → {csv_path}")
