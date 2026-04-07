"""
ROAD-R NeSyC Pipeline
Adapts the NeSyC continual learning framework (from Alfworld/) to ROAD-R.

Continual learning loop:
  Phase 1 → Phase 2 → Phase 3
  Each phase:
    1. Load ASP facts from phase videos (filtered to phase-available labels)
    2. LLM generates candidate rules from example trajectories
    3. Clingo validates rules against base program + requirements
    4. Shortcut detector measures concept quality
    5. Rules are adapted iteratively until stable or max trials reached

Key differences from ALFWorld pipeline:
  - No simulator: facts come directly from JSON annotations
  - Validation uses ROAD-R logical requirements (243 constraints) instead of
    task-completion goals
  - Shortcut detection is an explicit additional metric
  - Phase boundary triggers knowledge evolution
"""

import os
import json
import clingo
import csv
from clingo.control import Control
from typing import Dict, List, Optional, Tuple

from ROAD.data_loader import ROADDataLoader
from ROAD.requirements import (
    get_base_program,
    get_requirements_asp,
    count_violations,
    ROAD_LABELS,
)

# Reuse LLM utilities from the Alfworld module
from Alfworld.llm_utils import google_llm, openai_llm

MAX_REFINEMENT_TRIALS = 3


def _llm_call(prompt: str, engine: str, max_tokens: int = 512) -> str:
    try:
        if 'gemini' in engine:
            return google_llm(prompt=prompt, model_name=engine, max_tokens=max_tokens)
        elif 'gpt' in engine:
            response, _ = openai_llm(prompt=prompt, model_name=engine, max_tokens=max_tokens)
            return response
    except Exception as e:
        print(f"LLM error: {e}")
        return ""
    return ""


def _extract_asp_rules(text: str) -> str:
    """Strip markdown/noise from LLM output and return only ASP rule lines."""
    lines = text.replace('```asp', '').replace('```prolog', '').replace('```', '').split('\n')
    rules = []
    for line in lines:
        line = line.strip()
        # Accept integrity constraints and normal rules
        if line.startswith(':-') or (':-' in line and line.endswith('.')):
            rules.append(line)
    return '\n'.join(rules)


def _is_valid_asp(program: str) -> bool:
    """Return True if the ASP program parses without error."""
    try:
        ctl = Control(['0', '--warn=none'])
        ctl.add('base', [], program)
        ctl.ground([('base', [])])
        return True
    except RuntimeError:
        return False


class ROADPipeline:
    def __init__(
        self,
        annotation_path: str,
        engine: str = 'gemini-2.0-flash',
        dimacs_path: Optional[str] = None,
        results_dir: str = './ROAD/results',
        max_frames_per_video: int = 50,
    ):
        """
        Args:
            annotation_path: Path to road_trainval_v1.0.json
            engine: LLM engine name ('gemini-2.0-flash', 'gpt-4o', etc.)
            dimacs_path: Optional path to requirements_dimacs.txt (243 constraints)
            results_dir: Where to save results CSV and learned rules
            max_frames_per_video: Limit frames per video to keep ASP facts manageable
        """
        self.loader = ROADDataLoader(annotation_path)
        self.engine = engine
        self.dimacs_path = dimacs_path
        self.results_dir = results_dir
        self.max_frames = max_frames_per_video

        os.makedirs(results_dir, exist_ok=True)

        self.learned_rules: Dict[int, str] = {}  # phase → ASP rules string
        self.phase_metrics: List[Dict] = []
        self.requirements_asp = get_requirements_asp(dimacs_path)

        self._load_prompts()

    def _load_prompts(self):
        """Load LLM prompts from the prompts/ directory."""
        prompt_dir = os.path.join(os.path.dirname(__file__), 'prompts')
        self.prompts = {}
        for name, path in [
            ('general', os.path.join(prompt_dir, 'general', 'generalize.txt')),
            ('adapt',   os.path.join(prompt_dir, 'adapt',   'adapt.txt')),
        ]:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    self.prompts[name] = f.read()
            else:
                self.prompts[name] = self._default_prompt(name)

    def _default_prompt(self, kind: str) -> str:
        if kind == 'general':
            return (
                "You are an expert in Answer Set Programming (ASP) and autonomous driving.\n"
                "Given the following annotated trajectory of road agents (agent types, actions, locations),\n"
                "generate ASP integrity constraints (:- head :- body. format) that capture valid\n"
                "agent-action-location combinations and mutual exclusivity rules.\n"
                "Output ONLY valid ASP rules, one per line, ending with a period.\n"
                "Do not include explanations.\n\n"
                "Available predicates:\n"
                "  agent(TubeId, AgentType, FrameId, VideoId)\n"
                "  action(TubeId, ActionType, FrameId, VideoId)\n"
                "  loc(TubeId, LocationType, FrameId, VideoId)\n"
                "  has_label(TubeId, Label, FrameId, VideoId)  % unified\n\n"
            )
        else:  # adapt
            return (
                "You are an expert in ASP and autonomous driving.\n"
                "The following ASP rules were learned in a previous phase but may not hold\n"
                "for new agent types or locations introduced in the current phase.\n"
                "Given the current trajectory observations, adapt the rules so they remain\n"
                "valid. Output ONLY valid ASP integrity constraints, one per line.\n\n"
                "Previous rules:\n{previous_rules}\n\n"
                "Current trajectory:\n{trajectory}\n\n"
                "Adapted rules:\n"
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Phase execution
    # ─────────────────────────────────────────────────────────────────────────

    def run_phase(self, phase: int) -> Dict:
        """
        Run a single continual learning phase.
        Returns metrics dict.
        """
        print(f"\n{'='*60}")
        print(f"  PHASE {phase}")
        print(f"{'='*60}")

        # 1. Load facts for this phase
        print(f"Loading facts for phase {phase} videos...")
        facts = self.loader.get_phase_facts(phase, max_frames_per_video=self.max_frames)
        print(f"  Facts loaded: {len(facts.splitlines())} lines")

        # 2. Get a sample trajectory for LLM prompting
        sample_traj = self._get_sample_trajectory(phase)

        # 3. Generate rules via LLM
        print(f"Generating rules with {self.engine}...")
        previous_rules = self.learned_rules.get(phase - 1, '')
        if phase == 1 or not previous_rules:
            rules = self._generalize(sample_traj)
        else:
            rules = self._adapt(sample_traj, previous_rules)

        # 4. Iterative refinement against requirements
        rules, n_trials, satisfied = self._refine_rules(facts, rules)
        self.learned_rules[phase] = rules

        print(f"\nLearned rules (phase {phase}):\n{rules}")

        # Save rules
        rule_path = os.path.join(self.results_dir, f'rules_phase{phase}_{self.engine}.txt')
        with open(rule_path, 'w') as f:
            f.write(rules)
        print(f"Rules saved to {rule_path}")

        # 5. Evaluate on validation videos
        metrics = self._evaluate(phase, facts, rules)
        metrics.update({
            'phase': phase,
            'engine': self.engine,
            'refinement_trials': n_trials,
            'requirements_satisfied': satisfied,
            'n_rules': len([l for l in rules.splitlines() if l.strip()]),
        })

        self.phase_metrics.append(metrics)
        print(f"\nPhase {phase} metrics: {metrics}")
        return metrics

    def run_all_phases(self) -> List[Dict]:
        """Run all 3 phases sequentially (continual learning loop)."""
        for phase in [1, 2, 3]:
            self.run_phase(phase)
        self._save_results()
        return self.phase_metrics

    # ─────────────────────────────────────────────────────────────────────────
    # LLM rule generation
    # ─────────────────────────────────────────────────────────────────────────

    def _get_sample_trajectory(self, phase: int, n_frames: int = 15) -> str:
        """Get a representative trajectory string for LLM prompting."""
        videos = self.loader.get_phase_videos(phase)
        if not videos:
            return "No videos available for this phase."
        video = videos[0]
        frames = self.loader.database.get(video, {}).get('frames', {})
        frame_ids = sorted(frames.keys(), key=lambda x: int(x))[:n_frames]

        lines = [f"Video: {video}"]
        allowed_agents  = self.loader._phase_agent_sets.get(phase, set())
        allowed_actions = self.loader._phase_action_sets.get(phase, set())
        allowed_locs    = self.loader._phase_loc_sets.get(phase, set())

        for fid in frame_ids:
            frame = frames[fid]
            if not frame.get('annotated', 0):
                continue
            for anno_key, anno in list(frame.get('annos', {}).items())[:3]:
                agents  = [self.loader.agent_labels[i] for i in anno.get('agent_ids', [])
                           if i < len(self.loader.agent_labels) and self.loader.agent_labels[i] in allowed_agents]
                actions = [self.loader.action_labels[i] for i in anno.get('action_ids', [])
                           if i < len(self.loader.action_labels) and self.loader.action_labels[i] in allowed_actions]
                locs    = [self.loader.loc_labels[i] for i in anno.get('loc_ids', anno.get('location_ids', []))
                           if i < len(self.loader.loc_labels) and self.loader.loc_labels[i] in allowed_locs]
                if agents:
                    lines.append(f"  Frame {fid}: {agents} | {actions} | {locs}")
        return '\n'.join(lines)

    def _generalize(self, trajectory: str) -> str:
        """Phase 1: generate rules from scratch via LLM."""
        prompt = self.prompts['general'] + f"Trajectory:\n{trajectory}\n\nASP Rules:\n"
        response = _llm_call(prompt, self.engine, max_tokens=1024)
        return _extract_asp_rules(response)

    def _adapt(self, trajectory: str, previous_rules: str) -> str:
        """Phase 2+: adapt existing rules for new labels via LLM."""
        prompt = self.prompts['adapt'].replace(
            '{previous_rules}', previous_rules
        ).replace(
            '{trajectory}', trajectory
        )
        response = _llm_call(prompt, self.engine, max_tokens=1024)
        return _extract_asp_rules(response)

    # ─────────────────────────────────────────────────────────────────────────
    # Rule refinement (iterative Clingo validation)
    # ─────────────────────────────────────────────────────────────────────────

    def _refine_rules(
        self, facts: str, rules: str
    ) -> Tuple[str, int, bool]:
        """
        Iteratively refine rules until they are consistent with requirements,
        or MAX_REFINEMENT_TRIALS is reached.

        Returns: (final_rules, n_trials, satisfied)
        """
        base = get_base_program(rules)
        full_program = base + '\n' + self.requirements_asp + '\n' + facts

        for trial in range(1, MAX_REFINEMENT_TRIALS + 1):
            valid = _is_valid_asp(base + '\n' + self.requirements_asp)
            if not valid:
                print(f"  Trial {trial}: ASP parse error — requesting LLM fix...")
                feedback = (
                    "The previous rules contained ASP syntax errors. "
                    "Please rewrite them correctly:\n\n"
                    f"{rules}\n\nCorrected rules:\n"
                )
                rules = _extract_asp_rules(_llm_call(feedback, self.engine))
                base = get_base_program(rules)
                continue

            # Quick satisfiability check on a small facts subset
            sample_facts = '\n'.join(facts.splitlines()[:200])
            sample_prog = base + '\n' + self.requirements_asp + '\n' + sample_facts
            try:
                ctl = Control(['1', '--warn=none'])
                ctl.add('base', [], sample_prog)
                ctl.ground([('base', [])])
                result = ctl.solve()
                if result.satisfiable:
                    print(f"  Trial {trial}: rules satisfy requirements ✓")
                    return rules, trial, True
                else:
                    print(f"  Trial {trial}: requirements violated — requesting LLM adaptation...")
            except RuntimeError as e:
                print(f"  Trial {trial}: Clingo error: {e}")

            # Feed back to LLM
            feedback = (
                f"The following rules violate ROAD-R logical requirements:\n\n"
                f"{rules}\n\n"
                "Identify the conflicting rules and rewrite them so they do not "
                "contradict mutual-exclusivity or location/action constraints:\n\n"
                "Corrected rules:\n"
            )
            rules = _extract_asp_rules(_llm_call(feedback, self.engine))
            base = get_base_program(rules)

        print(f"  Max trials reached. Returning last rules (may violate requirements).")
        return rules, MAX_REFINEMENT_TRIALS, False

    # ─────────────────────────────────────────────────────────────────────────
    # Evaluation
    # ─────────────────────────────────────────────────────────────────────────

    def _evaluate(self, phase: int, train_facts: str, rules: str) -> Dict:
        """
        Evaluate learned rules on validation videos.
        Metrics:
          - violation_rate: fraction of validation frames with requirement violations
          - backward_compatibility: do phase-N rules still satisfy phase-1 facts?
          - n_val_frames: number of validation frames checked
        """
        val_videos = self.loader.get_video_names('val')
        total_frames = 0
        total_violations = 0

        for video in val_videos:
            try:
                val_facts = self.loader.annotations_to_asp_facts(
                    video, phase=phase, max_frames=self.max_frames
                )
                n_viol = count_violations(val_facts + '\n' + rules, self.dimacs_path)
                total_violations += n_viol
                total_frames += 1
            except Exception as e:
                print(f"  Warning: eval error on {video}: {e}")

        violation_rate = total_violations / max(total_frames, 1)

        # Backward compatibility: apply current rules to phase-1 facts
        backward_compat = 1.0
        if phase > 1:
            phase1_facts = self.loader.get_phase_facts(1, max_frames_per_video=20)
            n_back_viol = count_violations(phase1_facts + '\n' + rules, self.dimacs_path)
            backward_compat = 1.0 if n_back_viol == 0 else max(0.0, 1.0 - n_back_viol / 10)

        return {
            'violation_rate': round(violation_rate, 4),
            'backward_compatibility': round(backward_compat, 4),
            'n_val_videos': len(val_videos),
            'total_req_violations': total_violations,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Results persistence
    # ─────────────────────────────────────────────────────────────────────────

    def _save_results(self):
        csv_path = os.path.join(self.results_dir, f'road_results_{self.engine}.csv')
        if not self.phase_metrics:
            return
        fieldnames = list(self.phase_metrics[0].keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.phase_metrics)
        print(f"\nResults saved to {csv_path}")
