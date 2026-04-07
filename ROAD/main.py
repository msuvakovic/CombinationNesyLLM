"""
SHIELD: Shortcut-free Hybrid Incremental Learning with Evolving Logical Domains
ROAD-R Experiment Entry Point

Usage:
  # Full continual learning pipeline (all 3 phases):
  python -m ROAD.main --annotation_path /path/to/road_trainval_v1.0.json

  # Single phase:
  python -m ROAD.main --annotation_path /path/to/road_trainval_v1.0.json --phase 1

  # With DIMACS requirements file (full 243 constraints):
  python -m ROAD.main \\
    --annotation_path /path/to/road_trainval_v1.0.json \\
    --dimacs_path /path/to/requirements_dimacs.txt \\
    --engine gpt-4o

  # Shortcut detection only (provide saved rules):
  python -m ROAD.main \\
    --annotation_path /path/to/road_trainval_v1.0.json \\
    --mode detect \\
    --rules_dir ./ROAD/results

  # Dataset info only:
  python -m ROAD.main --annotation_path /path/to/road_trainval_v1.0.json --mode info
"""

import argparse
import json
import os
import sys
from tabulate import tabulate


def parse_args():
    parser = argparse.ArgumentParser(
        description="SHIELD — ROAD-R Continual Learning with Shortcut Detection"
    )
    parser.add_argument(
        '--annotation_path',
        type=str,
        required=True,
        help='Path to road_trainval_v1.0.json',
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'detect', 'info'],
        default='train',
        help=(
            'train: run full continual learning pipeline; '
            'detect: run shortcut detection on saved rules; '
            'info: print dataset label summary'
        ),
    )
    parser.add_argument(
        '--phase',
        type=int,
        choices=[0, 1, 2, 3],
        default=0,
        help='Phase to run (0 = all phases sequentially)',
    )
    parser.add_argument(
        '--engine',
        type=str,
        default='gemini-2.0-flash',
        help='LLM engine (gemini-2.0-flash, gpt-4o, etc.)',
    )
    parser.add_argument(
        '--dimacs_path',
        type=str,
        default=None,
        help='Path to requirements_dimacs.txt (full 243 ROAD-R constraints)',
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='./ROAD/results',
        help='Directory to save results, learned rules, and shortcut reports',
    )
    parser.add_argument(
        '--rules_dir',
        type=str,
        default=None,
        help='For --mode detect: directory containing saved rule files (rules_phaseN_<engine>.txt)',
    )
    parser.add_argument(
        '--max_frames',
        type=int,
        default=50,
        help='Maximum frames per video to load (reduces memory/compute)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (for reproducibility)',
    )
    return parser.parse_args()


def mode_info(annotation_path: str):
    from ROAD.data_loader import ROADDataLoader
    loader = ROADDataLoader(annotation_path)
    summary = loader.get_label_summary()

    print("\n" + "="*60)
    print("  ROAD-R Dataset Summary")
    print("="*60)
    print(f"  Total videos:    {summary['n_videos']}")
    print(f"  Agent types:     {summary['n_agents']}  → {summary['agents']}")
    print(f"  Action types:    {summary['n_actions']} → {summary['actions']}")
    print(f"  Location types:  {summary['n_locs']}  → {summary['locations']}")

    from ROAD.data_loader import PHASE_LABELS
    print("\n  Continual Learning Phases:")
    for phase in [1, 2, 3]:
        pl = PHASE_LABELS[phase]
        print(f"  Phase {phase}:")
        print(f"    Agents:   {pl['agents'] or '(all remaining)'}")
        print(f"    Actions:  {pl['actions']}")
        print(f"    Locs:     {pl['locs']}")
    print("="*60 + "\n")


def mode_train(args):
    from ROAD.pipeline import ROADPipeline
    from ROAD.shortcut_detect import shortcut_report, print_shortcut_report

    print(f"\nInitialising ROAD-R pipeline...")
    print(f"  Engine:          {args.engine}")
    print(f"  Annotation file: {args.annotation_path}")
    print(f"  DIMACS file:     {args.dimacs_path or 'not provided (using built-in core requirements)'}")
    print(f"  Results dir:     {args.results_dir}")
    print(f"  Max frames/vid:  {args.max_frames}")

    pipeline = ROADPipeline(
        annotation_path=args.annotation_path,
        engine=args.engine,
        dimacs_path=args.dimacs_path,
        results_dir=args.results_dir,
        max_frames_per_video=args.max_frames,
    )

    if args.phase == 0:
        # Run all phases
        metrics = pipeline.run_all_phases()
    else:
        metrics = [pipeline.run_phase(args.phase)]

    # Print results table
    if metrics:
        print("\n" + "="*60)
        print("  RESULTS SUMMARY")
        print("="*60)
        headers = list(metrics[0].keys())
        rows = [[m.get(h, '') for h in headers] for m in metrics]
        print(tabulate(rows, headers=headers, tablefmt='grid'))

    # Run shortcut detection if all 3 phases were completed
    if args.phase == 0 and len(pipeline.learned_rules) == 3:
        print("\nRunning shortcut detection...")
        phase_facts = {}
        for p in [1, 2, 3]:
            phase_facts[p] = pipeline.loader.get_phase_facts(
                p, max_frames_per_video=args.max_frames
            )

        report = shortcut_report(
            pipeline.learned_rules,
            phase_facts,
            dimacs_path=args.dimacs_path,
        )
        print_shortcut_report(report)

        # Save report
        report_path = os.path.join(args.results_dir, f'shortcut_report_{args.engine}.json')
        with open(report_path, 'w') as f:
            # Convert any non-serialisable types
            serialisable = json.loads(json.dumps(report, default=str))
            json.dump(serialisable, f, indent=2)
        print(f"Shortcut report saved to {report_path}")


def mode_detect(args):
    from ROAD.data_loader import ROADDataLoader
    from ROAD.shortcut_detect import shortcut_report, print_shortcut_report

    rules_dir = args.rules_dir or args.results_dir
    phase_rules = {}

    for phase in [1, 2, 3]:
        rule_file = os.path.join(rules_dir, f'rules_phase{phase}_{args.engine}.txt')
        if os.path.exists(rule_file):
            with open(rule_file, 'r') as f:
                phase_rules[phase] = f.read()
            print(f"Loaded phase {phase} rules from {rule_file}")
        else:
            print(f"Warning: no rules file found for phase {phase} at {rule_file}")

    if not phase_rules:
        print("Error: no rule files found. Run --mode train first.")
        sys.exit(1)

    print(f"\nLoading ROAD-R annotations...")
    loader = ROADDataLoader(args.annotation_path)
    phase_facts = {}
    for p in sorted(phase_rules.keys()):
        print(f"  Loading phase {p} facts...")
        phase_facts[p] = loader.get_phase_facts(p, max_frames_per_video=args.max_frames)

    print("\nRunning shortcut detection...")
    report = shortcut_report(phase_rules, phase_facts, dimacs_path=args.dimacs_path)
    print_shortcut_report(report)

    report_path = os.path.join(args.results_dir, f'shortcut_report_{args.engine}.json')
    os.makedirs(args.results_dir, exist_ok=True)
    with open(report_path, 'w') as f:
        serialisable = json.loads(json.dumps(report, default=str))
        json.dump(serialisable, f, indent=2)
    print(f"Report saved to {report_path}")


def main():
    args = parse_args()

    if args.mode == 'info':
        mode_info(args.annotation_path)
    elif args.mode == 'train':
        mode_train(args)
    elif args.mode == 'detect':
        mode_detect(args)


if __name__ == '__main__':
    main()
