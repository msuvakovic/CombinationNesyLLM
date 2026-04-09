"""
CLEVR-Hans SHIELD Pipeline — CLI Entry Point

Usage:
  python -m CLEVR.main --mode mock              # Generate mock data and run full pipeline
  python -m CLEVR.main --mode train             # Run full 3-phase training on real data
  python -m CLEVR.main --mode detect            # Shortcut detection only (needs saved rules)
  python -m CLEVR.main --mode info              # Dataset statistics
  python -m CLEVR.main --mode mock --engine grok-3-mini
"""

import argparse
import json
import os
import sys


def run_mock(args):
    """Generate mock CLEVR-Hans3 data and run the full pipeline on it."""
    from CLEVR.data_loader import generate_mock_clevr, CLEVRHansLoader
    from CLEVR.pipeline import CLEVRPipeline

    mock_dir = args.data_dir or '/tmp/mock_clevr_hans3'
    print(f"Generating mock CLEVR-Hans3 data at {mock_dir}...")
    generate_mock_clevr(mock_dir, n_per_class=args.n_scenes)

    # Verify it loaded correctly
    loader = CLEVRHansLoader(mock_dir)
    stats = loader.get_stats()
    print(f"\nDataset stats: {json.dumps(stats, indent=2)}")

    # Run pipeline
    pipeline = CLEVRPipeline(
        data_dir=mock_dir,
        engine=args.engine,
        results_dir=args.results_dir,
        max_scenes_per_phase=args.n_scenes,
    )
    metrics = pipeline.run_all_phases()

    # Shortcut detection
    _run_shortcut_report(pipeline)


def run_train(args):
    """Run full pipeline on real CLEVR-Hans3 data."""
    from CLEVR.pipeline import CLEVRPipeline

    if not args.data_dir or not os.path.isdir(args.data_dir):
        print(f"ERROR: --data_dir '{args.data_dir}' not found.")
        print("Download CLEVR-Hans3 from:")
        print("  https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/2611/CLEVR-Hans3.zip")
        print("Or run --mode mock for a synthetic test.")
        sys.exit(1)

    pipeline = CLEVRPipeline(
        data_dir=args.data_dir,
        engine=args.engine,
        results_dir=args.results_dir,
        max_scenes_per_phase=args.n_scenes,
    )
    metrics = pipeline.run_all_phases()
    _run_shortcut_report(pipeline)


def run_detect(args):
    """Run shortcut detection on previously saved rules."""
    from CLEVR.shortcut_detect import shortcut_report, print_shortcut_report
    from CLEVR.pipeline import CLEVRPipeline

    results_dir = args.results_dir
    engine = args.engine

    # Load saved rules
    phase_rules = {}
    for phase in [1, 2, 3]:
        path = os.path.join(results_dir, f'rules_phase{phase}_{engine}.txt')
        if os.path.exists(path):
            with open(path) as f:
                phase_rules[phase] = f.read()
            print(f"Loaded rules phase {phase} from {path}")
        else:
            print(f"No saved rules for phase {phase} at {path}")

    if not phase_rules:
        print("No saved rules found. Run --mode train or --mode mock first.")
        sys.exit(1)

    # Load facts for each phase
    data_dir = args.data_dir or '/tmp/mock_clevr_hans3'
    if not os.path.isdir(data_dir):
        print(f"Data dir {data_dir} not found. Run --mode mock first.")
        sys.exit(1)

    pipeline = CLEVRPipeline(
        data_dir=data_dir,
        engine=engine,
        results_dir=results_dir,
    )

    phase_facts = {}
    phase_train_acc = {}
    phase_test_acc = {}

    from CLEVR.data_loader import PHASE_CLASSES
    for phase in sorted(phase_rules.keys()):
        classes = PHASE_CLASSES.get(phase, [0])
        rules = phase_rules[phase]
        train_facts = pipeline.loader.get_phase_split(phase, 'train', max_scenes=50)
        test_facts  = pipeline.loader.get_phase_split(phase, 'test',  max_scenes=50)
        phase_facts[phase] = test_facts
        phase_train_acc[phase] = pipeline._compute_accuracy(train_facts, rules, classes)
        phase_test_acc[phase]  = pipeline._compute_accuracy(test_facts,  rules, classes)

    def compute_accuracy_fn(facts, rules, classes):
        return pipeline._compute_accuracy(facts, rules, classes)

    report = shortcut_report(
        phase_rules=phase_rules,
        phase_train_acc=phase_train_acc,
        phase_test_acc=phase_test_acc,
        phase_facts=phase_facts,
        compute_accuracy_fn=compute_accuracy_fn,
    )
    print_shortcut_report(report)

    report_path = os.path.join(results_dir, f'shortcut_report_{engine}.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Report saved → {report_path}")


def run_info(args):
    """Print dataset statistics."""
    from CLEVR.data_loader import CLEVRHansLoader

    data_dir = args.data_dir or '/tmp/mock_clevr_hans3'
    if not os.path.isdir(data_dir):
        print(f"Data dir '{data_dir}' not found.")
        return

    loader = CLEVRHansLoader(data_dir)
    stats = loader.get_stats()
    print(f"\nCLEVR-Hans3 dataset at: {data_dir}")
    print(json.dumps(stats, indent=2))

    # Show a sample trajectory
    for phase in [1, 2, 3]:
        try:
            traj = loader.get_sample_trajectory(phase, n_scenes=3)
            print(f"\n{traj}")
        except Exception as e:
            print(f"  Phase {phase}: {e}")


def _run_shortcut_report(pipeline):
    """Run and print shortcut detection using pipeline's phase metrics."""
    from CLEVR.shortcut_detect import shortcut_report, print_shortcut_report
    from CLEVR.data_loader import PHASE_CLASSES

    phase_rules = pipeline.learned_rules
    if not phase_rules:
        return

    phase_train_acc = {}
    phase_test_acc = {}
    phase_facts = {}

    for m in pipeline.phase_metrics:
        phase = m['phase']
        phase_train_acc[phase] = m.get('train_accuracy', 0.0)
        phase_test_acc[phase]  = m.get('test_accuracy',  0.0)

    for phase in sorted(phase_rules.keys()):
        phase_facts[phase] = pipeline.loader.get_phase_split(phase, 'test', max_scenes=50)

    def compute_accuracy_fn(facts, rules, classes):
        return pipeline._compute_accuracy(facts, rules, classes)

    report = shortcut_report(
        phase_rules=phase_rules,
        phase_train_acc=phase_train_acc,
        phase_test_acc=phase_test_acc,
        phase_facts=phase_facts,
        compute_accuracy_fn=compute_accuracy_fn,
    )
    print_shortcut_report(report)

    report_path = os.path.join(
        pipeline.results_dir,
        f'shortcut_report_{pipeline.engine}.json'
    )
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Shortcut report saved → {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='CLEVR-Hans SHIELD Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--mode', choices=['mock', 'train', 'detect', 'info'],
                        default='mock', help='Pipeline mode (default: mock)')
    parser.add_argument('--data_dir', default=None,
                        help='Path to CLEVR-Hans3 data directory')
    parser.add_argument('--engine', default='grok-3-mini',
                        help='LLM engine: grok-3-mini, gemini-2.0-flash, gpt-4o-mini')
    parser.add_argument('--results_dir', default='./CLEVR/results',
                        help='Directory to save results')
    parser.add_argument('--n_scenes', type=int, default=60,
                        help='Max scenes per phase (default: 60)')

    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    modes = {'mock': run_mock, 'train': run_train, 'detect': run_detect, 'info': run_info}
    modes[args.mode](args)


if __name__ == '__main__':
    main()
