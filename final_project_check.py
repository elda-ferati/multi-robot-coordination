"""Final smoke checks for the Multi-Robot Coordination project.

Put this file in the project root, next to gui.py, and run:
    python final_project_check.py

Tests use the existing scenario factory and pretrained models; no retraining.
"""
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))


def main():
    print('\nMULTI-ROBOT FINAL PROJECT CHECK')
    print('=' * 62)
    failures = []

    def check(name, test):
        try:
            detail = test()
        except Exception as exc:
            failures.append((name, str(exc)))
            print(f'[FAIL] {name}: {type(exc).__name__}: {exc}')
        else:
            print(f'[PASS] {name}' + (f' — {detail}' if detail else ''))

    def test_structure():
        required = [
            'gui.py', 'algorithm_comparison.py',
            'src/simulation.py', 'src/marl_model_v2.py',
            'src/marl_features_v2.py', 'src/marl_task_policy_v4.py',
            'experiments/benchmark_v3_v4_stress.py',
            'models_v4_dynamic/marl_v4_dynamic_episode_20.pt',
            'results/final_marl_dynamic_benchmark.csv',
        ]
        missing = [x for x in required if not (ROOT / x).is_file()]
        if missing:
            raise FileNotFoundError('Missing: ' + ', '.join(missing))
        return f'{len(required)} required files present'

    def test_syntax():
        import ast
        paths = [ROOT / 'gui.py', ROOT / 'algorithm_comparison.py']
        for path in paths:
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        return 'GUI and comparison code parse successfully'

    def test_benchmark_csv():
        from algorithm_comparison import read_results, PALETTE, METRICS
        groups = read_results()
        missing = set(PALETTE) - set(groups)
        if missing:
            raise AssertionError(f'Missing algorithms: {sorted(missing)}')
        for name in PALETTE:
            rows = groups[name]
            if len(rows) != 100:
                raise AssertionError(f'{name}: expected 100 results, got {len(rows)}')
            if not all(row['success'] for row in rows):
                raise AssertionError(f'{name}: some runs unsuccessful')
            for metric, _, _ in METRICS:
                if not all(row[metric] is not None and math.isfinite(row[metric]) for row in rows):
                    raise AssertionError(f'{name}: missing/invalid {metric}')
        return '6 algorithms, 100 successful runs each, valid metrics'

    def test_model():
        import torch
        from marl_model_v2 import MARLQNetworkV2
        path = ROOT / 'models_v4_dynamic' / 'marl_v4_dynamic_episode_20.pt'
        cp = torch.load(path, map_location='cpu', weights_only=True)
        net = MARLQNetworkV2()
        net.load_state_dict(cp['model'])
        net.eval()
        with torch.no_grad():
            q = net(torch.zeros((3, 17), dtype=torch.float32))
        if tuple(q.shape) != (3,):
            # Some versions output (N, 1); that is also valid.
            if tuple(q.shape) != (3, 1):
                raise AssertionError(f'Unexpected Q output shape {tuple(q.shape)}')
        return f'checkpoint loaded, output shape {tuple(q.shape)}'

    def test_scenarios():
        import torch
        import simulation
        from benchmark_v3_v4_stress import run_scenario
        from auction_allocation import allocate_tasks_auction
        from hungarian_v3 import allocate_tasks_hungarian_v3
        from hungarian_v4 import allocate_tasks_hungarian_v4
        from gui import allocate_marl_dynamic

        torch.set_num_threads(2)
        allocators = {
            'Auction': allocate_tasks_auction,
            'Hungarian V3': allocate_tasks_hungarian_v3,
            'Hungarian V4': allocate_tasks_hungarian_v4,
            'MARL Dynamic 20': allocate_marl_dynamic,
        }
        original = simulation.allocate_tasks
        try:
            summaries = []
            for seed in (100, 101):
                for name, allocator in allocators.items():
                    result = run_scenario(seed, name, allocator)
                    if result['error'] is not None:
                        raise AssertionError(f'{name} seed {seed}: {result["error"]}')
                    if result['completed'] != 40 or result['total_tasks'] != 40:
                        raise AssertionError(f'{name} seed {seed}: {result["completed"]}/{result["total_tasks"]}')
                    if result['collisions'] != 0:
                        raise AssertionError(f'{name} seed {seed}: collisions={result["collisions"]}')
                    summaries.append(f'{name}@{seed}: {result["finish"]} steps')
        finally:
            simulation.allocate_tasks = original
        return '; '.join(summaries)

    check('Required project files', test_structure)
    check('Python syntax', test_syntax)
    check('Saved comparison CSV', test_benchmark_csv)
    check('MARL Dynamic checkpoint', test_model)
    check('4 allocators × 2 identical scenario seeds', test_scenarios)

    print('\n' + '=' * 62)
    if failures:
        print(f'FINAL CHECK: {len(failures)} FAILED, {5-len(failures)} PASSED')
        return 1
    print('FINAL CHECK: ALL 5 CHECKS PASSED')
    print('Manual GUI checks: Start, Pause, Step, Reset, Comparison, Summary, Export.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
