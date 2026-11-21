r"""Diagnostic sensitivity test for the existing MARL Dynamic 20 checkpoint.

Compares unchanged 17-feature observations against neutralized team features
at INFERENCE time. This is not a retrained ablation or proof of collaboration.
Does not modify any existing source/model files.
Run from project root: .\.venv\Scripts\python.exe check_team_feature_sensitivity.py
"""
import csv
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))

import torch
from benchmark_v3_v4_stress import run_scenario
from marl_features_v2 import get_team_features
from marl_model_v2 import MARLQNetworkV2
from marl_task_policy_v4 import select_task_actions

CHECKPOINT = ROOT / 'models_v4_dynamic' / 'marl_v4_dynamic_episode_20.pt'
OUT = ROOT / 'results' / 'team_feature_sensitivity'
SEEDS = range(33000, 33100)
MAX_CANDIDATES = 8


def model_from_checkpoint():
    checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    model = MARLQNetworkV2()
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


def allocator_for(model, neutralize_team):
    def allocator(robots, tasks, environment):
        pending = sorted((t for t in tasks if t.status == 'pending'),
                         key=lambda t: (-t.priority, t.task_id))[:MAX_CANDIDATES]
        idle = sorted((r for r in robots if r.current_task is None and r.goal is None),
                      key=lambda r: r.robot_id)
        if not pending or not idle:
            return
        observations = {}
        for robot in idle:
            candidates = []
            for task in pending:
                features = get_team_features(robot, task, robots, tasks, environment)
                if neutralize_team:
                    # Neutralize precisely the five appended team-aware features.
                    # Keep 17 input dimensions to retain compatibility with model.
                    features = list(features[:12]) + [0.5, 0.5, 0.5, 0.5, 0.5]
                candidates.append({'task_id': task.task_id, 'features': features})
            observations[robot.robot_id] = candidates
        with torch.no_grad():
            actions = select_task_actions(model, observations)
        robots_by_id = {r.robot_id: r for r in robots}
        tasks_by_id = {t.task_id: t for t in tasks}
        for rid, tid in actions.items():
            if tid is not None and tid in tasks_by_id and tasks_by_id[tid].status == 'pending':
                robots_by_id[rid].assign_task(tasks_by_id[tid], environment)
    return allocator


def successful(result):
    return (result.get('error') is None and result.get('completed') == 40
            and result.get('total_tasks') == 40 and result.get('collisions') == 0)


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f'Checkpoint missing: {CHECKPOINT}')
    model = model_from_checkpoint()
    rows = []
    for seed in SEEDS:
        for name, masked in [('full_team_features', False), ('neutralized_team_features', True)]:
            result = run_scenario(seed, name, allocator_for(model, masked))
            rows.append({'seed': seed, 'variant': name,
                         'success': int(successful(result)),
                         'finish': result.get('finish'),
                         'distance': result.get('distance'),
                         'p1_avg': result.get('p1_avg'),
                         'p3_avg': result.get('p3_avg'),
                         'collisions': result.get('collisions'),
                         'error': result.get('error') or ''})
    OUT.mkdir(parents=True, exist_ok=True)
    outpath = OUT / 'team_feature_sensitivity_by_seed.csv'
    with outpath.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('TEAM FEATURE INFERENCE SENSITIVITY TEST')
    print('Important: masking pretrained inputs is NOT a controlled training ablation.')
    for variant in ['full_team_features', 'neutralized_team_features']:
        group = [r for r in rows if r['variant'] == variant]
        valid = [r for r in group if r['success']]
        print(f'\n{variant}: successful {len(valid)}/{len(group)}')
        if valid:
            for metric in ['finish', 'distance', 'p1_avg', 'p3_avg']:
                vals = [float(r[metric]) for r in valid if r[metric] is not None]
                print(f'  {metric}: {mean(vals):.3f}' if vals else f'  {metric}: N/A')
    print(f'\nSaved: {outpath}')
    print('Next: train a truly matched 12-vs-17-feature ablation if needed.')


if __name__ == '__main__':
    main()
