"""Controlled feature-information fine-tuning ablation for Multi-Robot Coordination.

Runs two INDEPENDENT fine-tuning branches from the SAME V4 checkpoint:
  full_17: all 17 observed features
  no_team_12_info: same 17-input architecture, with last five team features
                   fixed to 0.5 throughout training AND evaluation

This is a controlled *information ablation* (not a 12-neuron-input network).
It holds architecture, initial weights, seed schedules, and hyperparameters fixed.
It does NOT claim isolation of the shared reward (dynamic trainer uses its own
per-decision reward), nor does it guarantee globally optimal policies.

Place at project root. Requires existing experiments/train_marl_v4_dynamic.py.
Examples:
  .\\.venv\\Scripts\\python.exe controlled_team_ablation.py --episodes 20 --eval-seeds 30
  .\\.venv\\Scripts\\python.exe controlled_team_ablation.py --episodes 20 --eval-seeds 100
"""
import argparse
import csv
import random
import sys
from collections import deque
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))

import torch
from marl_model_v2 import MARLQNetworkV2
from marl_features_v2 import get_team_features as original_team_features
from marl_task_policy_v4 import select_task_actions
from benchmark_v3_v4_stress import run_scenario
import train_marl_v4_dynamic as trainer

BASE_CHECKPOINT = ROOT / 'models_v4' / 'marl_v4_episode_100.pt'
OUTPUT_DIR = ROOT / 'results' / 'controlled_team_ablation'
CHECKPOINT_DIR = ROOT / 'models_team_ablation'
NEUTRAL_TEAM = [0.5] * 5
VARIANTS = ('full_17', 'no_team_12_info')
METRICS = ('finish', 'distance', 'p1_avg', 'p3_avg')


def feature_function(include_team):
    def features(robot, task, robots, tasks, environment):
        full = original_team_features(robot, task, robots, tasks, environment)
        if len(full) != 17:
            raise ValueError(f'Expected 17 features, got {len(full)}')
        return full if include_team else list(full[:12]) + NEUTRAL_TEAM
    return features


def initialize_branch(base_state, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    model = MARLQNetworkV2()
    model.load_state_dict(base_state)
    target = MARLQNetworkV2()
    target.load_state_dict(base_state)
    target.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=trainer.LR)
    return model, target, optimizer


def train_variant(variant, base_state, args):
    trainer.get_team_features = feature_function(variant == 'full_17')
    model, target, optimizer = initialize_branch(base_state, args.train_seed)
    replay = deque(maxlen=20000)
    updates = 0
    history = []
    for episode in range(1, args.episodes + 1):
        epsilon = max(0.05, 0.4 * (0.97 ** (episode - 1)))
        # Each variant sees the same episode seed and initial model weights.
        info = trainer.train_episode(
            model, target, optimizer, replay,
            seed=args.episode_seed_start + episode,
            epsilon=epsilon, updates=updates,
        )
        updates = info['updates']
        history.append({
            'variant': variant, 'episode': episode,
            'success': int(info['success']),
            'finish': info['finish'], 'distance': info['distance'],
            'collisions': info['collisions'], 'updates': updates,
        })
        if episode == 1 or episode % 5 == 0 or episode == args.episodes:
            print(f'  {variant}: episode {episode}/{args.episodes} '
                  f"success={info['success']} finish={info['finish']} "
                  f"distance={info['distance']} updates={updates}", flush=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f'{variant}_finetuned_{args.episodes}_episodes.pt'
    torch.save({'model': model.state_dict(), 'episodes': args.episodes,
                'base_checkpoint': str(BASE_CHECKPOINT), 'variant': variant,
                'feature_policy': 'real team features' if variant == 'full_17'
                                  else 'last five features constant at 0.5'}, path)
    print(f'  Saved new (separate) checkpoint: {path}', flush=True)
    model.eval()
    return model, history


def allocator_for(model, include_team):
    get_features = feature_function(include_team)
    def allocator(robots, tasks, environment):
        pending = sorted(
            (t for t in tasks if t.status == 'pending'),
            key=lambda t: (-t.priority, t.task_id),
        )[:trainer.MAX_CANDIDATES]
        idle = sorted(
            (r for r in robots if r.current_task is None and r.goal is None),
            key=lambda r: r.robot_id,
        )
        if not pending or not idle:
            return
        observations = {}
        for robot in idle:
            observations[robot.robot_id] = [
                {'task_id': task.task_id,
                 'features': get_features(robot, task, robots, tasks, environment)}
                for task in pending
            ]
        with torch.no_grad():
            actions = select_task_actions(model, observations)
        task_by_id = {task.task_id: task for task in tasks}
        robot_by_id = {robot.robot_id: robot for robot in robots}
        for rid, task_id in actions.items():
            if task_id is None:
                continue
            task = task_by_id.get(task_id)
            if task is not None and task.status == 'pending':
                robot_by_id[rid].assign_task(task, environment)
    return allocator


def is_success(result):
    return (result.get('error') is None and result.get('completed') == 40
            and result.get('total_tasks') == 40 and result.get('collisions') == 0)


def write_csv(path, records, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--eval-seeds', type=int, default=30)
    parser.add_argument('--eval-seed-start', type=int, default=34000)
    parser.add_argument('--episode-seed-start', type=int, default=10000)
    parser.add_argument('--train-seed', type=int, default=42)
    args = parser.parse_args()
    if args.episodes < 1 or args.eval_seeds < 1:
        parser.error('--episodes and --eval-seeds must be positive')
    if not BASE_CHECKPOINT.is_file():
        raise FileNotFoundError(f'Missing initial V4 checkpoint: {BASE_CHECKPOINT}')

    torch.set_num_threads(2)
    print('CONTROLLED TEAM-INFORMATION FINE-TUNING ABLATION', flush=True)
    print('Same pretrained model, architecture, episode seeds and training settings.', flush=True)
    print('Difference: five team-aware inputs are real vs fixed at 0.5 in TRAIN + TEST.', flush=True)
    print('Input dimensions remain 17 for BOTH; this is information ablation.', flush=True)
    print(f'Train episodes: {args.episodes} per variant; eval seeds: {args.eval_seeds}', flush=True)
    ckpt = torch.load(BASE_CHECKPOINT, map_location='cpu', weights_only=True)
    base_state = ckpt['model']

    try:
        models = {}
        histories = []
        for variant in VARIANTS:
            print(f'\nTraining {variant}...', flush=True)
            model, hist = train_variant(variant, base_state, args)
            models[variant] = model
            histories.extend(hist)
    finally:
        trainer.get_team_features = original_team_features

    write_csv(OUTPUT_DIR / 'training_history.csv', histories,
              ['variant', 'episode', 'success', 'finish', 'distance', 'collisions', 'updates'])

    print('\nEvaluating on shared, previously unused-by-this-script seeds...', flush=True)
    results = []
    for idx, seed in enumerate(range(args.eval_seed_start, args.eval_seed_start + args.eval_seeds), 1):
        for variant in VARIANTS:
            result = run_scenario(seed, variant,
                                  allocator_for(models[variant], variant == 'full_17'))
            row = {'seed': seed, 'variant': variant, 'success': int(is_success(result)),
                   'finish': result.get('finish'), 'distance': result.get('distance'),
                   'p1_avg': result.get('p1_avg'), 'p3_avg': result.get('p3_avg'),
                   'collisions': result.get('collisions'), 'error': result.get('error') or ''}
            results.append(row)
        if idx % 10 == 0 or idx == args.eval_seeds:
            print(f'  Evaluated {idx}/{args.eval_seeds} seeds', flush=True)

    write_csv(OUTPUT_DIR / 'evaluation_by_seed.csv', results,
              ['seed', 'variant', 'success', 'finish', 'distance', 'p1_avg', 'p3_avg',
               'collisions', 'error'])

    print('\nRESULTS (lower is better; averages over successful runs ONLY)')
    for variant in VARIANTS:
        subset = [r for r in results if r['variant'] == variant]
        passed = [r for r in subset if r['success']]
        print(f'\n{variant}: success {len(passed)}/{len(subset)}')
        for metric in METRICS:
            values = [float(r[metric]) for r in passed if r[metric] is not None]
            print(f'  {metric}: {mean(values):.3f}' if values else f'  {metric}: unavailable')

    paired = []
    lookup = {(r['seed'], r['variant']): r for r in results}
    for seed in range(args.eval_seed_start, args.eval_seed_start + args.eval_seeds):
        full = lookup[(seed, VARIANTS[0])]
        no_team = lookup[(seed, VARIANTS[1])]
        if not (full['success'] and no_team['success']):
            continue
        row = {'seed': seed}
        for metric in METRICS:
            row[f'{metric}_no_team_minus_full'] = float(no_team[metric]) - float(full[metric])
        paired.append(row)
    write_csv(OUTPUT_DIR / 'paired_deltas.csv', paired,
              ['seed'] + [f'{m}_no_team_minus_full' for m in METRICS])
    print(f'\nPaired successful seeds: {len(paired)}/{args.eval_seeds}')
    if paired:
        for metric in METRICS:
            diffs = [r[f'{metric}_no_team_minus_full'] for r in paired]
            print(f'  delta {metric} (no team - full): {mean(diffs):+.3f}')
    print('\nSaved reports:', OUTPUT_DIR)
    print('Saved independent checkpoints:', CHECKPOINT_DIR)
    print('Existing models, GUI, training scripts, and benchmarks unchanged.')
    print('Caution: two fine-tuning branches are not independent from-scratch training.')
    print('Caution: one matched training seed is not enough for robust significance.')


if __name__ == '__main__':
    main()
