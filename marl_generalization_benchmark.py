r"""Evaluate a FIXED pretrained MARL Dynamic-20 checkpoint under scalability changes.

Put this script beside scalability_benchmark.py in the project root.

Examples (PowerShell):
    .\.venv\Scripts\python.exe marl_generalization_benchmark.py --seeds 10
    .\.venv\Scripts\python.exe marl_generalization_benchmark.py --seeds 30

The pretrained checkpoint is used for inference only (no optimization). All
results are written to results/marl_generalization/; existing benchmarks,
source files and checkpoints are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import torch
from marl_features_v2 import get_team_features
from marl_model_v2 import MARLQNetworkV2
from marl_task_policy_v4 import select_task_actions

from scalability_benchmark import (
    CONFIGS, run_one, summarize, write_csv,
)

CHECKPOINT = ROOT / "models_v4_dynamic" / "marl_v4_dynamic_episode_20.pt"
MAX_CANDIDATES = 8
ALGORITHM = "MARL Dynamic 20"


def load_model():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Missing pretrained checkpoint: {CHECKPOINT}")
    try:
        saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    except TypeError:  # older PyTorch versions
        saved = torch.load(CHECKPOINT, map_location="cpu")
    state = saved["model"] if isinstance(saved, dict) and "model" in saved else saved
    model = MARLQNetworkV2()
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def build_allocator(model):
    """Use exactly the trained 17-feature task selection, without exploration."""
    def allocator(robots, tasks, environment):
        pending = sorted(
            (t for t in tasks if t.status == "pending"),
            key=lambda t: (-t.priority, t.task_id),
        )
        idle = sorted(
            (r for r in robots if r.current_task is None and r.goal is None),
            key=lambda r: r.robot_id,
        )
        if not pending or not idle:
            return

        observations = {
            robot.robot_id: [
                {
                    "task_id": task.task_id,
                    "features": get_team_features(
                        robot, task, robots, tasks, environment
                    ),
                }
                for task in pending[:MAX_CANDIDATES]
            ]
            for robot in idle
        }

        with torch.no_grad():
            actions = select_task_actions(model, observations)

        robot_by_id = {r.robot_id: r for r in idle}
        task_by_id = {t.task_id: t for t in pending}
        reserved = set()
        for robot_id, task_id in actions.items():
            if task_id is None or task_id in reserved:
                continue
            robot = robot_by_id.get(robot_id)
            task = task_by_id.get(task_id)
            if robot is None or task is None or task.status != "pending":
                continue
            if robot.assign_task(task, environment):
                reserved.add(task_id)

    return allocator


def load_classical_comparison(start_seed, count):
    """Read already saved classical scalability results, if compatible."""
    path = ROOT / "results" / "scalability" / "scalability_runs.csv"
    if not path.is_file():
        return {}, "No saved classical scalability CSV found; MARL-only report."

    wanted = set(range(start_seed, start_seed + count))
    baseline = {}
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                seed = int(row["seed"])
                if seed not in wanted:
                    continue
                key = (seed, row["grid"], int(row["robots"]), row["algorithm"])
                baseline[key] = row
    except (OSError, ValueError, KeyError) as exc:
        return {}, f"Classical CSV could not be read: {exc}"
    return baseline, f"Loaded saved classical results: {path.name}"


def paired_comparison(marl_rows, classical):
    """Paired deltas on identical seed/config; positive means MARL uses more."""
    records = []
    for row in marl_rows:
        if not row["success"]:
            continue
        for algo in ("Auction", "Hungarian V3", "Hungarian V4"):
            match = classical.get((row["seed"], row["grid"], row["robots"], algo))
            if not match or str(match.get("success", "0")).strip() not in ("1", "True", "true"):
                continue
            out = {
                "seed": row["seed"], "grid": row["grid"], "robots": row["robots"],
                "comparator": algo,
            }
            valid = True
            for field in ("finish", "distance", "p1_avg", "p3_avg"):
                try:
                    a = float(row[field])
                    b = float(match[field])
                    if not (math.isfinite(a) and math.isfinite(b)):
                        valid = False
                        break
                    out[f"delta_{field}_marl_minus_classical"] = round(a - b, 6)
                except (TypeError, ValueError, KeyError):
                    valid = False
                    break
            if valid:
                records.append(out)
    return records


def main():
    parser = argparse.ArgumentParser(description="Fixed MARL model scalability/generalization test")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=41000)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be >= 1")

    torch.set_num_threads(2)
    model = load_model()
    allocator = build_allocator(model)

    print("MARL DYNAMIC 20 — ZERO-SHOT SCALABILITY / GENERALIZATION")
    print("Pretrained checkpoint:", CHECKPOINT.name)
    print("No retraining or checkpoint changes. 17 team-aware features; 8 candidates.")
    print("Configs:", CONFIGS, "| seeds/config:", args.seeds)
    print("Uses the SAME scenario generator as scalability_benchmark.py.")
    print("Important: 30x30 and different robot counts are out-of-training-distribution.\n")

    runs = []
    for rows, cols, robot_count in CONFIGS:
        print(f"CONFIGURATION: {rows}x{cols}, {robot_count} robots", flush=True)
        for index in range(args.seeds):
            seed = args.start_seed + index
            result = run_one(seed, rows, cols, robot_count, ALGORITHM, allocator)
            runs.append(result)
            state = "OK" if result["success"] else "FAILED"
            print(f"  seed={seed} {state:6s} tasks={result['completed']}/{result['total_tasks']} "
                  f"steps={result['finish']} distance={result['distance']} "
                  f"collisions={result['collisions']} error={result['error']}", flush=True)

    report_dir = ROOT / "results" / "marl_generalization"
    report_dir.mkdir(parents=True, exist_ok=True)
    raw_path = report_dir / f"marl_generalization_runs_{args.seeds}_seeds.csv"
    summary_path = report_dir / f"marl_generalization_summary_{args.seeds}_seeds.csv"
    write_csv(raw_path, runs)
    summary = summarize(runs)
    write_csv(summary_path, summary)

    print("\nSUMMARY (means over SUCCESSFUL runs only)")
    for row in summary:
        def fmt(value):
            return "N/A" if value is None else f"{value:.3f}"
        print(f"  {row['grid']}, {row['robots']} robots: success {row['successful']}/{row['runs']}"
              f", steps {fmt(row['mean_finish'])}, distance {fmt(row['mean_distance'])}"
              f", runtime {fmt(row['mean_runtime_seconds'])}s"
              f", urgent {fmt(row['mean_p3_avg'])}")

    classical, note = load_classical_comparison(args.start_seed, args.seeds)
    print("\n", note)
    paired = paired_comparison(runs, classical)
    if paired:
        paired_path = report_dir / f"marl_vs_classical_paired_{args.seeds}_seeds.csv"
        write_csv(paired_path, paired)
        print("PAIRED DELTAS: MARL - classical (negative means MARL is lower)")
        for (grid, robots, algo) in sorted({(r['grid'], r['robots'], r['comparator']) for r in paired}):
            subset = [r for r in paired if (r['grid'], r['robots'], r['comparator']) == (grid, robots, algo)]
            dfinish = statistics.mean(r["delta_finish_marl_minus_classical"] for r in subset)
            ddist = statistics.mean(r["delta_distance_marl_minus_classical"] for r in subset)
            print(f"  {grid}, {robots} robots vs {algo}: n={len(subset)}, "
                  f"delta steps={dfinish:+.3f}, delta distance={ddist:+.3f}")
        print("Saved paired:", paired_path)
    else:
        print("No complete paired comparisons available. Baseline CSV may use different seeds.")

    print("\nSaved:", raw_path, "\n      ", summary_path)
    print("Existing models, GUI, source code and classical benchmark CSVs unchanged.")


if __name__ == "__main__":
    main()
