"""Scalability benchmark for Multi-Robot Coordination.

Creates *new* CSV reports; does not modify models, GUI, source or prior results.

Examples:
  .\\.venv\\Scripts\\python.exe scalability_benchmark.py --seeds 10
  .\\.venv\\Scripts\\python.exe scalability_benchmark.py --seeds 30
"""
from __future__ import annotations

import argparse
import csv
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import simulation
from dynamic_tasks import DynamicTaskGenerator
from auction_allocation import allocate_tasks_auction
from hungarian_v3 import allocate_tasks_hungarian_v3
from hungarian_v4 import allocate_tasks_hungarian_v4

ALGORITHMS = {
    "Auction": allocate_tasks_auction,
    "Hungarian V3": allocate_tasks_hungarian_v3,
    "Hungarian V4": allocate_tasks_hungarian_v4,
}
# The 5@30 case helps separate robot count from grid area.
CONFIGS = [(20, 20, 5), (20, 20, 8), (30, 30, 5), (30, 30, 10)]
TOTAL_TASKS = 40
MAX_STEPS = 750


def robot_positions(rows, cols, count):
    # Original five positions for the 20x20 reference, scaled to other sizes.
    middle = ((rows - 1) // 2, (cols - 1) // 2)
    if rows == 20 and cols == 20:
        middle = (9, 9)
    proposed = [
        (0, 0), (0, cols - 1), (rows - 1, 0),
        (rows - 1, cols - 1), middle,
        (0, cols // 2), (rows - 1, cols // 2),
        (rows // 2, 0), (rows // 2, cols - 1),
        (rows // 2, cols // 2 - 2),
    ]
    unique = []
    for position in proposed:
        if position not in unique:
            unique.append(position)
    if len(unique) < count:
        raise ValueError("Not enough distinct robot positions")
    return unique[:count]


def create_scenario(seed, rows, cols, robot_count):
    random.seed(seed)
    sim = simulation.Simulation(rows=rows, cols=cols)
    # Preserve the exact existing shelf layout at 20x20.
    if (rows, cols) == (20, 20):
        shelf_rows = (5, 10, 15)
        shelf_cols = range(2, 18)
    else:
        shelf_rows = tuple(range(5, rows, 5))
        shelf_cols = range(2, cols - 2)
    for row in shelf_rows:
        for col in shelf_cols:
            if col % 6 not in (0, 1):
                sim.environment.add_obstacle(row, col)

    for robot_id, position in enumerate(robot_positions(rows, cols, robot_count), 1):
        if not sim.add_robot(robot_id, position):
            raise RuntimeError(f"Robot {robot_id} cannot start at {position}")

    # Keep 5 moving obstacles for comparability of robot-count tests.
    for _ in range(5):
        for _attempt in range(1000):
            position = (random.randrange(rows), random.randrange(cols))
            r, c = position
            if sim.environment.grid[r][c] != 0:
                continue
            if position in {robot.position for robot in sim.robots}:
                continue
            if sim.environment.add_dynamic_obstacle(r, c):
                break
        else:
            raise RuntimeError("Could not place a moving obstacle")

    generator = DynamicTaskGenerator(
        simulation=sim,
        seed=seed + 10000,
        interval=5,
        tasks_per_interval=4,
        urgent_probability=0.45,
        max_new_tasks=TOTAL_TASKS,
    )
    return sim, generator


def run_one(seed, rows, cols, robot_count, algorithm_name, allocator):
    old_allocator = simulation.allocate_tasks
    result = {
        "seed": seed, "grid": f"{rows}x{cols}", "rows": rows, "cols": cols,
        "robots": robot_count, "algorithm": algorithm_name,
        "success": 0, "completed": 0, "total_tasks": 0,
        "finish": None, "distance": None, "collisions": None,
        "astar_calls": None, "replanning_events": None,
        "yield_events": None, "runtime_seconds": None,
        "p1_avg": None, "p3_avg": None, "error": "",
    }
    try:
        sim, generator = create_scenario(seed, rows, cols, robot_count)
        simulation.allocate_tasks = allocator
        start = time.perf_counter()
        for _ in range(MAX_STEPS):
            sim.step()
            if sim.time_step <= 50:
                generator.update()
            if (sim.time_step >= 50 and len(sim.tasks) == TOTAL_TASKS
                and all(t.status == "completed" for t in sim.tasks)):
                break
        result["runtime_seconds"] = round(time.perf_counter() - start, 6)
        completed = sum(t.status == "completed" for t in sim.tasks)
        result.update({
            "success": int(completed == TOTAL_TASKS and len(sim.tasks) == TOTAL_TASKS
                           and sim.collision_count == 0),
            "completed": completed,
            "total_tasks": len(sim.tasks), "finish": sim.time_step,
            "distance": sim.total_distance,
            "collisions": sim.collision_count,
            "astar_calls": sim.astar_calls,
            "replanning_events": sim.replanning_events,
            "yield_events": sim.yield_events,
        })
        for priority in (1, 3):
            durations = [t.completed_step - t.created_step for t in sim.tasks
                         if t.priority == priority and t.status == "completed"]
            if durations:
                result[f"p{priority}_avg"] = round(statistics.mean(durations), 6)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        simulation.allocate_tasks = old_allocator
    return result


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def summarize(results):
    groups = defaultdict(list)
    for row in results:
        groups[(row["grid"], row["robots"], row["algorithm"])].append(row)
    summary = []
    for (grid, robots, algorithm), records in groups.items():
        good = [r for r in records if r["success"]]
        out = {"grid": grid, "robots": robots, "algorithm": algorithm,
               "runs": len(records), "successful": len(good),
               "success_rate_pct": round(100 * len(good) / len(records), 2)}
        for field in ("finish", "distance", "runtime_seconds", "astar_calls",
                      "replanning_events", "yield_events", "p1_avg", "p3_avg"):
            values = [r[field] for r in good if r[field] is not None]
            out[f"mean_{field}"] = round(statistics.mean(values), 4) if values else None
        out["total_collisions"] = sum(r["collisions"] or 0 for r in records)
        out["error_runs"] = sum(bool(r["error"]) for r in records)
        summary.append(out)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Grid and robot-count scalability benchmark")
    parser.add_argument("--seeds", type=int, default=10,
                        help="Number of scenario seeds, e.g. 10 for smoke test or 30 for report")
    parser.add_argument("--start-seed", type=int, default=41000)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be >= 1")

    print("MULTI-ROBOT SCALABILITY BENCHMARK")
    print("Algorithms: " + ", ".join(ALGORITHMS))
    print(f"Configurations: {CONFIGS}; seeds per configuration: {args.seeds}")
    print("40 tasks per run, 5 moving obstacles, maximum 750 steps")
    print("Note: longer grid distances change problem difficulty; compare with care.\n")

    results = []
    for rows, cols, robots in CONFIGS:
        print(f"CONFIGURATION: {rows}x{cols}, {robots} robots", flush=True)
        for index in range(args.seeds):
            seed = args.start_seed + index
            for name, allocator in ALGORITHMS.items():
                result = run_one(seed, rows, cols, robots, name, allocator)
                results.append(result)
                status = "OK" if result["success"] else "FAILED"
                print(f"  seed={seed} {name:12s} {status:6s} "
                      f"done={result['completed']}/{result['total_tasks']} "
                      f"steps={result['finish']} error={result['error']}", flush=True)

    folder = ROOT / "results" / "scalability"
    folder.mkdir(parents=True, exist_ok=True)
    detailed = folder / "scalability_runs.csv"
    summary_file = folder / "scalability_summary.csv"
    write_csv(detailed, results)
    summary = summarize(results)
    write_csv(summary_file, summary)

    print("\nSUMMARY (metrics averaged over successful runs only)")
    print(f"{'Grid':8} {'Robots':>6} {'Algorithm':13} {'Success':>8} "
          f"{'Steps':>9} {'Distance':>10} {'Runtime(s)':>11}")
    for r in summary:
        fmt = lambda x: f"{x:.2f}" if x is not None else "N/A"
        print(f"{r['grid']:8} {r['robots']:6} {r['algorithm']:13} "
              f"{r['successful']}/{r['runs']:<5} "
              f"{fmt(r['mean_finish']):>9} {fmt(r['mean_distance']):>10} "
              f"{fmt(r['mean_runtime_seconds']):>11}")
    print("\nSaved:", detailed, "\n      ", summary_file)
    print("Existing source, GUI, checkpoints and prior benchmarks were not changed.")


if __name__ == "__main__":
    main()
