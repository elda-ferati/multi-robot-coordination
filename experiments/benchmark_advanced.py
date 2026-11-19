
import sys
import csv
import random
import time
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import simulation
from benchmark_scenarios import (
    SCENARIOS,
    create_simulation,
)
from task_allocation import allocate_tasks as global_auction
from hungarian_allocation import allocate_tasks_hungarian
from hungarian_v2 import allocate_tasks_hungarian_v2
from hungarian_v3 import allocate_tasks_hungarian_v3


# Increase to 30 later if results require it.
SEEDS = list(range(100, 120))

ALGORITHMS = {
    "Global Auction": global_auction,
    "Hungarian V1": allocate_tasks_hungarian,
    "Hungarian V2": allocate_tasks_hungarian_v2,
    "Hungarian V3": allocate_tasks_hungarian_v3,
}

OBSTACLE_LEVELS = {
    "Normal": 1.0,
    "High": 1.5,
}


def scenario_config(base, multiplier):
    config = base.copy()

    config["dynamic"] = max(
        1,
        round(base["dynamic"] * multiplier)
    )

    return config


def run_experiment(config, seed, algorithm_name, allocator, level):
    random.seed(seed)

    # Keep the allocation algorithm separate from
    # the movement and collision-handling logic.
    simulation.allocate_tasks = allocator

    result = {
        "scenario": config["name"],
        "grid": f"{config['size']}x{config['size']}",
        "obstacle_level": level,
        "dynamic_obstacles": config["dynamic"],
        "algorithm": algorithm_name,
        "seed": seed,
        "robots": config["robots"],
        "tasks": config["tasks"],
        "completed": 0,
        "completion_rate": 0.0,
        "finish_step": "",
        "total_distance": 0,
        "waiting_steps": 0,
        "yield_events": 0,
        "yield_moves": 0,
        "failed_yield_attempts": 0,
        "collisions": 0,
        "runtime_seconds": 0.0,
        "error": "",
    }

    start = time.perf_counter()
    sim = None

    try:
        sim = create_simulation(config)

        for _ in range(config["steps"]):
            sim.step()

            if all(
                task.status == "completed"
                for task in sim.tasks
            ):
                break

    except Exception as exc:
        result["error"] = (
            f"{type(exc).__name__}: {exc}"
        )

    result["runtime_seconds"] = round(
        time.perf_counter() - start, 6
    )

    if sim is None:
        return result

    completed = sum(
        task.status == "completed"
        for task in sim.tasks
    )

    total_tasks = len(sim.tasks)

    result["completed"] = completed
    result["completion_rate"] = round(
        100 * completed / total_tasks, 2
    ) if total_tasks else 0.0

    if (
        completed == total_tasks
        and total_tasks > 0
        and not result["error"]
    ):
        result["finish_step"] = sim.time_step

    result["total_distance"] = sim.total_distance
    result["waiting_steps"] = sum(
        sim.robot_waiting.values()
    )
    result["yield_events"] = sim.yield_events
    result["yield_moves"] = sim.yield_moves
    result["failed_yield_attempts"] = (
        sim.failed_yield_attempts
    )
    result["collisions"] = sim.collision_count

    return result


def print_summary(results):
    print("\n" + "=" * 100)
    print("ADVANCED BENCHMARK SUMMARY")
    print("=" * 100)

    for scenario in SCENARIOS:
        for level in OBSTACLE_LEVELS:
            print(
                f"\n{scenario['name']} | "
                f"Obstacles: {level}"
            )

            for algorithm_name in ALGORITHMS:
                group = [
                    row for row in results
                    if row["scenario"] == scenario["name"]
                    and row["obstacle_level"] == level
                    and row["algorithm"] == algorithm_name
                ]

                if not group:
                    continue

                successful = [
                    row for row in group
                    if row["finish_step"] != ""
                ]

                completion = mean(
                    row["completion_rate"]
                    for row in group
                )

                success_rate = (
                    100 * len(successful) / len(group)
                )

                avg_distance = mean(
                    row["total_distance"]
                    for row in group
                )

                avg_waiting = mean(
                    row["waiting_steps"]
                    for row in group
                )

                avg_yield = mean(
                    row["yield_events"]
                    for row in group
                )

                avg_runtime = mean(
                    row["runtime_seconds"]
                    for row in group
                )

                avg_finish = (
                    round(
                        mean(
                            row["finish_step"]
                            for row in successful
                        ), 2
                    )
                    if successful else "N/A"
                )

                errors = sum(
                    bool(row["error"])
                    for row in group
                )

                collisions = sum(
                    row["collisions"]
                    for row in group
                )

                print(
                    f"{algorithm_name:15} "
                    f"success={success_rate:5.1f}% "
                    f"completion={completion:5.1f}% "
                    f"distance={avg_distance:8.1f} "
                    f"finish={str(avg_finish):>7} "
                    f"waiting={avg_waiting:8.1f} "
                    f"yield={avg_yield:6.1f} "
                    f"runtime={avg_runtime:.4f}s "
                    f"collisions={collisions} "
                    f"errors={errors}"
                )


def main():
    results = []

    for base_config in SCENARIOS:
        for level, multiplier in OBSTACLE_LEVELS.items():
            config = scenario_config(
                base_config, multiplier
            )

            print(
                f"\n=== {config['name']} "
                f"{config['size']}x{config['size']} "
                f"| {level} obstacles "
                f"({config['dynamic']}) ==="
            )

            for seed in SEEDS:
                for name, allocator in ALGORITHMS.items():

                    # Exhaustive Global Auction is
                    # too expensive for >5 robots.
                    if (
                        name == "Global Auction"
                        and config["robots"] > 5
                    ):
                        continue

                    result = run_experiment(
                        config,
                        seed,
                        name,
                        allocator,
                        level
                    )

                    results.append(result)

                    print(
                        f"{name:15} "
                        f"seed={seed} "
                        f"done={result['completed']}/"
                        f"{result['tasks']} "
                        f"distance={result['total_distance']} "
                        f"finish={result['finish_step']} "
                        f"yield={result['yield_events']} "
                        f"error={result['error'] or 'none'}"
                    )

    output = (
        ROOT / "results" /
        "benchmark_advanced.csv"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys())
        )

        writer.writeheader()
        writer.writerows(results)

    print_summary(results)

    print(
        f"\nSaved {len(results)} runs to {output}"
    )


if __name__ == "__main__":
    main()
