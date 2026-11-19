
import sys
import random
import time
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import simulation
from dynamic_tasks import DynamicTaskGenerator
from hungarian_v3 import allocate_tasks_hungarian_v3
from hungarian_v4 import allocate_tasks_hungarian_v4


ALGORITHMS = {
    "V3": allocate_tasks_hungarian_v3,
    "V4": allocate_tasks_hungarian_v4,
}

SEEDS = range(100, 200)
MAX_STEPS = 500


def run_scenario(seed, algorithm_name, allocator):
    random.seed(seed)

    simulation.allocate_tasks = allocator

    sim = simulation.Simulation(rows=20, cols=20)

    # Static warehouse shelves
    for row in [5, 10, 15]:
        for col in range(2, 18):
            if col % 6 not in (0, 1):
                sim.environment.add_obstacle(row, col)

    positions = [
        (0, 0), (0, 19),
        (19, 0), (19, 19),
        (9, 9)
    ]

    for i, pos in enumerate(positions, 1):
        assert sim.add_robot(i, pos)

    # Dynamic obstacles
    for _ in range(5):
        for _attempt in range(100):
            pos = (
                random.randrange(20),
                random.randrange(20)
            )

            if sim.environment.grid[pos[0]][pos[1]] != 0:
                continue

            if pos in [r.position for r in sim.robots]:
                continue

            sim.environment.add_dynamic_obstacle(*pos)
            break

    generator = DynamicTaskGenerator(
        simulation=sim,
        seed=seed + 10000,
        interval=5,
        tasks_per_interval=4,
        urgent_probability=0.45,
        max_new_tasks=40
    )

    start = time.perf_counter()
    error = None

    try:
        for _ in range(MAX_STEPS):
            sim.step()

            if sim.time_step <= 50:
                generator.update()

            if (
                sim.time_step >= 50
                and len(sim.tasks) == 40
                and all(
                    t.status == "completed"
                    for t in sim.tasks
                )
            ):
                break
    except Exception as exc:
        error = str(exc)

    runtime = time.perf_counter() - start

    result = {
        "algorithm": algorithm_name,
        "seed": seed,
        "completed": sum(
            t.status == "completed" for t in sim.tasks
        ),
        "total_tasks": len(sim.tasks),
        "finish": sim.time_step,
        "distance": sim.total_distance,
        "collisions": sim.collision_count,
        "runtime": runtime,
        "error": error,
    }

    for priority in (1, 3):
        group = [
            t for t in sim.tasks
            if t.priority == priority
        ]

        durations = [
            t.completed_step - t.created_step
            for t in group
            if t.status == "completed"
        ]

        result[f"p{priority}_count"] = len(group)
        result[f"p{priority}_completed"] = len(durations)
        result[f"p{priority}_avg"] = (
            sum(durations) / len(durations)
            if durations else None
        )

    return result


def summarize(results, algorithm):
    rows = [
        r for r in results
        if r["algorithm"] == algorithm
    ]

    successful = [
        r for r in rows
        if r["error"] is None
        and r["completed"] == 40
        and r["total_tasks"] == 40
    ]

    print(f"\n{algorithm}")
    print("-" * 50)
    print(f"Successful: {len(successful)}/{len(rows)}")

    if not successful:
        return

    for key in [
        "finish", "distance",
        "p1_avg", "p3_avg", "runtime"
    ]:
        values = [
            r[key] for r in successful
            if r[key] is not None
        ]

        if values:
            print(
                f"{key}: "
                f"{sum(values) / len(values):.3f}"
            )

    print(
        "Total collisions:",
        sum(r["collisions"] for r in rows)
    )


def main():
    results = []

    for seed in SEEDS:
        for name, allocator in ALGORITHMS.items():
            result = run_scenario(
                seed, name, allocator
            )
            results.append(result)

            print(
                f"{name} seed={seed} "
                f"done={result['completed']}/"
                f"{result['total_tasks']} "
                f"finish={result['finish']} "
                f"distance={result['distance']} "
                f"error={result['error']}"
            )

    for name in ALGORITHMS:
        summarize(results, name)

    output = ROOT / "results"
    output.mkdir(exist_ok=True)

    filepath = output / "benchmark_v3_v4_stress.csv"

    with open(
        filepath, "w",
        newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nSaved results to:", filepath)


if __name__ == "__main__":
    main()
