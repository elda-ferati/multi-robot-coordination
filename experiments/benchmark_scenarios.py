
import sys
import csv
import random
import time
from pathlib import Path
from statistics import mean

# First, add the src folder to Python's search path.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Import project modules only AFTER configuring sys.path.
import simulation

from task_allocation import allocate_tasks as global_auction
from hungarian_allocation import allocate_tasks_hungarian
from hungarian_v2 import allocate_tasks_hungarian_v2
from hungarian_v3 import allocate_tasks_hungarian_v3
from pathfinding import astar

ALGORITHMS = {
    "Global Auction": global_auction,
    "Hungarian V1": allocate_tasks_hungarian,
    "Hungarian V2": allocate_tasks_hungarian_v2,
    "Hungarian V3": allocate_tasks_hungarian_v3,
}


SCENARIOS = [
    {"name": "Small", "size": 10, "robots": 2,
     "tasks": 5, "dynamic": 2, "steps": 200},
    {"name": "Medium", "size": 20, "robots": 5,
     "tasks": 12, "dynamic": 5, "steps": 400},
    {"name": "Large", "size": 40, "robots": 8,
     "tasks": 20, "dynamic": 12, "steps": 650},
    {"name": "Very Large", "size": 60, "robots": 12,
     "tasks": 30, "dynamic": 20, "steps": 900},
]

SEEDS = [42, 43, 44]


def create_simulation(config):
    size = config["size"]
    sim = simulation.Simulation(size, size)

    # Static shelves with several corridors
    for row in range(5, size - 2, 7):
        for col in range(2, size - 2):
            if col % 7 not in (0, 1):
                sim.environment.add_obstacle(row, col)

    def free_cells():
        return [
            (r, c)
            for r in range(size)
            for c in range(size)
            if sim.environment.grid[r][c] == 0
        ]

    # Place robots
    cells = free_cells()
    random.shuffle(cells)

    robot_positions = cells[:config["robots"]]
    assert len(robot_positions) == config["robots"]

    for i, position in enumerate(robot_positions, 1):
        assert sim.add_robot(i, position)

    occupied = set(robot_positions)

    # Place moving obstacles
    candidates = [
        p for p in free_cells()
        if p not in occupied
    ]
    random.shuffle(candidates)

    added = 0
    for row, col in candidates:
        if sim.environment.add_dynamic_obstacle(row, col):
            added += 1
        if added == config["dynamic"]:
            break

    assert added == config["dynamic"]

    # Generate reachable delivery tasks
    candidates = free_cells()
    created = 0
    attempts = 0
    max_attempts = config["tasks"] * 500

    while created < config["tasks"] and attempts < max_attempts:
        attempts += 1
        pickup, delivery = random.sample(candidates, 2)

        if astar(sim.environment, pickup, delivery) is None:
            continue

        if not any(
            astar(sim.environment, robot.position, pickup)
            is not None
            for robot in sim.robots
        ):
            continue

        priority = random.choice([1, 1, 2, 3])

        if sim.add_task(
            created + 1, pickup, delivery, priority
        ):
            created += 1

    if created != config["tasks"]:
        raise RuntimeError(
            f"Could only generate {created} tasks "
            f"for {config['name']}"
        )

    return sim


def run_one(config, seed, algorithm_name, allocator):
    random.seed(seed)
    simulation.allocate_tasks = allocator

    sim = create_simulation(config)

    start = time.perf_counter()
    error = ""

    try:
        for _ in range(config["steps"]):
            sim.step()

            if all(
                task.status == "completed"
                for task in sim.tasks
            ):
                break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    runtime = time.perf_counter() - start

    completed = sum(
        task.status == "completed"
        for task in sim.tasks
    )

    finished = completed == len(sim.tasks) and not error

    return {
        "scenario": config["name"],
        "grid": f"{config['size']}x{config['size']}",
        "algorithm": algorithm_name,
        "seed": seed,
        "robots": config["robots"],
        "tasks": config["tasks"],
        "completed": completed,
        "completion_rate": round(
            100 * completed / len(sim.tasks), 2
        ),
        "finish_step": sim.time_step if finished else "",
        "total_distance": sim.total_distance,
        "waiting_steps": sum(sim.robot_waiting.values()),
        "collisions": sim.collision_count,
        "runtime_seconds": round(runtime, 6),
        "error": error,
    }


def main():
    results = []

    for config in SCENARIOS:
        print(f"\n=== {config['name']} "
              f"{config['size']}x{config['size']} ===")

        for seed in SEEDS:
            for name, allocator in ALGORITHMS.items():

                # Factorial Global Auction is impractical
                # for large robot counts.
                if name == "Global Auction" and config["robots"] > 5:
                    continue

                result = run_one(
                    config, seed, name, allocator
                )
                results.append(result)

                print(
                    f"{name:15} seed={seed} "
                    f"done={result['completed']}/{result['tasks']} "
                    f"distance={result['total_distance']} "
                    f"finish={result['finish_step']} "
                    f"time={result['runtime_seconds']:.4f}s "
                    f"error={result['error'] or 'none'}"
                )

    output = ROOT / "results" / "benchmark_scenarios.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\n=== SUMMARY ===")

    for config in SCENARIOS:
        for name in ALGORITHMS:
            group = [
                r for r in results
                if r["scenario"] == config["name"]
                and r["algorithm"] == name
            ]

            if not group:
                continue

            finish_values = [
                r["finish_step"] for r in group
                if r["finish_step"] != ""
            ]

            print(
                f"{config['name']:12} {name:15} "
                f"completion={mean(r['completion_rate'] for r in group):.1f}% "
                f"avg_distance={mean(r['total_distance'] for r in group):.1f} "
                f"avg_finish={mean(finish_values) if finish_values else 'N/A'} "
                f"avg_time={mean(r['runtime_seconds'] for r in group):.4f}s"
            )

    print(f"\nSaved {len(results)} runs to {output}")


if __name__ == "__main__":
    main()
