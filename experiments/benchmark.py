
import sys
import csv
import random
import time
from pathlib import Path

# Project folders
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simulation import Simulation
import simulation
from task_allocation import allocate_tasks as global_auction
from hungarian_allocation import allocate_tasks_hungarian
from hungarian_v2 import allocate_tasks_hungarian_v2


ALGORITHMS = {
    "Global Auction": global_auction,
    "Hungarian V1": allocate_tasks_hungarian,
    "Hungarian V2": allocate_tasks_hungarian_v2,
}

SEEDS = [42, 43, 44, 45, 46]
STEPS = 150


def create_simulation():
    sim = Simulation(20, 20)

    # Static warehouse shelves
    for row in [5, 10, 15]:
        for col in range(3, 17):
            if col not in [6, 13]:
                sim.environment.add_obstacle(row, col)

    # Dynamic obstacles
    assert sim.environment.add_dynamic_obstacle(7, 8)
    assert sim.environment.add_dynamic_obstacle(12, 12)

    # Robots
    assert sim.add_robot(1, (0, 0))
    assert sim.add_robot(2, (0, 19))
    assert sim.add_robot(3, (19, 10))

    # Delivery tasks
    assert sim.add_task(1, (2, 2), (18, 18), priority=1)
    assert sim.add_task(2, (2, 17), (18, 2), priority=3)
    assert sim.add_task(3, (17, 10), (1, 10), priority=2)
    assert sim.add_task(4, (8, 3), (12, 17), priority=1)
    assert sim.add_task(5, (12, 3), (3, 15), priority=3)
    assert sim.add_task(6, (17, 17), (2, 3), priority=2)

    return sim


def run_experiment(algorithm_name, allocation_function, seed):
    random.seed(seed)

    # Simulation.step uses this function
    simulation.allocate_tasks = allocation_function

    sim = create_simulation()

    start_time = time.perf_counter()
    sim.run(STEPS)
    runtime = time.perf_counter() - start_time

    completed = sum(
        task.status == "completed"
        for task in sim.tasks
    )

    finish_step = None

    for state in sim.history:
        if all(
            task["status"] == "completed"
            for task in state["tasks"]
        ):
            finish_step = state["step"]
            break

    return {
        "algorithm": algorithm_name,
        "seed": seed,
        "grid_size": "20x20",
        "robots": len(sim.robots),
        "tasks": len(sim.tasks),
        "completed_tasks": completed,
        "completion_rate": round(
            completed / len(sim.tasks) * 100, 2
        ),
        "finish_step": finish_step,
        "total_distance": sim.total_distance,
        "collisions": sim.collision_count,
        "astar_calls_simulation": sim.astar_calls,
        "replanning_events": sim.replanning_events,
        "total_waiting_steps": sum(
            sim.robot_waiting.values()
        ),
        "runtime_seconds": round(runtime, 6),
    }


def main():
    results = []

    for seed in SEEDS:
        for name, function in ALGORITHMS.items():
            result = run_experiment(name, function, seed)
            results.append(result)

            print(
                f"{name:16} | "
                f"seed={seed} | "
                f"completed={result['completed_tasks']}/"
                f"{result['tasks']} | "
                f"distance={result['total_distance']} | "
                f"finish={result['finish_step']} | "
                f"time={result['runtime_seconds']:.6f}s"
            )

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_file = results_dir / "benchmark_v2.csv"

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    print()
    print(f"Saved {len(results)} experiments to {output_file}")

    print("\nAVERAGE RESULTS")
    print("-" * 75)

    for name in ALGORITHMS:
        group = [
            result for result in results
            if result["algorithm"] == name
        ]

        avg_distance = sum(
            result["total_distance"]
            for result in group
        ) / len(group)

        avg_runtime = sum(
            result["runtime_seconds"]
            for result in group
        ) / len(group)

        avg_finish = (
            sum(
                result["finish_step"]
                for result in group
                if result["finish_step"] is not None
            ) /
            sum(
                result["finish_step"] is not None
                for result in group
            )
            if any(
                result["finish_step"] is not None
                for result in group
            )
            else None
        )

        print(
            f"{name:16} | "
            f"avg distance={avg_distance:.2f} | "
            f"avg finish={avg_finish} | "
            f"avg runtime={avg_runtime:.6f}s"
        )


if __name__ == "__main__":
    main()
