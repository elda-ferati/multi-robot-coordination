
import sys
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import simulation

from dynamic_tasks import DynamicTaskGenerator
from hungarian_allocation import allocate_tasks_hungarian
from hungarian_v2 import allocate_tasks_hungarian_v2
from hungarian_v3 import allocate_tasks_hungarian_v3
from hungarian_v4 import allocate_tasks_hungarian_v4


ALGORITHMS = {
    "Hungarian V3": allocate_tasks_hungarian_v3,
    "Hungarian V4": allocate_tasks_hungarian_v4,
}

SEEDS = range(42, 52)
MAX_STEPS = 350
GENERATION_END = 100


def create_environment(seed):
    random.seed(seed)

    sim = simulation.Simulation(rows=20, cols=20)

    for row in [5, 10, 15]:
        for col in range(2, 18):
            if col % 6 not in (0, 1):
                sim.environment.add_obstacle(row, col)

    positions = [
        (0, 0),
        (0, 19),
        (19, 0),
        (19, 19),
        (9, 9),
    ]

    for i, position in enumerate(positions, 1):
        if not sim.add_robot(i, position):
            raise RuntimeError("Cannot add robot")

    for _ in range(5):
        for attempt in range(100):
            row = random.randrange(20)
            col = random.randrange(20)

            if sim.environment.grid[row][col] != 0:
                continue

            if any(
                r.position == (row, col)
                for r in sim.robots
            ):
                continue

            sim.environment.add_dynamic_obstacle(
                row, col
            )
            break
        else:
            raise RuntimeError(
                "Cannot place dynamic obstacle"
            )

    return sim


def run_test(name, allocator, seed):
    sim = create_environment(seed)
    simulation.allocate_tasks = allocator

    generator = DynamicTaskGenerator(
        simulation=sim,
        seed=seed + 1000,
        interval=10,
        tasks_per_interval=2,
        urgent_probability=0.30,
        max_new_tasks=20,
    )

    assignment_times = {}
    pickup_times = {}

    for _ in range(MAX_STEPS):
        sim.step()

        if sim.time_step <= GENERATION_END:
            generator.update()

        for task in sim.tasks:
            if (
                task.task_id not in assignment_times
                and task.status != "pending"
            ):
                assignment_times[task.task_id] = (
                    sim.time_step
                )

            if (
                task.task_id not in pickup_times
                and task.status in ("picked_up", "completed")
            ):
                pickup_times[task.task_id] = (
                    sim.time_step
                )

        if (
            sim.time_step >= GENERATION_END
            and all(
                task.status == "completed"
                for task in sim.tasks
            )
        ):
            break

    records = []

    for task in sim.tasks:
        created = task.created_step
        assigned = assignment_times.get(task.task_id)
        picked = pickup_times.get(task.task_id)
        completed = task.completed_step

        records.append({
            "priority": task.priority,
            "waiting": (
                assigned - created
                if assigned is not None else None
            ),
            "response": (
                picked - created
                if picked is not None else None
            ),
            "completion": (
                completed - created
                if completed is not None else None
            ),
            "finished": completed is not None,
        })

    return {
        "algorithm": name,
        "seed": seed,
        "records": records,
        "total_distance": sim.total_distance,
        "finish_step": sim.time_step,
        "collisions": sim.collision_count,
        "yield_events": sim.yield_events,
    }


def average(values):
    valid = [v for v in values if v is not None]
    return (
        round(statistics.mean(valid), 2)
        if valid else None
    )


def main():
    results = []

    for name, allocator in ALGORITHMS.items():
        for seed in SEEDS:
            result = run_test(name, allocator, seed)
            results.append(result)

            completed = sum(
                record["finished"]
                for record in result["records"]
            )

            print(
                f"{name:15} seed={seed} "
                f"completed={completed}/20 "
                f"distance={result['total_distance']} "
                f"finish={result['finish_step']} "
                f"collisions={result['collisions']}"
            )

    print("\n" + "=" * 65)
    print("PRIORITY BENCHMARK SUMMARY")
    print("=" * 65)

    for name in ALGORITHMS:
        group = [
            result for result in results
            if result["algorithm"] == name
        ]

        print(f"\n{name}")

        for priority in [1, 3]:
            records = [
                record
                for result in group
                for record in result["records"]
                if record["priority"] == priority
            ]

            completed = sum(
                record["finished"]
                for record in records
            )

            print(
                f"Priority {priority}: "
                f"completed={completed}/{len(records)} "
                f"waiting={average([r['waiting'] for r in records])} "
                f"response={average([r['response'] for r in records])} "
                f"completion={average([r['completion'] for r in records])}"
            )

        print(
            "Average total distance:",
            average([
                r["total_distance"]
                for r in group
            ])
        )

        print(
            "Average finish step:",
            average([
                r["finish_step"]
                for r in group
            ])
        )

        print(
            "Total collisions:",
            sum(r["collisions"] for r in group)
        )


if __name__ == "__main__":
    main()
