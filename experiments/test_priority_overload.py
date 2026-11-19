
import sys
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import simulation

from dynamic_tasks import DynamicTaskGenerator
from hungarian_v4 import allocate_tasks_hungarian_v4


def main():
    random.seed(42)

    # Use Hungarian V4
    simulation.allocate_tasks = allocate_tasks_hungarian_v4

    sim = simulation.Simulation(rows=20, cols=20)

    # Static obstacles
    for row in [5, 10, 15]:
        for col in range(2, 18):
            if col % 6 not in (0, 1):
                sim.environment.add_obstacle(row, col)

    # Robot starting positions
    positions = [
        (0, 0),
        (0, 19),
        (19, 0),
        (19, 19),
        (9, 9),
    ]

    for i, position in enumerate(positions, 1):
        if not sim.add_robot(i, position):
            raise RuntimeError(
                "Could not add robot"
            )

    # Dynamic obstacles
    for _ in range(5):
        for attempt in range(100):
            row = random.randrange(20)
            col = random.randrange(20)

            if sim.environment.grid[row][col] != 0:
                continue

            if any(
                robot.position == (row, col)
                for robot in sim.robots
            ):
                continue

            sim.environment.add_dynamic_obstacle(
                row, col
            )
            break
        else:
            raise RuntimeError(
                "Could not place obstacle"
            )

    # Dynamic task generator
    generator = DynamicTaskGenerator(
        simulation=sim,
        seed=777,
        interval=5,
        tasks_per_interval=4,
        urgent_probability=0.45,
        max_new_tasks=40
    )

    print("\nPRIORITY OVERLOAD TEST - HUNGARIAN V4")
    print("=" * 60)

    for _ in range(400):
        sim.step()

        if sim.time_step <= 50:
            events = generator.update()

            for event in events:
                print(
                    f"Step {event['step']}: "
                    f"Task {event['task_id']} "
                    f"priority={event['priority']}"
                )

        if sim.time_step % 10 == 0:
            busy = sum(
                robot.current_task is not None
                for robot in sim.robots
            )

            pending_urgent = sum(
                task.status == "pending"
                and task.priority == 3
                for task in sim.tasks
            )

            completed = sum(
                task.status == "completed"
                for task in sim.tasks
            )

            print(
                f"Step={sim.time_step:3} "
                f"busy={busy}/5 "
                f"pending_urgent={pending_urgent} "
                f"completed={completed}/"
                f"{len(sim.tasks)}"
            )

        if (
            sim.time_step >= 50
            and all(
                task.status == "completed"
                for task in sim.tasks
            )
        ):
            break

    print("\nFINAL RESULTS")
    print("=" * 60)

    sim.print_metrics()

    # Priority-specific metrics
    for priority in [1, 3]:
        tasks = [
            task for task in sim.tasks
            if task.priority == priority
        ]

        completed = [
            task for task in tasks
            if task.status == "completed"
        ]

        durations = [
            task.completed_step - task.created_step
            for task in completed
            if task.completed_step is not None
            and task.created_step is not None
        ]

        avg_duration = (
            sum(durations) / len(durations)
            if durations else None
        )

        print(
            f"Priority {priority}: "
            f"completed={len(completed)}/"
            f"{len(tasks)}, "
            f"avg_completion_time={avg_duration}"
        )


if __name__ == "__main__":
    main()
