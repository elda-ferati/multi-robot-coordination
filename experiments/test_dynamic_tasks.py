
import sys
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import simulation

from dynamic_tasks import DynamicTaskGenerator
from hungarian_v3 import allocate_tasks_hungarian_v3


def main():
    random.seed(42)

    simulation.allocate_tasks = (
        allocate_tasks_hungarian_v3
    )

    sim = simulation.Simulation(
        rows=20,
        cols=20
    )

    # Add warehouse shelves.
    for row in [5, 10, 15]:
        for col in range(2, 18):
            if col % 6 not in (0, 1):
                sim.environment.add_obstacle(
                    row, col
                )

    robot_positions = [
        (0, 0),
        (0, 19),
        (19, 0),
        (19, 19),
        (9, 9)
    ]

    for index, position in enumerate(
        robot_positions, start=1
    ):
        if not sim.add_robot(index, position):
            raise RuntimeError(
                f"Cannot add robot at {position}"
            )

    # Add moving obstacles.
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
                "Cannot place dynamic obstacle"
            )

    generator = DynamicTaskGenerator(
        simulation=sim,
        seed=123,
        interval=10,
        tasks_per_interval=2,
        urgent_probability=0.30,
        max_new_tasks=20
    )

    print("\nDYNAMIC TASK SIMULATION")
    print("=" * 50)

    # Generate tasks only during the first
    # 100 steps; allow up to 300 steps to finish.
    for _ in range(300):
        sim.step()

        if sim.time_step <= 100:
            new_events = generator.update()

            for event in new_events:
                print(
                    f"Step {event['step']}: "
                    f"New task {event['task_id']} "
                    f"priority={event['priority']} "
                    f"pickup={event['pickup']} "
                    f"delivery={event['delivery']}"
                )

        completed = sum(
            task.status == "completed"
            for task in sim.tasks
        )

        if sim.time_step % 20 == 0:
            print(
                f"Step {sim.time_step}: "
                f"completed={completed}/"
                f"{len(sim.tasks)} "
                f"distance={sim.total_distance}"
            )

        if (
            sim.time_step >= 100
            and all(
                task.status == "completed"
                for task in sim.tasks
            )
        ):
            break

    print("\nFINAL RESULTS")
    print("=" * 50)

    sim.print_metrics()

    normal_tasks = [
        task for task in sim.tasks
        if task.priority == 1
    ]

    urgent_tasks = [
        task for task in sim.tasks
        if task.priority == 3
    ]

    print("\nDYNAMIC TASK METRICS")
    print("--------------------")
    print(
        "Generated tasks:",
        generator.generated_tasks
    )
    print(
        "Normal tasks:",
        len(normal_tasks)
    )
    print(
        "Urgent tasks:",
        len(urgent_tasks)
    )

    print(
        "Completed urgent:",
        sum(
            task.status == "completed"
            for task in urgent_tasks
        ),
        "/",
        len(urgent_tasks)
    )

    print(
        "Completed normal:",
        sum(
            task.status == "completed"
            for task in normal_tasks
        ),
        "/",
        len(normal_tasks)
    )

    unfinished = [
        task.task_id
        for task in sim.tasks
        if task.status != "completed"
    ]

    print("Unfinished task IDs:", unfinished)


if __name__ == "__main__":
    main()
