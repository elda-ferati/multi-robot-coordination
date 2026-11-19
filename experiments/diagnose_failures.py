
import sys
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import simulation
from benchmark_scenarios import SCENARIOS, create_simulation
from hungarian_allocation import allocate_tasks_hungarian
from hungarian_v2 import allocate_tasks_hungarian_v2
from hungarian_v3 import allocate_tasks_hungarian_v3
from pathfinding import astar


ALGORITHMS = {
    "Hungarian V1": allocate_tasks_hungarian,
    "Hungarian V2": allocate_tasks_hungarian_v2,
    "Hungarian V3": allocate_tasks_hungarian_v3,
}

CASES = [
    ("Small", 43, ["Hungarian V1", "Hungarian V2", "Hungarian V3"]),
    ("Very Large", 44, ["Hungarian V1", "Hungarian V2", "Hungarian V3"]),
]


def diagnose(config, seed, algorithm_name):
    print("\n" + "=" * 75)
    print(
        f"SCENARIO: {config['name']} | "
        f"SEED: {seed} | ALGORITHM: {algorithm_name}"
    )
    print("=" * 75)

    random.seed(seed)
    simulation.allocate_tasks = ALGORITHMS[algorithm_name]

    sim = create_simulation(config)

    # Run the same number of steps as the benchmark.
    for _ in range(config["steps"]):
        sim.step()

        if all(task.status == "completed" for task in sim.tasks):
            break

    completed = [
        task for task in sim.tasks
        if task.status == "completed"
    ]

    unfinished = [
        task for task in sim.tasks
        if task.status != "completed"
    ]

    print(f"\nSteps executed: {sim.time_step}")
    print(f"Completed tasks: {len(completed)}/{len(sim.tasks)}")
    print(f"Unfinished tasks: {len(unfinished)}")
    print(f"Total distance: {sim.total_distance}")
    print(f"Collisions: {sim.collision_count}")

    print("\n--- UNFINISHED TASKS ---")

    if not unfinished:
        print("All tasks completed successfully.")

    for task in unfinished:
        print(f"\nTask ID: {task.task_id}")
        print(f"Status: {task.status}")
        print(f"Priority: {task.priority}")
        print(f"Pickup: {task.pickup}")
        print(f"Delivery: {task.delivery}")
        print(f"Assigned robot: {task.assigned_robot}")

        delivery_path = astar(
            sim.environment,
            task.pickup,
            task.delivery
        )

        print(
            "Pickup-to-delivery path exists:",
            delivery_path is not None
        )

        for robot in sim.robots:
            if robot.current_task is task:
                print(f"\nResponsible Robot: {robot.robot_id}")
                print(f"Position: {robot.position}")
                print(f"Goal: {robot.goal}")
                print(f"Carrying package: {robot.carrying_package}")
                print(f"Remaining path length: {len(robot.path)}")
                print(
                    "Waiting steps:",
                    sim.robot_waiting[robot.robot_id]
                )
                print(
                    "Consecutive blocked steps:",
                    sim.blocked_steps[robot.robot_id]
                )

                if robot.goal is not None:
                    path = astar(
                        sim.environment,
                        robot.position,
                        robot.goal
                    )

                    blocked = {
                        other.position
                        for other in sim.robots
                        if other.robot_id != robot.robot_id
                    }

                    path_avoiding_robots = astar(
                        sim.environment,
                        robot.position,
                        robot.goal,
                        blocked_positions=blocked
                    )

                    print(
                        "Path ignoring other robots:",
                        path is not None
                    )
                    print(
                        "Path avoiding other robots:",
                        path_avoiding_robots is not None
                    )

    print("\n--- ROBOT STATES ---")

    for robot in sim.robots:
        print(
            f"Robot {robot.robot_id}: "
            f"position={robot.position}, "
            f"goal={robot.goal}, "
            f"task={robot.current_task.task_id if robot.current_task else None}, "
            f"waiting={sim.robot_waiting[robot.robot_id]}, "
            f"blocked={sim.blocked_steps[robot.robot_id]}"
        )

    print("\n--- FINAL DYNAMIC OBSTACLES ---")
    print(sim.environment.dynamic_obstacles)


def main():
    for scenario_name, seed, algorithm_names in CASES:
        config = next(
            scenario for scenario in SCENARIOS
            if scenario["name"] == scenario_name
        )

        for algorithm_name in algorithm_names:
            diagnose(config, seed, algorithm_name)


if __name__ == "__main__":
    main()
