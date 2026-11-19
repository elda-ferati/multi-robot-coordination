
from itertools import combinations, permutations
from pathfinding import astar


def calculate_task_cost(robot, task, environment):
    pickup_path = astar(
        environment,
        robot.position,
        task.pickup
    )

    delivery_path = astar(
        environment,
        task.pickup,
        task.delivery
    )

    if pickup_path is None or delivery_path is None:
        return float("inf")

    return (
        len(pickup_path) - 1
        + len(delivery_path) - 1
    )


def allocate_tasks(robots, tasks, environment):
    available_robots = [
        robot for robot in robots
        if robot.current_task is None
        and robot.goal is None
    ]

    pending_tasks = sorted(
        [
            task for task in tasks
            if task.status == "pending"
        ],
        key=lambda task: (
            -task.priority,
            task.task_id
        )
    )

    if not available_robots or not pending_tasks:
        return

    # Prioritize urgent tasks
    selected_tasks = pending_tasks[
        :len(available_robots)
    ]

    best_assignment = None
    best_cost = float("inf")

    number = min(
        len(available_robots),
        len(selected_tasks)
    )

    # Evaluate possible robot-task combinations
    for robot_subset in combinations(
        available_robots, number
    ):
        for task_order in permutations(
            selected_tasks, number
        ):
            total_cost = 0
            assignment = []

            for robot, task in zip(
                robot_subset, task_order
            ):
                cost = calculate_task_cost(
                    robot,
                    task,
                    environment
                )

                if cost == float("inf"):
                    total_cost = float("inf")
                    break

                total_cost += cost
                assignment.append(
                    (robot, task)
                )

            if total_cost < best_cost:
                best_cost = total_cost
                best_assignment = assignment

    if best_assignment is None:
        return

    for robot, task in best_assignment:
        robot.assign_task(
            task,
            environment
        )
