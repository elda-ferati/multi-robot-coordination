
from pathfinding import astar


def allocate_tasks(robots, tasks, environment):
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

    for task in pending_tasks:
        best_robot = None
        best_cost = float("inf")

        delivery_path = astar(
            environment,
            task.pickup,
            task.delivery
        )

        if delivery_path is None:
            continue

        for robot in robots:
            if robot.current_task is not None:
                continue

            pickup_path = astar(
                environment,
                robot.position,
                task.pickup
            )

            if pickup_path is None:
                continue

            cost = (
                len(pickup_path) - 1
                + len(delivery_path) - 1
            )

            if cost < best_cost:
                best_cost = cost
                best_robot = robot

        if best_robot is not None:
            best_robot.assign_task(
                task,
                environment
            )
