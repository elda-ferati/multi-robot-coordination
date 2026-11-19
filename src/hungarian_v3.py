
from math import inf

from hungarian_allocation import hungarian
from pathfinding import astar


def allocate_tasks_hungarian_v3(robots, tasks, environment):
    """
    Hungarian V3:
    - Respects task priorities.
    - Considers all pending tasks at each priority level.
    - Caches pickup-to-delivery distances.
    - Handles unreachable tasks.
    - Minimizes assignment cost within each priority level.
    """

    available = [
        robot for robot in robots
        if robot.current_task is None and robot.goal is None
    ]

    pending = sorted(
        [task for task in tasks if task.status == "pending"],
        key=lambda task: (-task.priority, task.task_id)
    )

    if not available or not pending:
        return

    # Calculate pickup-to-delivery distances once per task.
    delivery_costs = {}

    for task in pending:
        path = astar(
            environment,
            task.pickup,
            task.delivery
        )

        delivery_costs[task.task_id] = (
            len(path) - 1 if path is not None else inf
        )

    priorities = sorted(
        {task.priority for task in pending},
        reverse=True
    )

    for priority in priorities:
        if not available:
            break

        group = [
            task for task in pending
            if task.priority == priority
        ]

        if not group:
            continue

        # Build robot-task cost matrix.
        costs = []

        for robot in available:
            row = []

            for task in group:
                delivery_cost = delivery_costs[task.task_id]

                if delivery_cost == inf:
                    row.append(inf)
                    continue

                pickup_path = astar(
                    environment,
                    robot.position,
                    task.pickup
                )

                if pickup_path is None:
                    row.append(inf)
                else:
                    row.append(
                        len(pickup_path) - 1 + delivery_cost
                    )

            costs.append(row)

        feasible = [
            cost
            for row in costs
            for cost in row
            if cost != inf
        ]

        if not feasible:
            continue

        # Dynamic penalties.
        n = len(available)
        max_cost = max(feasible)

        dummy_penalty = n * max_cost + 1
        impossible_penalty = (n + 1) * dummy_penalty

        matrix = []

        for row in costs:
            matrix.append(
                [
                    impossible_penalty if cost == inf else cost
                    for cost in row
                ] + [dummy_penalty] * n
            )

        assignments = hungarian(matrix)

        successful = set()

        for robot_index, task_index in assignments:
            if task_index >= len(group):
                continue

            if costs[robot_index][task_index] == inf:
                continue

            robot = available[robot_index]
            task = group[task_index]

            if robot.assign_task(task, environment):
                successful.add(robot.robot_id)

        available = [
            robot for robot in available
            if robot.robot_id not in successful
        ]
