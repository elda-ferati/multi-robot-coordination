
from math import inf

from hungarian_allocation import hungarian
from pathfinding import astar


def allocate_tasks_hungarian_v2(robots, tasks, environment):
    """
    Same task-selection policy as Global Auction:
    1. Select highest-priority pending tasks.
    2. Select up to one task per available robot.
    3. Minimize total assignment cost using Hungarian.
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

    selected = pending[:len(available)]

    # Precompute each task's pickup-to-delivery distance.
    delivery_costs = {}

    for task in selected:
        path = astar(
            environment,
            task.pickup,
            task.delivery
        )

        delivery_costs[task.task_id] = (
            len(path) - 1 if path is not None else inf
        )

    costs = []

    for robot in available:
        row = []

        for task in selected:
            path = astar(
                environment,
                robot.position,
                task.pickup
            )

            if path is None or delivery_costs[task.task_id] == inf:
                row.append(inf)
            else:
                row.append(
                    len(path) - 1
                    + delivery_costs[task.task_id]
                )

        costs.append(row)

    # A dummy column represents an unassigned robot.
    # Its penalty must be greater than every feasible
    # real assignment cost.
    feasible = [
        cost
        for row in costs
        for cost in row
        if cost != inf
    ]

    if not feasible:
        return

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
            ]
            + [dummy_penalty] * n
        )

    assignments = hungarian(matrix)

    for robot_index, task_index in assignments:
        if task_index >= len(selected):
            continue

        if costs[robot_index][task_index] == inf:
            continue

        robot = available[robot_index]
        task = selected[task_index]

        robot.assign_task(task, environment)
