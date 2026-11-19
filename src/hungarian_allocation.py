
from math import inf
from pathfinding import astar


def hungarian(cost_matrix):
    """
    Minimum-cost assignment for a rectangular matrix.
    Returns (row_index, column_index) pairs.
    Requires number of rows <= number of columns.
    """
    n = len(cost_matrix)
    if n == 0:
        return []

    m = len(cost_matrix[0])
    if m < n or any(len(row) != m for row in cost_matrix):
        raise ValueError("Invalid cost matrix dimensions")

    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (m + 1)
        used = [False] * (m + 1)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = 0

            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost_matrix[i0 - 1][j - 1] - u[i0] - v[j]

                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0

                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    return [(p[j] - 1, j - 1) for j in range(1, m + 1) if p[j] != 0]


def calculate_task_cost(robot, task, environment):
    pickup_path = astar(environment, robot.position, task.pickup)
    delivery_path = astar(environment, task.pickup, task.delivery)

    if pickup_path is None or delivery_path is None:
        return inf

    return len(pickup_path) + len(delivery_path) - 2


def allocate_tasks_hungarian(robots, tasks, environment):
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

    # Process priority levels from highest to lowest.
    priorities = sorted({task.priority for task in pending}, reverse=True)

    for priority in priorities:
        if not available:
            break

        group = [task for task in pending if task.priority == priority]
        if not group:
            continue

        # Large finite cost for impossible assignments.
        impossible_cost = 10**9

        costs = [
            [calculate_task_cost(robot, task, environment) for task in group]
            for robot in available
        ]

        # Add dummy columns so robots may remain unassigned.
        matrix = []
        for row in costs:
            matrix.append([
                impossible_cost if cost == inf else cost
                for cost in row
            ] + [impossible_cost // 2] * len(available))

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
