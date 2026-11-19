
from math import inf
from pathfinding import astar


def allocate_tasks_auction(robots, tasks, environment):
    """
    Priority-aware, centralized auction allocation.

    Each available robot bids its estimated A* travel
    distance for each pending task.

    The lowest valid bid wins each round.
    Assigned robots and tasks leave the auction.
    Higher-priority tasks are processed first.
    """

    available = [
        robot for robot in robots
        if robot.current_task is None
        and robot.goal is None
    ]

    pending = [
        task for task in tasks
        if task.status == "pending"
    ]

    if not available or not pending:
        return

    # Cache delivery distances for all tasks.
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

    # Process urgent tasks before normal tasks.
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

        bids = []

        # Every robot submits a bid for each task.
        for robot in available:
            for task in group:
                delivery_cost = delivery_costs[task.task_id]

                if delivery_cost == inf:
                    continue

                pickup_path = astar(
                    environment,
                    robot.position,
                    task.pickup
                )

                if pickup_path is None:
                    continue

                bid = (
                    len(pickup_path) - 1
                    + delivery_cost
                )

                bids.append((
                    bid,
                    robot.robot_id,
                    task.task_id,
                    robot,
                    task
                ))

        # Lowest-cost bids win first.
        bids.sort(key=lambda item: item[:3])

        assigned_robots = set()
        assigned_tasks = set()

        for bid, robot_id, task_id, robot, task in bids:
            if robot_id in assigned_robots:
                continue

            if task_id in assigned_tasks:
                continue

            if robot.assign_task(task, environment):
                assigned_robots.add(robot_id)
                assigned_tasks.add(task_id)

        # Remove allocated robots before next priority.
        available = [
            robot for robot in available
            if robot.robot_id not in assigned_robots
        ]
