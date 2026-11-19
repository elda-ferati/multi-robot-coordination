
from math import inf

from hungarian_v3 import allocate_tasks_hungarian_v3
from pathfinding import astar


MIN_TIME_GAIN = 3
MAX_PREEMPTIONS_PER_TASK = 1


def distance(environment, start, goal):
    path = astar(environment, start, goal)
    return len(path) - 1 if path is not None else inf


def allocate_tasks_hungarian_v4(robots, tasks, environment):
    # 1. Use V3 for all currently available robots.
    allocate_tasks_hungarian_v3(
        robots, tasks, environment
    )

    # 2. Urgent tasks still waiting.
    urgent_tasks = sorted(
        [
            task for task in tasks
            if task.status == "pending"
            and task.priority >= 3
        ],
        key=lambda task: (-task.priority, task.task_id)
    )

    if not urgent_tasks:
        return

    # 3. Only robots heading toward a lower-priority pickup
    # can be interrupted.
    candidates = [
        robot for robot in robots
        if robot.current_task is not None
        and not robot.carrying_package
        and robot.current_task.status == "assigned"
        and getattr(
            robot.current_task, "preemption_count", 0
        ) < MAX_PREEMPTIONS_PER_TASK
    ]

    if not candidates:
        return

    # 4. Find feasible robot-urgent pairs.
    # The baseline is the earliest estimated availability
    # of a robot already working on a task.
    busy_remaining = []

    for robot in robots:
        if robot.current_task is None:
            continue

        task = robot.current_task

        if robot.carrying_package:
            remaining = distance(
                environment, robot.position, task.delivery
            )
        else:
            to_pickup = distance(
                environment, robot.position, task.pickup
            )
            to_delivery = distance(
                environment, task.pickup, task.delivery
            )
            remaining = to_pickup + to_delivery

        if remaining != inf:
            busy_remaining.append(remaining)

    if not busy_remaining:
        return

    earliest_free = min(busy_remaining)

    pairs = []

    for urgent in urgent_tasks:
        for robot in candidates:
            old_task = robot.current_task

            if old_task.priority >= urgent.priority:
                continue

            urgent_pickup = distance(
                environment, robot.position, urgent.pickup
            )

            if urgent_pickup == inf:
                continue

            urgent_delivery = distance(
                environment, urgent.pickup, urgent.delivery
            )

            if urgent_delivery == inf:
                continue

            # Estimate when the urgent pickup could be served
            # without preemption, using currently busy robots.
            estimated_wait = min(
                (
                    distance(
                        environment, other.position,
                        other.current_task.delivery
                    )
                    if other.carrying_package
                    else (
                        distance(
                            environment, other.position,
                            other.current_task.pickup
                        )
                        + distance(
                            environment,
                            other.current_task.pickup,
                            other.current_task.delivery
                        )
                    )
                ) + distance(
                    environment,
                    other.current_task.delivery,
                    urgent.pickup
                )
                for other in robots
                if other.current_task is not None
            )

            gain = estimated_wait - urgent_pickup

            if gain >= MIN_TIME_GAIN:
                pairs.append((
                    -gain,
                    urgent_pickup,
                    robot.robot_id,
                    urgent.task_id,
                    robot,
                    urgent
                ))

    # 5. Highest estimated gain first.
    pairs.sort(key=lambda item: item[:4])

    used_robots = set()
    used_tasks = set()

    for _, _, robot_id, task_id, robot, urgent in pairs:
        if robot_id in used_robots or task_id in used_tasks:
            continue

        if urgent.status != "pending":
            continue

        old_task = robot.current_task

        if (
            old_task is None
            or robot.carrying_package
            or old_task.status != "assigned"
            or old_task.priority >= urgent.priority
        ):
            continue

        # Check path before changing any state.
        new_path = astar(
            environment, robot.position, urgent.pickup
        )

        if new_path is None:
            continue

        # 6. Release the old normal task.
        old_task.status = "pending"
        old_task.assigned_robot = None
        old_task.preemption_count = (
            getattr(old_task, "preemption_count", 0) + 1
        )

        # 7. Assign the urgent task.
        robot.current_task = urgent
        robot.carrying_package = False
        robot.goal = urgent.pickup
        robot.path = new_path[1:]

        urgent.status = "assigned"
        urgent.assigned_robot = robot.robot_id

        used_robots.add(robot_id)
        used_tasks.add(task_id)
