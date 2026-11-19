
from marl_features import get_robot_task_features


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_team_features(
    robot, task, robots, tasks, environment
):
    base = get_robot_task_features(
        robot, task, robots, tasks, environment
    )

    rows = environment.rows
    cols = environment.cols
    max_distance = max(1, rows + cols - 2)

    other_robots = [
        r for r in robots
        if r.robot_id != robot.robot_id
    ]

    available_others = [
        r for r in other_robots
        if r.current_task is None and r.goal is None
    ]

    # 13: Proximity to another robot.
    if other_robots:
        nearest_robot = min(
            manhattan(robot.position, r.position)
            for r in other_robots
        ) / max_distance
    else:
        nearest_robot = 1.0

    # 14: Best competing robot's pickup distance.
    if available_others:
        competitor_distance = min(
            manhattan(r.position, task.pickup)
            for r in available_others
        ) / max_distance
    else:
        competitor_distance = 1.0

    # 15: Relative advantage for this task.
    own_distance = (
        manhattan(robot.position, task.pickup)
        / max_distance
    )

    advantage = (
        competitor_distance - own_distance + 1.0
    ) / 2.0

    # 16: Fraction of pending urgent tasks.
    pending = [
        t for t in tasks
        if t.status == "pending"
    ]

    urgent_count = sum(
        t.priority >= 3 for t in pending
    )

    urgent_fraction = (
        urgent_count / len(pending)
        if pending else 0.0
    )

    # 17: Fraction of robots currently idle.
    idle_count = sum(
        r.current_task is None and r.goal is None
        for r in robots
    )

    idle_fraction = (
        idle_count / len(robots)
        if robots else 0.0
    )

    features = base + [
        nearest_robot,
        competitor_distance,
        advantage,
        urgent_fraction,
        idle_fraction
    ]

    assert len(features) == 17

    return features
