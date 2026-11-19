
import torch


def select_joint_actions(agent, observations):
    """
    Centralized greedy allocation using Q-values.

    Each robot receives at most one task.
    Each task is assigned to at most one robot.
    """

    actions = {
        robot_id: None
        for robot_id in observations
    }

    pairs = []

    for robot_id, candidates in observations.items():
        if not candidates:
            continue

        features = torch.tensor(
            [c["features"] for c in candidates],
            dtype=torch.float32,
            device=agent.device
        )

        with torch.no_grad():
            scores = agent.policy_net(features)

        for candidate, score in zip(candidates, scores):
            pairs.append((
                float(score.item()),
                robot_id,
                candidate["task_id"]
            ))

    pairs.sort(
        key=lambda item: (-item[0], item[1], item[2])
    )

    assigned_robots = set()
    assigned_tasks = set()

    for score, robot_id, task_id in pairs:
        if robot_id in assigned_robots:
            continue

        if task_id in assigned_tasks:
            continue

        actions[robot_id] = task_id

        assigned_robots.add(robot_id)
        assigned_tasks.add(task_id)

    return actions
