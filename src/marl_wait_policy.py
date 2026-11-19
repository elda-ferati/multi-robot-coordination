
import torch
import torch.nn as nn

from marl_model_v2 import MARLQNetworkV2


class MARLQNetworkV2Wait(MARLQNetworkV2):
    """
    V2 model with task Q-values and
    a separate learnable WAIT Q-value.
    """

    def __init__(self):
        super().__init__()

        self.wait_network = nn.Sequential(
            nn.Linear(17, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def score_wait(self, candidate_features):
        """
        Uses the mean candidate representation
        to estimate whether waiting is useful.
        """
        if candidate_features.ndim != 2:
            raise ValueError("Expected a 2D tensor")

        if candidate_features.shape[1] != 17:
            raise ValueError("Expected 17 features")

        if candidate_features.shape[0] == 0:
            raise ValueError("No candidate features")

        summary = candidate_features.mean(
            dim=0, keepdim=True
        )

        return self.wait_network(summary).squeeze()


def select_actions_with_wait(model, observations):
    """
    Joint allocation using learned Q-values.

    Each robot can receive a task or WAIT.
    Tasks cannot be assigned twice.
    """
    actions = {
        robot_id: None
        for robot_id in observations
    }

    pairs = []
    wait_scores = {}

    model.eval()

    with torch.no_grad():
        for robot_id, candidates in observations.items():
            if not candidates:
                continue

            features = torch.tensor(
                [c["features"] for c in candidates],
                dtype=torch.float32
            )

            task_scores = model(features)
            wait_scores[robot_id] = float(
                model.score_wait(features).item()
            )

            for candidate, score in zip(
                candidates, task_scores
            ):
                pairs.append((
                    float(score.item()),
                    robot_id,
                    candidate["task_id"]
                ))

    pairs.sort(
        key=lambda x: (-x[0], x[1], x[2])
    )

    assigned_robots = set()
    assigned_tasks = set()

    for score, robot_id, task_id in pairs:
        if robot_id in assigned_robots:
            continue

        if task_id in assigned_tasks:
            continue

        # WAIT is preferable for this robot.
        if score <= wait_scores[robot_id]:
            continue

        actions[robot_id] = task_id
        assigned_robots.add(robot_id)
        assigned_tasks.add(task_id)

    return actions
