
from functools import lru_cache
from pathlib import Path

import torch

from marl_model_v2 import MARLQNetworkV2
from marl_features_v2 import get_team_features
from marl_task_policy_v4 import select_task_actions


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT / "models_v4" / "marl_v4_episode_100.pt"
)

MAX_CANDIDATES = 8


@lru_cache(maxsize=1)
def load_marl_v4():
    """
    Load the trained model only once.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MARL checkpoint not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    model = MARLQNetworkV2()
    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model


def allocate_tasks_marl_v4(
    robots,
    tasks,
    environment
):
    """
    Allocate pending tasks to idle robots
    using trained MARL V4 Q-values.
    """

    model = load_marl_v4()

    pending = [
        task for task in tasks
        if task.status == "pending"
    ]

    idle = [
        robot for robot in robots
        if robot.current_task is None
        and robot.goal is None
    ]

    if not pending or not idle:
        return

    # Match the V2 observation candidate ordering.
    pending.sort(
        key=lambda task: (
            -task.priority,
            task.task_id
        )
    )

    observations = {}

    for robot in sorted(
        idle,
        key=lambda r: r.robot_id
    ):
        candidates = []

        for task in pending[:MAX_CANDIDATES]:
            features = get_team_features(
                robot,
                task,
                robots,
                tasks,
                environment
            )

            candidates.append({
                "task_id": task.task_id,
                "features": features
            })

        observations[robot.robot_id] = candidates

    # Model selects a joint allocation.
    actions = select_task_actions(
        model,
        observations
    )

    task_lookup = {
        task.task_id: task
        for task in tasks
    }

    robot_lookup = {
        robot.robot_id: robot
        for robot in robots
    }

    assigned = 0

    for robot_id, task_id in actions.items():
        if task_id is None:
            continue

        robot = robot_lookup[robot_id]
        task = task_lookup[task_id]

        # Confirm the task is still available.
        if task.status != "pending":
            continue

        success = robot.assign_task(
            task,
            environment
        )

        if success:
            assigned += 1

    return assigned
