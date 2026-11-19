
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import torch

from test_marl_episodes import create_episode
from marl_environment_v2 import MARLEnvironmentV2
from marl_model_v2 import MARLQNetworkV2
from marl_task_policy_v4 import select_task_actions


def main():
    torch.manual_seed(42)

    original = create_episode(seed=42)

    env = MARLEnvironmentV2(
        original.sim,
        max_steps=original.max_steps,
        max_candidates=original.max_candidates
    )

    model = MARLQNetworkV2()

    observations = env.get_observations()

    actions = select_task_actions(
        model, observations
    )

    assigned = [
        task_id
        for task_id in actions.values()
        if task_id is not None
    ]

    print("\nMARL V4 TASK POLICY TEST")
    print("=" * 50)
    print("Robot actions:", actions)
    print("Assigned tasks:", len(assigned))
    print("Unique tasks:", len(set(assigned)))

    assert len(assigned) == len(set(assigned))
    assert len(assigned) == 3

    # Test a complete episode.
    for _ in range(env.max_steps):
        obs = env.get_observations()

        actions = select_task_actions(model, obs)

        _, _, done, info = env.step(actions)

        if done:
            break

    print("Completed:", info["completed"])
    print("Total tasks:", info["total_tasks"])
    print("Steps:", info["time_step"])
    print("Distance:", info["distance"])
    print("Collisions:", info["collisions"])

    assert info["success"]
    assert info["collisions"] == 0

    print("\nMARL V4 TASK POLICY TEST PASSED")


if __name__ == "__main__":
    main()
