
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import torch

from test_marl_episodes import (
    create_episode,
    baseline_actions
)
from test_joint_marl import load_agent as load_v1
from marl_team_policy import select_joint_actions
from marl_environment_v2 import MARLEnvironmentV2
from marl_wait_policy import (
    MARLQNetworkV2Wait,
    select_actions_with_wait
)


def load_v2():
    path = (
        ROOT / "models_v2" /
        "marl_v2_episode_200.pt"
    )

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    model = MARLQNetworkV2Wait()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def evaluate(seed, policy, v1, v2):
    original = create_episode(seed)

    if policy == "v2":
        env = MARLEnvironmentV2(
            original.sim,
            max_steps=original.max_steps,
            max_candidates=original.max_candidates
        )
    else:
        env = original

    wait_count = 0
    decisions = 0

    for _ in range(env.max_steps):
        obs = env.get_observations()

        if policy == "greedy":
            actions = baseline_actions(env)
        elif policy == "v1":
            actions = select_joint_actions(v1, obs)
        else:
            actions = select_actions_with_wait(v2, obs)

        # Count WAIT only when a robot is idle
        # and has candidate tasks available.
        if policy == "v2":
            for robot_id, candidates in obs.items():
                if candidates:
                    decisions += 1

                    if actions[robot_id] is None:
                        wait_count += 1

        _, _, done, info = env.step(actions)

        if done:
            break

    return info, wait_count, decisions


def main():
    torch.manual_seed(42)
    torch.set_num_threads(2)

    v1 = load_v1()
    v2 = load_v2()

    print("\nMARL V2 EVALUATION")
    print("=" * 65)

    for policy in ["greedy", "v1", "v2"]:
        rows = [
            evaluate(seed, policy, v1, v2)
            for seed in range(600, 630)
        ]

        info_rows = [row[0] for row in rows]

        success = sum(
            int(info["success"])
            for info in info_rows
        )

        steps = sum(
            info["time_step"]
            for info in info_rows
        ) / len(info_rows)

        distance = sum(
            info["distance"]
            for info in info_rows
        ) / len(info_rows)

        collisions = sum(
            info["collisions"]
            for info in info_rows
        )

        print(f"\n{policy.upper()}")
        print(f"Success: {success}/30")
        print(f"Average steps: {steps:.3f}")
        print(f"Average distance: {distance:.3f}")
        print(f"Collisions: {collisions}")

        if policy == "v2":
            waits = sum(row[1] for row in rows)
            decisions = sum(row[2] for row in rows)

            wait_rate = (
                100 * waits / decisions
                if decisions else 0.0
            )

            print(f"WAIT decisions: {waits}")
            print(f"Decision opportunities: {decisions}")
            print(f"WAIT rate: {wait_rate:.2f}%")

    print("\nMARL V2 EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
