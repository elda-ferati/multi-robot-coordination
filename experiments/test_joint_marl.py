
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import torch

from marl_agent import MARLAgent
from marl_team_policy import select_joint_actions
from test_marl_episodes import create_episode, baseline_actions
from test_marl_integration import select_team_actions


CHECKPOINT = ROOT / "models" / "marl_episode_75.pt"


def load_agent():
    agent = MARLAgent(epsilon=0.0)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True
    )

    agent.policy_net.load_state_dict(
        checkpoint["model"]
    )

    agent.policy_net.eval()
    agent.epsilon = 0.0

    return agent


def evaluate(seed, policy, agent):
    env = create_episode(seed)

    for _ in range(env.max_steps):
        observations = env.get_observations()

        if policy == "greedy":
            actions = baseline_actions(env)

        elif policy == "sequential":
            actions = select_team_actions(
                agent, observations
            )

        else:
            actions = select_joint_actions(
                agent, observations
            )

        _, _, done, info = env.step(actions)

        if done:
            break

    return info


def main():
    torch.manual_seed(42)
    torch.set_num_threads(2)

    agent = load_agent()

    policies = ["greedy", "sequential", "joint"]

    print("\nJOINT MARL VALIDATION")
    print("=" * 65)

    for policy in policies:
        results = [
            evaluate(seed, policy, agent)
            for seed in range(300, 330)
        ]

        successes = sum(
            int(r["success"]) for r in results
        )

        avg_steps = sum(
            r["time_step"] for r in results
        ) / len(results)

        avg_distance = sum(
            r["distance"] for r in results
        ) / len(results)

        collisions = sum(
            r["collisions"] for r in results
        )

        print(
            f"{policy:12} "
            f"success={successes}/30 "
            f"steps={avg_steps:.3f} "
            f"distance={avg_distance:.3f} "
            f"collisions={collisions}"
        )

        assert collisions == 0

    print("\nJOINT MARL TEST COMPLETE")


if __name__ == "__main__":
    main()
