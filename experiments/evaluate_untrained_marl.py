
import sys
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import torch
from marl_agent import MARLAgent
from test_marl_episodes import (
    create_episode,
    baseline_actions
)
from test_marl_integration import select_team_actions


def evaluate(seed, policy, agent=None):
    env = create_episode(seed)
    total_reward = 0.0

    for _ in range(env.max_steps):
        if policy == "greedy":
            actions = baseline_actions(env)
        else:
            actions = select_team_actions(
                agent,
                env.get_observations()
            )

        _, reward, done, info = env.step(actions)
        total_reward += reward

        if done:
            break

    return {
        "success": info["success"],
        "steps": info["time_step"],
        "distance": info["distance"],
        "reward": total_reward,
        "collisions": info["collisions"]
    }


def main():
    random.seed(42)
    torch.manual_seed(42)

    agent = MARLAgent(epsilon=0.0)
    agent.policy_net.eval()

    results = {
        "greedy": [],
        "untrained": []
    }

    for seed in range(200, 230):
        for policy in results:
            result = evaluate(
                seed,
                policy,
                agent
            )

            results[policy].append(result)

            assert result["collisions"] == 0

    print("\nBASELINE EVALUATION")
    print("=" * 50)

    for policy, rows in results.items():
        n = len(rows)

        success = sum(
            row["success"] for row in rows
        )

        avg_steps = sum(
            row["steps"] for row in rows
        ) / n

        avg_distance = sum(
            row["distance"] for row in rows
        ) / n

        avg_reward = sum(
            row["reward"] for row in rows
        ) / n

        print(f"\n{policy.upper()}")
        print(f"Success: {success}/{n}")
        print(f"Average steps: {avg_steps:.3f}")
        print(f"Average distance: {avg_distance:.3f}")
        print(f"Average reward: {avg_reward:.3f}")

    print("\nBASELINE EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
