
import sys
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import torch

from marl_agent import MARLAgent
from test_marl_episodes import create_episode, baseline_actions
from test_marl_integration import select_team_actions


CHECKPOINT = ROOT / "models" / "marl_episode_200.pt"


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


def load_trained_agent():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    agent = MARLAgent(epsilon=0.0)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True
    )

    agent.policy_net.load_state_dict(
        checkpoint["model"]
    )

    agent.target_net.load_state_dict(
        checkpoint["target"]
    )

    agent.policy_net.eval()
    agent.target_net.eval()
    agent.epsilon = 0.0

    return agent


def print_summary(name, rows):
    n = len(rows)

    success = sum(row["success"] for row in rows)

    avg_steps = sum(
        row["steps"] for row in rows
    ) / n

    avg_distance = sum(
        row["distance"] for row in rows
    ) / n

    avg_reward = sum(
        row["reward"] for row in rows
    ) / n

    collisions = sum(
        row["collisions"] for row in rows
    )

    print(f"\n{name}")
    print(f"Success: {success}/{n}")
    print(f"Average steps: {avg_steps:.3f}")
    print(f"Average distance: {avg_distance:.3f}")
    print(f"Average reward: {avg_reward:.3f}")
    print(f"Total collisions: {collisions}")

    return avg_steps, avg_distance


def main():
    random.seed(42)
    torch.manual_seed(42)

    # Reproduce the initial untrained model.
    untrained = MARLAgent(epsilon=0.0)
    untrained.policy_net.eval()

    trained = load_trained_agent()

    results = {
        "GREEDY": [],
        "UNTRAINED DQN": [],
        "TRAINED DQN": []
    }

    for seed in range(200, 230):
        results["GREEDY"].append(
            evaluate(seed, "greedy")
        )

        results["UNTRAINED DQN"].append(
            evaluate(seed, "dqn", untrained)
        )

        results["TRAINED DQN"].append(
            evaluate(seed, "dqn", trained)
        )

    print("\nFINAL MARL EVALUATION")
    print("=" * 55)

    metrics = {}

    for name, rows in results.items():
        metrics[name] = print_summary(name, rows)

    greedy_steps, greedy_dist = metrics["GREEDY"]
    trained_steps, trained_dist = metrics["TRAINED DQN"]

    steps_improvement = (
        (greedy_steps - trained_steps)
        / greedy_steps * 100
    )

    distance_improvement = (
        (greedy_dist - trained_dist)
        / greedy_dist * 100
    )

    print("\nTRAINED DQN VS GREEDY")
    print(
        f"Steps improvement: "
        f"{steps_improvement:+.2f}%"
    )
    print(
        f"Distance improvement: "
        f"{distance_improvement:+.2f}%"
    )

    print("\nEVALUATION COMPLETE")


if __name__ == "__main__":
    main()
