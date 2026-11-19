
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import torch
from marl_agent import MARLAgent
from evaluate_trained_marl import evaluate


CHECKPOINTS = range(25, 201, 25)


def load_agent(episode):
    path = ROOT / "models" / f"marl_episode_{episode}.pt"

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    agent = MARLAgent(epsilon=0.0)

    agent.policy_net.load_state_dict(
        checkpoint["model"]
    )
    agent.policy_net.eval()
    agent.epsilon = 0.0

    return agent


def test_policy(policy, seeds, agent=None):
    rows = [
        evaluate(seed, policy, agent)
        for seed in seeds
    ]

    n = len(rows)

    return {
        "success": sum(
            int(r["success"]) for r in rows
        ),
        "steps": sum(
            r["steps"] for r in rows
        ) / n,
        "distance": sum(
            r["distance"] for r in rows
        ) / n,
        "reward": sum(
            r["reward"] for r in rows
        ) / n,
        "collisions": sum(
            r["collisions"] for r in rows
        )
    }


def show(label, result):
    print(
        f"{label:18} "
        f"success={result['success']}/30 "
        f"steps={result['steps']:.2f} "
        f"distance={result['distance']:.2f} "
        f"reward={result['reward']:.2f} "
        f"collisions={result['collisions']}"
    )


def main():
    torch.set_num_threads(2)

    validation_seeds = range(300, 330)
    test_seeds = range(400, 430)

    print("\nCHECKPOINT VALIDATION")
    print("=" * 75)

    greedy = test_policy(
        "greedy", validation_seeds
    )
    show("Greedy", greedy)

    results = []

    for episode in CHECKPOINTS:
        agent = load_agent(episode)

        result = test_policy(
            "dqn", validation_seeds, agent
        )

        show(f"Episode {episode}", result)

        results.append((episode, result))

    # Prefer successful, collision-free models.
    eligible = [
        (episode, result)
        for episode, result in results
        if result["success"] == 30
        and result["collisions"] == 0
    ]

    if not eligible:
        print("No eligible checkpoint found.")
        return

    # Select using distance, then steps.
    best_episode, best_result = min(
        eligible,
        key=lambda item: (
            item[1]["distance"],
            item[1]["steps"]
        )
    )

    print(f"\nBEST CHECKPOINT: {best_episode}")
    show("Best validation", best_result)

    # Held-out test, evaluated only after selection.
    print("\nFINAL HELD-OUT TEST")
    print("=" * 75)

    best_agent = load_agent(best_episode)

    greedy_test = test_policy(
        "greedy", test_seeds
    )

    dqn_test = test_policy(
        "dqn", test_seeds, best_agent
    )

    show("Greedy", greedy_test)
    show("Selected DQN", dqn_test)

    improvement = (
        greedy_test["distance"]
        - dqn_test["distance"]
    ) / greedy_test["distance"] * 100

    print(
        f"\nDistance improvement vs Greedy: "
        f"{improvement:+.2f}%"
    )

    print("\nCHECKPOINT SELECTION COMPLETE")


if __name__ == "__main__":
    main()
