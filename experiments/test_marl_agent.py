
import sys
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from marl_agent import MARLAgent


def main():
    random.seed(42)
    torch.manual_seed(42)

    agent = MARLAgent(
        batch_size=8,
        epsilon=0.0,
        target_update=2
    )

    candidates = [
        {
            "task_id": 1,
            "features": [0.2] * 12
        },
        {
            "task_id": 2,
            "features": [0.8] * 12
        }
    ]

    selected = agent.select_action(candidates)

    assert selected is not None
    assert selected["task_id"] in (1, 2)

    # Verify that reserved tasks are excluded.
    alternative = agent.select_action(
        candidates,
        excluded_tasks={selected["task_id"]}
    )

    assert alternative is not None
    assert alternative["task_id"] != selected["task_id"]

    print("Selected task:", selected["task_id"])
    print("Alternative task:", alternative["task_id"])

    # Store synthetic transitions for a unit test.
    # These are NOT actual simulation experiences.
    for i in range(40):
        value = (i % 10) / 10

        features = [value] * 12

        next_features = [
            [min(1.0, value + 0.1)] * 12
        ]

        agent.remember(
            features=features,
            reward=1.0 if i % 2 == 0 else -0.5,
            next_features=next_features,
            done=(i % 5 == 0)
        )

    assert len(agent.memory) == 40

    before = [
        p.detach().clone()
        for p in agent.policy_net.parameters()
    ]

    loss1 = agent.train_step()
    loss2 = agent.train_step()

    assert loss1 is not None
    assert loss2 is not None

    changed = any(
        not torch.equal(old, new)
        for old, new in zip(
            before,
            agent.policy_net.parameters()
        )
    )

    assert changed, "Network parameters did not change"

    agent.epsilon = 1.0
    agent.decay_epsilon()

    print("Replay buffer size:", len(agent.memory))
    print("Training steps:", agent.training_steps)
    print("Loss 1:", round(loss1, 6))
    print("Loss 2:", round(loss2, 6))
    print("Epsilon:", agent.epsilon)

    print("\nMARL AGENT TEST PASSED")


if __name__ == "__main__":
    main()
