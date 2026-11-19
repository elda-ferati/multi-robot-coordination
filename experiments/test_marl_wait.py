
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import torch

from test_marl_episodes import create_episode
from marl_environment_v2 import MARLEnvironmentV2
from marl_wait_policy import (
    MARLQNetworkV2Wait,
    select_actions_with_wait
)


def set_final_score(layer, value):
    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.fill_(value)


def main():
    torch.manual_seed(42)

    original = create_episode(seed=42)

    env = MARLEnvironmentV2(
        original.sim,
        max_steps=original.max_steps,
        max_candidates=original.max_candidates
    )

    model = MARLQNetworkV2Wait()
    observations = env.get_observations()

    # Test 1: WAIT has higher Q-value.
    set_final_score(model.network[-1], -1.0)
    set_final_score(model.wait_network[-1], 1.0)

    wait_actions = select_actions_with_wait(
        model, observations
    )

    assert all(
        action is None
        for action in wait_actions.values()
    )

    print("WAIT decisions:", wait_actions)

    # Test 2: Assigning tasks has higher Q-value.
    set_final_score(model.network[-1], 1.0)
    set_final_score(model.wait_network[-1], -1.0)

    task_actions = select_actions_with_wait(
        model, observations
    )

    assigned = [
        task_id for task_id in task_actions.values()
        if task_id is not None
    ]

    assert len(assigned) == len(set(assigned))
    assert len(assigned) == 3

    print("Task decisions:", task_actions)

    # Test 3: WAIT branch supports gradients.
    candidates = observations[1]

    features = torch.tensor(
        [c["features"] for c in candidates],
        dtype=torch.float32
    )

    model.train()
    model.zero_grad()

    wait_value = model.score_wait(features)
    wait_value.backward()

    assert any(
        parameter.grad is not None
        for parameter in model.wait_network.parameters()
    )

    print("WAIT output:", round(wait_value.item(), 4))
    print("\nMARL WAIT TEST PASSED")


if __name__ == "__main__":
    main()
