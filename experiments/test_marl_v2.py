
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import torch

from test_marl_episodes import create_episode
from marl_environment_v2 import MARLEnvironmentV2
from marl_model_v2 import MARLQNetworkV2


def main():
    torch.manual_seed(42)

    old_env = create_episode(seed=42)

    env = MARLEnvironmentV2(
        old_env.sim,
        max_steps=old_env.max_steps,
        max_candidates=old_env.max_candidates
    )

    model = MARLQNetworkV2()

    observations = env.get_observations()

    all_features = [
        candidate["features"]
        for candidates in observations.values()
        for candidate in candidates
    ]

    assert len(all_features) > 0
    assert all(
        len(features) == 17
        for features in all_features
    )

    inputs = torch.tensor(
        all_features,
        dtype=torch.float32
    )

    q_values = model(inputs)

    assert q_values.shape == (
        len(all_features),
    )

    assert torch.isfinite(q_values).all()

    # Check that gradients can update the network.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    targets = torch.ones_like(q_values)

    before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    optimizer.zero_grad()

    loss = torch.nn.functional.mse_loss(
        q_values,
        targets
    )

    loss.backward()
    optimizer.step()

    changed = any(
        not torch.equal(old, new)
        for old, new in zip(
            before,
            model.parameters()
        )
    )

    assert changed

    print("Number of candidates:", len(all_features))
    print("Input shape:", tuple(inputs.shape))
    print("Output shape:", tuple(q_values.shape))
    print("Loss:", round(loss.item(), 6))

    print("\nMARL V2 MODEL TEST PASSED")


if __name__ == "__main__":
    main()
