
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from marl_model import MARLQNetwork


def main():
    torch.manual_seed(42)

    model = MARLQNetwork(
        input_size=12,
        hidden_size=64
    )

    # Example batch with four candidate actions.
    features = torch.rand(4, 12)

    q_values = model(features)

    print("Input shape:", tuple(features.shape))
    print("Output shape:", tuple(q_values.shape))
    print("Q-values:", q_values.detach().tolist())

    assert q_values.shape == (4,)
    assert torch.isfinite(q_values).all()

    # Verify that the network can learn
    # through a single optimization step.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    targets = torch.tensor([
        1.0, 0.5, -0.5, 2.0
    ])

    loss_fn = torch.nn.MSELoss()

    optimizer.zero_grad()

    loss = loss_fn(q_values, targets)

    loss.backward()

    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
    )

    optimizer.step()

    print("Training loss:", round(loss.item(), 6))
    print("\nMARL MODEL TEST PASSED")


if __name__ == "__main__":
    main()
