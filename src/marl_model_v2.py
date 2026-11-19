
import torch
import torch.nn as nn


class MARLQNetworkV2(nn.Module):
    """
    Shared Q-network for multi-robot task allocation.

    Input: 17 team-aware features.
    Output: one Q-value per candidate.
    """

    def __init__(self, input_size=17, hidden_size=128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, 1)
        )

    def forward(self, features):
        if features.shape[-1] != 17:
            raise ValueError(
                "MARL V2 expects 17 input features"
            )

        return self.network(features).squeeze(-1)
