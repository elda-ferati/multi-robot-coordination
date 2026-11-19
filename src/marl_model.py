
import torch
import torch.nn as nn


class MARLQNetwork(nn.Module):
    """
    Shared Q-network for robot-task allocation.

    Input: 12 features per robot-task pair.
    Output: one Q-value per candidate action.
    """

    def __init__(self, input_size=12, hidden_size=64):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, features):
        return self.network(features).squeeze(-1)
