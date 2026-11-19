
import random
from collections import deque

import torch
import torch.nn.functional as F

from marl_model import MARLQNetwork


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)

    def add(self, features, reward, next_features, done):
        self.memory.append((
            list(features),
            float(reward),
            [list(x) for x in next_features],
            bool(done)
        ))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class MARLAgent:
    def __init__(
        self,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        batch_size=32,
        target_update=100
    ):
        self.device = torch.device("cpu")

        self.policy_net = MARLQNetwork().to(self.device)
        self.target_net = MARLQNetwork().to(self.device)

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate
        )

        self.memory = ReplayBuffer()

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.training_steps = 0

    def select_action(self, candidates, excluded_tasks=None):
        """
        candidates:
            [{"task_id": ..., "features": [...]}, ...]

        Returns the selected candidate dictionary
        or None if no legal candidate exists.
        """
        excluded_tasks = (
            set() if excluded_tasks is None
            else set(excluded_tasks)
        )

        legal = [
            candidate for candidate in candidates
            if candidate["task_id"] not in excluded_tasks
        ]

        if not legal:
            return None

        if random.random() < self.epsilon:
            return random.choice(legal)

        features = torch.tensor(
            [c["features"] for c in legal],
            dtype=torch.float32,
            device=self.device
        )

        with torch.no_grad():
            q_values = self.policy_net(features)

        index = int(torch.argmax(q_values).item())
        return legal[index]

    def remember(
        self, features, reward, next_features, done
    ):
        self.memory.add(
            features, reward, next_features, done
        )

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = self.memory.sample(self.batch_size)

        states = torch.tensor(
            [item[0] for item in batch],
            dtype=torch.float32,
            device=self.device
        )

        rewards = torch.tensor(
            [item[1] for item in batch],
            dtype=torch.float32,
            device=self.device
        )

        q_predictions = self.policy_net(states)

        targets = rewards.clone()

        with torch.no_grad():
            for i, (_, _, next_features, done) in enumerate(batch):
                if done or not next_features:
                    continue

                next_states = torch.tensor(
                    next_features,
                    dtype=torch.float32,
                    device=self.device
                )

                next_q = self.target_net(next_states)
                targets[i] += (
                    self.gamma * next_q.max()
                )

        loss = F.smooth_l1_loss(
            q_predictions,
            targets
        )

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            max_norm=1.0
        )

        self.optimizer.step()

        self.training_steps += 1

        if self.training_steps % self.target_update == 0:
            self.target_net.load_state_dict(
                self.policy_net.state_dict()
            )

        return float(loss.item())

    def decay_epsilon(self):
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )
