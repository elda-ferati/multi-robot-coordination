
import sys
import random
from pathlib import Path
from collections import deque

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F

from marl_agent import MARLAgent
from test_marl_episodes import create_episode


EPISODES = 200
BATCH_SIZE = 32
GAMMA = 0.99
SAVE_EVERY = 25

MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


def choose_actions(agent, observations):
    actions = {}
    chosen = {}
    reserved = set()

    for robot_id in sorted(observations):
        selected = agent.select_action(
            observations[robot_id],
            excluded_tasks=reserved
        )

        if selected is None:
            actions[robot_id] = None
            continue

        task_id = selected["task_id"]

        actions[robot_id] = task_id
        chosen[robot_id] = selected
        reserved.add(task_id)

    return actions, chosen


def optimize(agent, replay):
    if len(replay) < BATCH_SIZE:
        return None

    batch = random.sample(replay, BATCH_SIZE)

    states = torch.tensor(
        [x[0] for x in batch],
        dtype=torch.float32
    )

    rewards = torch.tensor(
        [x[1] for x in batch],
        dtype=torch.float32
    )

    durations = torch.tensor(
        [x[3] for x in batch],
        dtype=torch.float32
    )

    dones = torch.tensor(
        [x[4] for x in batch],
        dtype=torch.bool
    )

    predictions = agent.policy_net(states)
    targets = rewards.clone()

    with torch.no_grad():
        for i, item in enumerate(batch):
            next_candidates = item[2]

            if dones[i].item() or not next_candidates:
                continue

            next_states = torch.tensor(
                next_candidates,
                dtype=torch.float32
            )

            # Double DQN: online network chooses,
            # target network evaluates.
            online_values = agent.policy_net(next_states)
            best_index = int(online_values.argmax())

            target_value = agent.target_net(
                next_states
            )[best_index]

            targets[i] += (
                GAMMA ** durations[i]
            ) * target_value

    loss = F.smooth_l1_loss(
        predictions, targets
    )

    agent.optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        agent.policy_net.parameters(),
        1.0
    )

    agent.optimizer.step()
    agent.training_steps += 1

    if agent.training_steps % agent.target_update == 0:
        agent.target_net.load_state_dict(
            agent.policy_net.state_dict()
        )

    return float(loss.item())


def run_episode(agent, seed, replay):
    env = create_episode(seed)
    active_decisions = {}

    total_reward = 0.0
    losses = []

    for _ in range(env.max_steps):
        observations = env.get_observations()

        actions, chosen = choose_actions(
            agent, observations
        )

        # Track the start of each assignment.
        for robot_id, candidate in chosen.items():
            robot = next(
                r for r in env.sim.robots
                if r.robot_id == robot_id
            )

            active_decisions[robot_id] = {
                "task_id": candidate["task_id"],
                "features": candidate["features"],
                "start_step": env.sim.time_step,
                "start_distance":
                    env.sim.robot_distance[robot_id],
            }

        next_obs, step_reward, done, info = (
            env.step(actions)
        )

        total_reward += step_reward

        tasks_by_id = {
            task.task_id: task
            for task in env.sim.tasks
        }

        # Record experience only after the
        # assigned task has been completed.
        for robot_id, decision in list(
            active_decisions.items()
        ):
            task = tasks_by_id[decision["task_id"]]

            if task.status != "completed":
                continue

            duration = max(
                1,
                env.sim.time_step - decision["start_step"]
            )

            traveled = (
                env.sim.robot_distance[robot_id]
                - decision["start_distance"]
            )

            reward = (
                (20.0 if task.priority >= 3 else 10.0)
                - 0.05 * duration
                - 0.02 * traveled
            )

            next_features = [
                candidate["features"]
                for candidate in next_obs[robot_id]
            ]

            replay.append((
                decision["features"],
                reward,
                next_features,
                duration,
                done
            ))

            del active_decisions[robot_id]

            loss = optimize(agent, replay)

            if loss is not None:
                losses.append(loss)

        if done:
            break

    # Penalize assignments not completed
    # before the episode time limit.
    for decision in active_decisions.values():
        replay.append((
            decision["features"],
            -10.0,
            [],
            max(
                1,
                env.sim.time_step
                - decision["start_step"]
            ),
            True
        ))

    return info, total_reward, losses


def save_checkpoint(agent, episode):
    path = MODEL_DIR / f"marl_episode_{episode}.pt"

    torch.save({
        "episode": episode,
        "model": agent.policy_net.state_dict(),
        "target": agent.target_net.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "epsilon": agent.epsilon,
        "training_steps": agent.training_steps,
    }, path)

    print("Checkpoint saved:", path.name)


def main():
    random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(2)

    agent = MARLAgent(
        batch_size=BATCH_SIZE,
        epsilon=1.0,
        epsilon_decay=0.98,
        epsilon_min=0.05,
        target_update=100
    )

    replay = deque(maxlen=10000)

    recent_steps = []
    recent_distance = []
    recent_success = []

    print("Starting MARL training...")

    for episode in range(1, EPISODES + 1):
        info, reward, losses = run_episode(
            agent,
            seed=episode + 1000,
            replay=replay
        )

        agent.decay_epsilon()

        recent_steps.append(info["time_step"])
        recent_distance.append(info["distance"])
        recent_success.append(int(info["success"]))

        if episode % 20 == 0:
            avg_steps = sum(recent_steps) / len(recent_steps)
            avg_distance = (
                sum(recent_distance) / len(recent_distance)
            )

            print(
                f"Episode {episode}/{EPISODES} "
                f"success={sum(recent_success)}/"
                f"{len(recent_success)} "
                f"steps={avg_steps:.2f} "
                f"distance={avg_distance:.2f} "
                f"epsilon={agent.epsilon:.3f} "
                f"buffer={len(replay)} "
                f"updates={agent.training_steps}"
            )

            recent_steps.clear()
            recent_distance.clear()
            recent_success.clear()

        if episode % SAVE_EVERY == 0:
            save_checkpoint(agent, episode)

    print("\nTRAINING COMPLETE")
    print("Total updates:", agent.training_steps)
    print("Final epsilon:", round(agent.epsilon, 4))


if __name__ == "__main__":
    main()
