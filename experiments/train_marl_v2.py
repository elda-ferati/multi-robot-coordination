
import sys
import random
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import torch
import torch.nn.functional as F

from test_marl_episodes import create_episode
from marl_environment_v2 import MARLEnvironmentV2
from marl_wait_policy import (
    MARLQNetworkV2Wait,
    select_actions_with_wait
)


EPISODES = 200
GAMMA = 0.99
BATCH_SIZE = 32

MODEL_DIR = ROOT / "models_v2"
MODEL_DIR.mkdir(exist_ok=True)

torch.set_num_threads(2)


def create_v2_episode(seed):
    original = create_episode(seed)

    return MARLEnvironmentV2(
        original.sim,
        max_steps=original.max_steps,
        max_candidates=original.max_candidates
    )


def choose_actions(model, observations, epsilon):
    if random.random() >= epsilon:
        return select_actions_with_wait(
            model, observations
        )

    # Exploration: random legal tasks or WAIT.
    actions = {}
    reserved = set()

    for robot_id in sorted(observations):
        available = [
            c["task_id"]
            for c in observations[robot_id]
            if c["task_id"] not in reserved
        ]

        if not available or random.random() < 0.2:
            actions[robot_id] = None
        else:
            task_id = random.choice(available)
            actions[robot_id] = task_id
            reserved.add(task_id)

    return actions


def best_next_q(target, candidates):
    if not candidates:
        return 0.0

    features = torch.tensor(
        [c["features"] for c in candidates],
        dtype=torch.float32
    )

    task_q = target(features).max()
    wait_q = target.score_wait(features)

    return max(
        float(task_q.item()),
        float(wait_q.item())
    )


def optimize(model, target, optimizer, memory):
    if len(memory) < BATCH_SIZE:
        return None

    batch = random.sample(memory, BATCH_SIZE)

    states = torch.tensor(
        [item[0] for item in batch],
        dtype=torch.float32
    )

    is_wait = torch.tensor(
        [item[1] == "wait" for item in batch],
        dtype=torch.bool
    )

    task_q = model(states)
    wait_q = model.wait_network(states).squeeze(-1)

    predictions = torch.where(
        is_wait, wait_q, task_q
    )

    targets = []

    with torch.no_grad():
        for _, _, reward, next_obs, duration, done in batch:
            value = reward

            if not done:
                value += (
                    GAMMA ** duration
                ) * best_next_q(target, next_obs)

            targets.append(value)

    targets = torch.tensor(
        targets,
        dtype=torch.float32
    )

    loss = F.smooth_l1_loss(
        predictions, targets
    )

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1.0
    )

    optimizer.step()

    return float(loss.item())


def run_episode(
    model, target, optimizer, memory,
    seed, epsilon, update_counter
):
    env = create_v2_episode(seed)

    # Decisions lasting more than one step.
    active = {}

    total_reward = 0.0
    losses = []

    for _ in range(env.max_steps):
        observations = env.get_observations()

        actions = choose_actions(
            model, observations, epsilon
        )

        # Start decisions for currently idle robots.
        for robot_id, candidates in observations.items():
            if not candidates or robot_id in active:
                continue

            task_id = actions[robot_id]

            if task_id is None:
                features = [
                    c["features"] for c in candidates
                ]

                state = [
                    sum(col) / len(col)
                    for col in zip(*features)
                ]

                kind = "wait"
            else:
                selected = next(
                    c for c in candidates
                    if c["task_id"] == task_id
                )

                state = selected["features"]
                kind = "task"

            active[robot_id] = {
                "state": state,
                "kind": kind,
                "task_id": task_id,
                "reward": 0.0,
                "duration": 0
            }

        next_obs, reward, done, info = (
            env.step(actions)
        )

        total_reward += reward

        robots_by_id = {
            r.robot_id: r
            for r in env.sim.robots
        }

        for robot_id, decision in list(active.items()):
            age = decision["duration"]

            decision["reward"] += (
                GAMMA ** age
            ) * reward

            decision["duration"] += 1

            robot = robots_by_id[robot_id]

            finished = (
                decision["kind"] == "wait"
                or robot.current_task is None
                or done
            )

            if not finished:
                continue

            memory.append((
                decision["state"],
                decision["kind"],
                decision["reward"],
                [] if done else next_obs[robot_id],
                decision["duration"],
                done
            ))

            del active[robot_id]

        # Train periodically rather than every step.
        if (
            len(memory) >= BATCH_SIZE
            and env.sim.time_step % 4 == 0
        ):
            loss = optimize(
                model, target, optimizer, memory
            )

            if loss is not None:
                losses.append(loss)
                update_counter += 1

                if update_counter % 100 == 0:
                    target.load_state_dict(
                        model.state_dict()
                    )

        if done:
            break

    return info, total_reward, losses, update_counter


def main():
    random.seed(42)
    torch.manual_seed(42)

    model = MARLQNetworkV2Wait()
    target = MARLQNetworkV2Wait()

    target.load_state_dict(model.state_dict())
    target.eval()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0005
    )

    memory = deque(maxlen=20000)
    update_counter = 0

    recent = []

    print("Starting MARL V2 training...")

    for episode in range(1, EPISODES + 1):
        epsilon = max(
            0.05,
            1.0 * (0.98 ** (episode - 1))
        )

        info, reward, losses, update_counter = (
            run_episode(
                model,
                target,
                optimizer,
                memory,
                seed=episode + 2000,
                epsilon=epsilon,
                update_counter=update_counter
            )
        )

        recent.append(info)

        if episode % 20 == 0:
            count = len(recent)

            success = sum(
                r["success"] for r in recent
            )

            avg_steps = sum(
                r["time_step"] for r in recent
            ) / count

            avg_distance = sum(
                r["distance"] for r in recent
            ) / count

            print(
                f"Episode {episode}/{EPISODES} "
                f"success={success}/{count} "
                f"steps={avg_steps:.2f} "
                f"distance={avg_distance:.2f} "
                f"epsilon={epsilon:.3f} "
                f"buffer={len(memory)} "
                f"updates={update_counter}"
            )

            recent.clear()

        if episode % 25 == 0:
            path = (
                MODEL_DIR
                / f"marl_v2_episode_{episode}.pt"
            )

            torch.save({
                "episode": episode,
                "model": model.state_dict(),
                "target": target.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epsilon": epsilon,
                "updates": update_counter
            }, path)

            print("Checkpoint saved:", path.name)

    print("\nMARL V2 TRAINING COMPLETE")
    print("Total updates:", update_counter)


if __name__ == "__main__":
    main()
