
import sys
import random
import argparse
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import torch
import torch.nn.functional as F

from test_marl_episodes import create_episode, baseline_actions
from marl_environment_v2 import MARLEnvironmentV2
from marl_model_v2 import MARLQNetworkV2
from marl_task_policy_v4 import select_task_actions


GAMMA = 0.99
BATCH_SIZE = 32
LEARNING_RATE = 0.0003

MODEL_DIR = ROOT / "models_v4"
MODEL_DIR.mkdir(exist_ok=True)


def create_env(seed):
    original = create_episode(seed)

    return MARLEnvironmentV2(
        original.sim,
        max_steps=original.max_steps,
        max_candidates=original.max_candidates
    )


def choose_actions(model, observations, epsilon):
    if random.random() >= epsilon:
        return select_task_actions(model, observations)

    actions = {
        rid: None for rid in observations
    }

    reserved = set()

    robots = list(observations.keys())
    random.shuffle(robots)

    for rid in robots:
        available = [
            c["task_id"]
            for c in observations[rid]
            if c["task_id"] not in reserved
        ]

        if available:
            task_id = random.choice(available)
            actions[rid] = task_id
            reserved.add(task_id)

    return actions


def double_dqn_next_value(online, target, candidates):
    if not candidates:
        return 0.0

    x = torch.tensor(
        candidates,
        dtype=torch.float32
    )

    # Online network selects the next task.
    online_q = online(x).reshape(-1)
    best_index = int(torch.argmax(online_q).item())

    # Target network evaluates that selected task.
    target_q = target(x).reshape(-1)

    return float(target_q[best_index].item())


def optimize(online, target, optimizer, replay):
    if len(replay) < BATCH_SIZE:
        return None

    batch = random.sample(replay, BATCH_SIZE)

    states = torch.tensor(
        [item[0] for item in batch],
        dtype=torch.float32
    )

    online.train()

    predicted = online(states).reshape(-1)

    targets = []

    with torch.no_grad():
        online.eval()
        target.eval()

        for state, reward, next_features, duration, done in batch:
            value = reward

            if not done and next_features:
                next_value = double_dqn_next_value(
                    online, target, next_features
                )

                value += (GAMMA ** duration) * next_value

            targets.append(value)

    online.train()

    expected = torch.tensor(
        targets,
        dtype=torch.float32
    )

    loss = F.smooth_l1_loss(predicted, expected)

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        online.parameters(), 1.0
    )

    optimizer.step()

    return float(loss.item())


def run_training_episode(
    online, target, optimizer,
    replay, seed, epsilon, updates
):
    env = create_env(seed)

    active = {}
    losses = []

    previous_distance = dict(env.sim.robot_distance)

    for _ in range(env.max_steps):
        observations = env.get_observations()

        actions = choose_actions(
            online, observations, epsilon
        )

        tasks_by_id = {
            task.task_id: task
            for task in env.sim.tasks
        }

        # Record decisions made by idle robots.
        for rid, candidates in observations.items():
            task_id = actions.get(rid)

            if task_id is None or rid in active:
                continue

            candidate = next(
                c for c in candidates
                if c["task_id"] == task_id
            )

            active[rid] = {
                "state": candidate["features"],
                "task_id": task_id,
                "duration": 0,
                "return": 0.0
            }

        next_obs, _, done, info = env.step(actions)

        # Accumulate discounted time and movement costs
        # for each active robot.
        for rid, decision in list(active.items()):
            current_distance = env.sim.robot_distance[rid]

            moved = max(
                0,
                current_distance - previous_distance.get(
                    rid, current_distance
                )
            )

            age = decision["duration"]

            step_cost = -0.12 - 0.04 * moved

            decision["return"] += (
                GAMMA ** age
            ) * step_cost

            decision["duration"] += 1

            task = tasks_by_id[decision["task_id"]]

            completed = task.status == "completed"

            if not completed and not done:
                continue

            duration = decision["duration"]
            reward = decision["return"]

            if completed:
                bonus = (
                    24.0 if task.priority >= 3
                    else 12.0
                )

                reward += (
                    GAMMA ** (duration - 1)
                ) * bonus

            else:
                reward -= (
                    GAMMA ** (duration - 1)
                ) * 15.0

            next_features = [
                c["features"]
                for c in next_obs.get(rid, [])
            ]

            replay.append((
                decision["state"],
                reward,
                next_features,
                duration,
                done
            ))

            del active[rid]

        previous_distance = dict(
            env.sim.robot_distance
        )

        if (
            len(replay) >= BATCH_SIZE
            and env.sim.time_step % 4 == 0
        ):
            loss = optimize(
                online, target, optimizer, replay
            )

            if loss is not None:
                losses.append(loss)
                updates += 1

                if updates % 100 == 0:
                    target.load_state_dict(
                        online.state_dict()
                    )

        if done:
            break

    return info, losses, updates


def evaluate(model, seeds, greedy=False):
    results = []

    model.eval()

    for seed in seeds:
        env = create_env(seed)

        for _ in range(env.max_steps):
            observations = env.get_observations()

            if greedy:
                actions = baseline_actions(env)
            else:
                actions = select_task_actions(
                    model, observations
                )

            _, _, done, info = env.step(actions)

            if done:
                break

        results.append(info)

    count = len(results)

    return {
        "success": sum(
            int(r["success"]) for r in results
        ),
        "steps": sum(
            r["time_step"] for r in results
        ) / count,
        "distance": sum(
            r["distance"] for r in results
        ) / count,
        "collisions": sum(
            r["collisions"] for r in results
        )
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--episodes", type=int, default=20
    )

    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(2)

    online = MARLQNetworkV2()
    target = MARLQNetworkV2()

    target.load_state_dict(
        online.state_dict()
    )

    target.eval()

    optimizer = torch.optim.Adam(
        online.parameters(),
        lr=LEARNING_RATE
    )

    replay = deque(maxlen=20000)

    updates = 0
    recent = []

    print("\nSTARTING MARL V4 DOUBLE DQN TRAINING")

    for episode in range(1, args.episodes + 1):
        epsilon = max(
            0.05,
            0.98 ** (episode - 1)
        )

        info, losses, updates = run_training_episode(
            online,
            target,
            optimizer,
            replay,
            seed=5000 + episode,
            epsilon=epsilon,
            updates=updates
        )

        recent.append(info)

        if episode % 10 == 0:
            n = len(recent)

            success = sum(
                int(r["success"])
                for r in recent
            )

            steps = sum(
                r["time_step"]
                for r in recent
            ) / n

            distance = sum(
                r["distance"]
                for r in recent
            ) / n

            print(
                f"Episode {episode}/{args.episodes} "
                f"success={success}/{n} "
                f"steps={steps:.2f} "
                f"distance={distance:.2f} "
                f"epsilon={epsilon:.3f} "
                f"buffer={len(replay)} "
                f"updates={updates}"
            )

            recent.clear()

    checkpoint_path = (
        MODEL_DIR /
        f"marl_v4_episode_{args.episodes}.pt"
    )

    torch.save({
        "model": online.state_dict(),
        "target": target.state_dict(),
        "optimizer": optimizer.state_dict(),
        "episodes": args.episodes,
        "updates": updates
    }, checkpoint_path)

    print("\nCHECKPOINT SAVED:", checkpoint_path.name)

    # These seeds are separate from training.
    seeds = range(6000, 6010)

    greedy_result = evaluate(
        online, seeds, greedy=True
    )

    v4_result = evaluate(
        online, seeds, greedy=False
    )

    print("\nMARL V4 SMOKE EVALUATION")
    print("=" * 60)

    for name, result in [
        ("GREEDY", greedy_result),
        ("V4", v4_result)
    ]:
        print(
            f"{name}: "
            f"success={result['success']}/10 "
            f"steps={result['steps']:.2f} "
            f"distance={result['distance']:.2f} "
            f"collisions={result['collisions']}"
        )

    print("\nMARL V4 TEST COMPLETE")


if __name__ == "__main__":
    main()
