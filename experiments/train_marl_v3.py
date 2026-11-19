
import sys
import random
import argparse
from pathlib import Path
from collections import deque

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import torch
import torch.nn.functional as F

from test_marl_episodes import (
    create_episode,
    baseline_actions
)
from marl_environment_v2 import MARLEnvironmentV2
from marl_wait_policy import (
    MARLQNetworkV2Wait,
    select_actions_with_wait
)


GAMMA = 0.99
BATCH_SIZE = 32

MODEL_DIR = ROOT / "models_v3"
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
        return select_actions_with_wait(model, observations)

    actions = {}
    reserved = set()

    for rid in sorted(observations):
        legal = [
            c["task_id"]
            for c in observations[rid]
            if c["task_id"] not in reserved
        ]

        if not legal or random.random() < 0.15:
            actions[rid] = None
        else:
            task_id = random.choice(legal)
            actions[rid] = task_id
            reserved.add(task_id)

    return actions


def wait_features(candidates):
    values = [c["features"] for c in candidates]

    return [
        sum(column) / len(column)
        for column in zip(*values)
    ]


def optimize(model, target, optimizer, memory):
    if len(memory) < BATCH_SIZE:
        return None

    batch = random.sample(memory, BATCH_SIZE)

    states = torch.tensor(
        [item[0] for item in batch],
        dtype=torch.float32
    )

    wait_mask = torch.tensor(
        [item[1] == "wait" for item in batch],
        dtype=torch.bool
    )

    model.train()

    task_q = model(states)
    wait_q = model.wait_network(states).squeeze(-1)

    predictions = torch.where(
        wait_mask, wait_q, task_q
    )

    targets = []

    with torch.no_grad():
        for _, _, reward, next_obs, duration, done in batch:
            q_target = reward

            if not done and next_obs:
                next_x = torch.tensor(
                    next_obs,
                    dtype=torch.float32
                )

                next_task_q = target(next_x).max()
                next_wait_q = target.score_wait(next_x)

                best_next = torch.maximum(
                    next_task_q, next_wait_q
                )

                q_target += (
                    GAMMA ** duration
                ) * float(best_next.item())

            targets.append(q_target)

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
    seed, epsilon, updates
):
    env = create_env(seed)
    active = {}

    wait_count = 0
    opportunities = 0
    losses = []

    for _ in range(env.max_steps):
        obs = env.get_observations()

        actions = choose_actions(
            model, obs, epsilon
        )

        robots = {
            r.robot_id: r
            for r in env.sim.robots
        }

        tasks = {
            t.task_id: t
            for t in env.sim.tasks
        }

        # Track only new decisions by idle robots.
        for rid, candidates in obs.items():
            if not candidates:
                continue

            opportunities += 1
            task_id = actions[rid]

            if task_id is None:
                wait_count += 1

                state = wait_features(candidates)

                # WAIT is a one-step decision.
                active[rid] = {
                    "kind": "wait",
                    "state": state,
                    "task_id": None
                }

            else:
                selected = next(
                    c for c in candidates
                    if c["task_id"] == task_id
                )

                active[rid] = {
                    "kind": "task",
                    "state": selected["features"],
                    "task_id": task_id,
                    "start_step": env.sim.time_step,
                    "start_distance":
                        env.sim.robot_distance[rid]
                }

        next_obs, _, done, info = env.step(actions)

        for rid, decision in list(active.items()):
            if decision["kind"] == "wait":
                # Individual waiting cost.
                urgent_pending = sum(
                    t.priority >= 3
                    and t.status == "pending"
                    for t in env.sim.tasks
                )

                reward = -0.8 - 0.1 * urgent_pending

                memory.append((
                    decision["state"],
                    "wait",
                    reward,
                    [
                        c["features"]
                        for c in next_obs[rid]
                    ],
                    1,
                    done
                ))

                del active[rid]
                continue

            task = tasks[decision["task_id"]]

            if task.status != "completed" and not done:
                continue

            duration = max(
                1,
                env.sim.time_step - decision["start_step"]
            )

            traveled = (
                env.sim.robot_distance[rid]
                - decision["start_distance"]
            )

            if task.status == "completed":
                reward = (
                    (24.0 if task.priority >= 3 else 12.0)
                    - 0.10 * duration
                    - 0.03 * traveled
                )
            else:
                # Task not completed at time limit.
                reward = -12.0 - 0.10 * duration

            memory.append((
                decision["state"],
                "task",
                reward,
                [
                    c["features"]
                    for c in next_obs[rid]
                ],
                duration,
                done
            ))

            del active[rid]

        if (
            len(memory) >= BATCH_SIZE
            and env.sim.time_step % 4 == 0
        ):
            loss = optimize(
                model, target, optimizer, memory
            )

            if loss is not None:
                losses.append(loss)
                updates += 1

                if updates % 100 == 0:
                    target.load_state_dict(
                        model.state_dict()
                    )

        if done:
            break

    wait_rate = (
        100 * wait_count / opportunities
        if opportunities else 0.0
    )

    return info, wait_rate, losses, updates


def evaluate(model, seeds, greedy=False):
    results = []
    total_waits = 0
    opportunities = 0

    for seed in seeds:
        env = create_env(seed)

        for _ in range(env.max_steps):
            obs = env.get_observations()

            if greedy:
                actions = baseline_actions(env)
            else:
                actions = select_actions_with_wait(
                    model, obs
                )

                for rid, candidates in obs.items():
                    if candidates:
                        opportunities += 1
                        if actions[rid] is None:
                            total_waits += 1

            _, _, done, info = env.step(actions)

            if done:
                break

        results.append(info)

    n = len(results)

    return {
        "success": sum(r["success"] for r in results),
        "steps": sum(r["time_step"] for r in results) / n,
        "distance": sum(r["distance"] for r in results) / n,
        "collisions": sum(r["collisions"] for r in results),
        "wait_rate": (
            100 * total_waits / opportunities
            if opportunities else 0.0
        )
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes",
        type=int,
        default=20
    )
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(2)

    model = MARLQNetworkV2Wait()
    target = MARLQNetworkV2Wait()

    target.load_state_dict(model.state_dict())
    target.eval()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0005
    )

    memory = deque(maxlen=20000)
    updates = 0

    print("\nSTARTING MARL V3 TRAINING")

    recent = []

    for episode in range(1, args.episodes + 1):
        epsilon = max(
            0.05,
            0.98 ** (episode - 1)
        )

        info, wait_rate, losses, updates = run_episode(
            model,
            target,
            optimizer,
            memory,
            seed=3000 + episode,
            epsilon=epsilon,
            updates=updates
        )

        recent.append((info, wait_rate))

        if episode % 10 == 0:
            n = len(recent)

            success = sum(
                r[0]["success"] for r in recent
            )

            steps = sum(
                r[0]["time_step"] for r in recent
            ) / n

            distance = sum(
                r[0]["distance"] for r in recent
            ) / n

            waits = sum(
                r[1] for r in recent
            ) / n

            print(
                f"Episode {episode}/{args.episodes} "
                f"success={success}/{n} "
                f"steps={steps:.2f} "
                f"distance={distance:.2f} "
                f"wait={waits:.2f}% "
                f"epsilon={epsilon:.3f} "
                f"updates={updates}"
            )

            recent.clear()

    checkpoint = (
        MODEL_DIR /
        f"marl_v3_episode_{args.episodes}.pt"
    )

    torch.save({
        "model": model.state_dict(),
        "target": target.state_dict(),
        "optimizer": optimizer.state_dict(),
        "episodes": args.episodes,
        "updates": updates
    }, checkpoint)

    print("\nCHECKPOINT SAVED:", checkpoint.name)

    # New seeds, not used during training.
    test_seeds = range(4000, 4010)

    greedy_results = evaluate(
        model, test_seeds, greedy=True
    )

    v3_results = evaluate(
        model, test_seeds, greedy=False
    )

    print("\nMARL V3 SMOKE EVALUATION")
    print("=" * 55)

    for name, result in [
        ("GREEDY", greedy_results),
        ("V3", v3_results)
    ]:
        print(
            f"{name}: "
            f"success={result['success']}/10 "
            f"steps={result['steps']:.2f} "
            f"distance={result['distance']:.2f} "
            f"wait={result['wait_rate']:.2f}% "
            f"collisions={result['collisions']}"
        )

    print("\nMARL V3 TEST COMPLETE")


if __name__ == "__main__":
    main()
