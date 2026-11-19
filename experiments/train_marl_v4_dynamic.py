
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

import simulation
from dynamic_tasks import DynamicTaskGenerator
from marl_model_v2 import MARLQNetworkV2
from marl_features_v2 import get_team_features
from marl_task_policy_v4 import select_task_actions
from auction_allocation import allocate_tasks_auction
from benchmark_v3_v4_stress import run_scenario


GAMMA = 0.99
BATCH_SIZE = 32
LR = 0.0002
MAX_STEPS = 500
MAX_CANDIDATES = 8

OUTPUT = ROOT / "models_v4_dynamic"
OUTPUT.mkdir(exist_ok=True)


def create_simulation(seed):
    random.seed(seed)

    sim = simulation.Simulation(rows=20, cols=20)

    for row in (5, 10, 15):
        for col in range(2, 18):
            if col % 6 not in (0, 1):
                sim.environment.add_obstacle(row, col)

    positions = [
        (0, 0),
        (0, 19),
        (19, 0),
        (19, 19),
        (9, 9)
    ]

    for rid, position in enumerate(positions, 1):
        assert sim.add_robot(rid, position)

    for _ in range(5):
        for _attempt in range(100):
            position = (
                random.randrange(20),
                random.randrange(20)
            )

            row, col = position

            if sim.environment.grid[row][col] != 0:
                continue

            if position in [
                robot.position for robot in sim.robots
            ]:
                continue

            sim.environment.add_dynamic_obstacle(
                row, col
            )
            break

    generator = DynamicTaskGenerator(
        simulation=sim,
        seed=seed + 10000,
        interval=5,
        tasks_per_interval=4,
        urgent_probability=0.45,
        max_new_tasks=40
    )

    return sim, generator


def observations_from_sim(sim):
    pending = sorted(
        (
            task for task in sim.tasks
            if task.status == "pending"
        ),
        key=lambda task: (
            -task.priority, task.task_id
        )
    )

    observations = {}

    for robot in sorted(
        sim.robots,
        key=lambda r: r.robot_id
    ):
        if (
            robot.current_task is not None
            or robot.goal is not None
        ):
            continue

        candidates = []

        for task in pending[:MAX_CANDIDATES]:
            features = get_team_features(
                robot,
                task,
                sim.robots,
                sim.tasks,
                sim.environment
            )

            candidates.append({
                "task_id": task.task_id,
                "features": features
            })

        if candidates:
            observations[robot.robot_id] = candidates

    return observations


def choose_actions(model, obs, epsilon):
    if random.random() >= epsilon:
        return select_task_actions(model, obs)

    actions = {rid: None for rid in obs}
    reserved = set()

    ids = list(obs)
    random.shuffle(ids)

    for rid in ids:
        available = [
            c["task_id"]
            for c in obs[rid]
            if c["task_id"] not in reserved
        ]

        if available:
            task_id = random.choice(available)
            actions[rid] = task_id
            reserved.add(task_id)

    return actions


def next_q_double_dqn(online, target, features):
    if not features:
        return 0.0

    x = torch.tensor(
        features, dtype=torch.float32
    )

    selected = online(x).reshape(-1).argmax()
    return float(
        target(x).reshape(-1)[selected].item()
    )


def optimize(online, target, optimizer, replay):
    if len(replay) < BATCH_SIZE:
        return None

    batch = random.sample(replay, BATCH_SIZE)

    states = torch.tensor(
        [item[0] for item in batch],
        dtype=torch.float32
    )

    online.eval()
    target.eval()

    with torch.no_grad():
        expected = []

        for state, reward, next_state, duration, terminal in batch:
            value = reward

            if not terminal and next_state:
                value += (
                    GAMMA ** duration
                    * next_q_double_dqn(
                        online, target, next_state
                    )
                )

            expected.append(value)

    targets = torch.tensor(
        expected, dtype=torch.float32
    )

    online.train()
    prediction = online(states).reshape(-1)

    loss = F.smooth_l1_loss(
        prediction, targets
    )

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        online.parameters(), 1.0
    )

    optimizer.step()

    return float(loss.item())


def train_episode(
    online, target, optimizer,
    replay, seed, epsilon, updates
):
    sim, generator = create_simulation(seed)

    active = {}
    previous_distance = dict(sim.robot_distance)

    def training_allocator(robots, tasks, environment):
        obs = observations_from_sim(sim)
        actions = choose_actions(online, obs, epsilon)

        robot_lookup = {
            r.robot_id: r for r in robots
        }

        task_lookup = {
            t.task_id: t for t in tasks
        }

        for rid, task_id in actions.items():
            if task_id is None or rid in active:
                continue

            candidate = next(
                c for c in obs[rid]
                if c["task_id"] == task_id
            )

            robot = robot_lookup[rid]
            task = task_lookup[task_id]

            if task.status != "pending":
                continue

            if robot.assign_task(task, environment):
                active[rid] = {
                    "features": candidate["features"],
                    "task_id": task_id,
                    "duration": 0,
                    "return": 0.0
                }

    original_allocator = simulation.allocate_tasks
    simulation.allocate_tasks = training_allocator

    try:
        for _ in range(MAX_STEPS):
            sim.step()

            if sim.time_step <= 50:
                generator.update()

            tasks = {
                task.task_id: task
                for task in sim.tasks
            }

            all_finished = (
                sim.time_step >= 50
                and len(sim.tasks) == 40
                and all(
                    t.status == "completed"
                    for t in sim.tasks
                )
            )

            terminal = (
                all_finished
                or sim.time_step >= MAX_STEPS
            )

            next_obs = observations_from_sim(sim)

            for rid, decision in list(active.items()):
                moved = max(
                    0,
                    sim.robot_distance[rid]
                    - previous_distance[rid]
                )

                age = decision["duration"]

                step_reward = -0.12 - 0.04 * moved

                decision["return"] += (
                    GAMMA ** age
                ) * step_reward

                decision["duration"] += 1

                task = tasks[decision["task_id"]]
                completed = task.status == "completed"

                if not completed and not terminal:
                    continue

                duration = decision["duration"]
                reward = decision["return"]

                bonus = (
                    24.0 if task.priority >= 3
                    else 12.0
                )

                if completed:
                    reward += (
                        GAMMA ** (duration - 1)
                    ) * bonus
                else:
                    reward -= (
                        GAMMA ** (duration - 1)
                    ) * 15.0

                features = [
                    c["features"]
                    for c in next_obs.get(rid, [])
                ]

                replay.append((
                    decision["features"],
                    reward,
                    features,
                    duration,
                    terminal
                ))

                del active[rid]

            previous_distance = dict(
                sim.robot_distance
            )

            if (
                len(replay) >= BATCH_SIZE
                and sim.time_step % 4 == 0
            ):
                loss = optimize(
                    online, target,
                    optimizer, replay
                )

                if loss is not None:
                    updates += 1

                    if updates % 100 == 0:
                        target.load_state_dict(
                            online.state_dict()
                        )

            if terminal:
                break

    finally:
        simulation.allocate_tasks = original_allocator

    completed = sum(
        t.status == "completed"
        for t in sim.tasks
    )

    return {
        "success": (
            completed == 40
            and len(sim.tasks) == 40
        ),
        "finish": sim.time_step,
        "distance": sim.total_distance,
        "collisions": sim.collision_count,
        "updates": updates
    }


def evaluate_model(model, seed):
    def allocator(robots, tasks, environment):
        # Use the same 17-feature inference policy,
        # without exploration.
        from marl_allocation_v4 import (
            allocate_tasks_marl_v4
        )

        # Temporarily provide the fine-tuned model
        # through the existing cached loader.
        from marl_allocation_v4 import load_marl_v4

        old_loader = load_marl_v4.__wrapped__

        # Direct allocation, avoiding checkpoint changes.
        pending = [
            t for t in tasks
            if t.status == "pending"
        ]
        idle = [
            r for r in robots
            if r.current_task is None
            and r.goal is None
        ]

        if not pending or not idle:
            return

        pending.sort(
            key=lambda t: (-t.priority, t.task_id)
        )

        observations = {}

        for robot in sorted(
            idle, key=lambda r: r.robot_id
        ):
            observations[robot.robot_id] = [
                {
                    "task_id": task.task_id,
                    "features": get_team_features(
                        robot, task,
                        robots, tasks, environment
                    )
                }
                for task in pending[:MAX_CANDIDATES]
            ]

        actions = select_task_actions(
            model, observations
        )

        robot_lookup = {
            r.robot_id: r for r in robots
        }
        task_lookup = {
            t.task_id: t for t in tasks
        }

        for rid, tid in actions.items():
            if tid is not None:
                task = task_lookup[tid]
                if task.status == "pending":
                    robot_lookup[rid].assign_task(
                        task, environment
                    )

    return run_scenario(
        seed, "MARL V4 Dynamic", allocator
    )


def summarize(name, results):
    good = [
        r for r in results
        if r["error"] is None
        and r["completed"] == 40
        and r["total_tasks"] == 40
        and r["collisions"] == 0
    ]

    print(
        f"{name}: success={len(good)}/{len(results)}",
        end=""
    )

    if good:
        print(
            f" finish={sum(r['finish'] for r in good)/len(good):.2f}"
            f" distance={sum(r['distance'] for r in good)/len(good):.2f}"
            f" urgent={sum(r['p3_avg'] for r in good)/len(good):.2f}"
        )
    else:
        print()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--episodes", type=int, default=20
    )

    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(2)

    checkpoint = torch.load(
        ROOT / "models_v4" / "marl_v4_episode_100.pt",
        map_location="cpu",
        weights_only=True
    )

    online = MARLQNetworkV2()
    online.load_state_dict(checkpoint["model"])

    target = MARLQNetworkV2()
    target.load_state_dict(online.state_dict())
    target.eval()

    optimizer = torch.optim.Adam(
        online.parameters(), lr=LR
    )

    replay = deque(maxlen=20000)
    updates = 0
    recent = []

    print("\nMARL V4 DYNAMIC TRAINING")
    print("=" * 60)

    for episode in range(1, args.episodes + 1):
        epsilon = max(
            0.05,
            0.4 * (0.97 ** (episode - 1))
        )

        result = train_episode(
            online, target, optimizer,
            replay,
            seed=10000 + episode,
            epsilon=epsilon,
            updates=updates
        )

        updates = result["updates"]
        recent.append(result)

        if episode % 5 == 0:
            n = len(recent)

            print(
                f"Episode {episode}/{args.episodes} "
                f"success={sum(r['success'] for r in recent)}/{n} "
                f"finish={sum(r['finish'] for r in recent)/n:.2f} "
                f"distance={sum(r['distance'] for r in recent)/n:.2f} "
                f"epsilon={epsilon:.3f} "
                f"updates={updates}"
            )

            recent.clear()

        if episode % 10 == 0:
            path = (
                OUTPUT /
                f"marl_v4_dynamic_episode_{episode}.pt"
            )

            torch.save({
                "model": online.state_dict(),
                "target": target.state_dict(),
                "optimizer": optimizer.state_dict(),
                "episodes": episode,
                "updates": updates
            }, path)

            print("Saved:", path.name)

    print("\nDYNAMIC VALIDATION")
    print("=" * 60)

    seeds = range(30000, 30010)

    marl_results = [
        evaluate_model(online, seed)
        for seed in seeds
    ]

    auction_results = [
        run_scenario(
            seed, "Auction", allocate_tasks_auction
        )
        for seed in seeds
    ]

    summarize("Auction", auction_results)
    summarize("MARL Dynamic", marl_results)

    print("\nMARL V4 DYNAMIC TEST COMPLETE")


if __name__ == "__main__":
    main()
