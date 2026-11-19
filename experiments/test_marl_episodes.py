
import sys
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from simulation import Simulation
from marl_environment import MARLEnvironment


def create_episode(seed):
    rng = random.Random(seed)

    sim = Simulation(rows=12, cols=12)

    positions = [
        (0, 0),
        (11, 11),
        (0, 11),
    ]

    for robot_id, position in enumerate(positions, 1):
        assert sim.add_robot(robot_id, position)

    for task_id in range(1, 9):
        pickup = (
            rng.randrange(12),
            rng.randrange(12)
        )

        delivery = pickup
        while delivery == pickup:
            delivery = (
                rng.randrange(12),
                rng.randrange(12)
            )

        priority = 3 if rng.random() < 0.4 else 1

        assert sim.add_task(
            task_id,
            pickup,
            delivery,
            priority=priority
        )

    return MARLEnvironment(
        sim,
        max_steps=200,
        max_candidates=8
    )


def baseline_actions(env):
    """
    Simple priority-aware greedy policy.
    This is NOT a trained neural network.
    """

    observations = env.get_observations()
    actions = {}
    reserved = set()

    tasks_by_id = {
        task.task_id: task
        for task in env.sim.tasks
    }

    for robot in env.sim.robots:
        candidates = observations[robot.robot_id]

        choices = [
            tasks_by_id[candidate["task_id"]]
            for candidate in candidates
            if candidate["task_id"] not in reserved
        ]

        if not choices:
            actions[robot.robot_id] = None
            continue

        def cost(task):
            distance = (
                abs(robot.position[0] - task.pickup[0])
                + abs(robot.position[1] - task.pickup[1])
            )

            return (-task.priority, distance, task.task_id)

        selected = min(choices, key=cost)

        actions[robot.robot_id] = selected.task_id
        reserved.add(selected.task_id)

    return actions


def main():
    successes = 0
    total_rewards = []

    for seed in range(10):
        env = create_episode(seed)
        total_reward = 0.0

        for _ in range(env.max_steps):
            actions = baseline_actions(env)

            observations, reward, done, info = (
                env.step(actions)
            )

            total_reward += reward

            if done:
                break

        successes += int(info["success"])
        total_rewards.append(total_reward)

        print(
            f"Seed={seed} "
            f"completed={info['completed']}/"
            f"{info['total_tasks']} "
            f"steps={info['time_step']} "
            f"distance={info['distance']} "
            f"reward={total_reward:.2f} "
            f"success={info['success']}"
        )

        assert info["collisions"] == 0

    print("\nEPISODE SUMMARY")
    print(f"Successful: {successes}/10")
    print(
        "Average reward:",
        round(sum(total_rewards) / 10, 3)
    )

    assert successes == 10, (
        "Some episodes did not finish. "
        "Inspect the failed seeds."
    )

    print("\nMARL EPISODES TEST PASSED")


if __name__ == "__main__":
    main()
