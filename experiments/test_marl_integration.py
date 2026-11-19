
import sys
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from marl_agent import MARLAgent
from test_marl_episodes import create_episode


def select_team_actions(agent, observations):
    """
    Use the shared Q-network to choose tasks.

    Reserved tasks are excluded so two robots
    cannot receive the same task.
    """
    actions = {}
    reserved = set()

    for robot_id in sorted(observations):
        candidates = observations[robot_id]

        selected = agent.select_action(
            candidates,
            excluded_tasks=reserved
        )

        if selected is None:
            actions[robot_id] = None
        else:
            task_id = selected["task_id"]
            actions[robot_id] = task_id
            reserved.add(task_id)

    return actions


def main():
    random.seed(42)
    torch.manual_seed(42)

    agent = MARLAgent(epsilon=0.0)
    agent.policy_net.eval()

    successes = 0

    for seed in range(5):
        env = create_episode(seed + 100)
        total_reward = 0.0

        for _ in range(env.max_steps):
            observations = env.get_observations()

            actions = select_team_actions(
                agent, observations
            )

            _, reward, done, info = env.step(actions)

            total_reward += reward

            if done:
                break

        successes += int(info["success"])

        print(
            f"Seed={seed + 100} "
            f"completed={info['completed']}/"
            f"{info['total_tasks']} "
            f"steps={info['time_step']} "
            f"distance={info['distance']} "
            f"reward={total_reward:.2f} "
            f"success={info['success']}"
        )

        assert info["collisions"] == 0

    print("\nINTEGRATION SUMMARY")
    print(f"Successful: {successes}/5")
    print("Network training steps:", agent.training_steps)

    assert successes == 5

    print("\nMARL INTEGRATION TEST PASSED")


if __name__ == "__main__":
    main()
