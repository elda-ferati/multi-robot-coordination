
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from simulation import Simulation
from marl_environment import MARLEnvironment


def main():
    sim = Simulation(rows=10, cols=10)

    assert sim.add_robot(1, (0, 0))
    assert sim.add_robot(2, (9, 9))

    assert sim.add_task(
        1, (0, 2), (0, 7), priority=3
    )

    assert sim.add_task(
        2, (8, 8), (5, 5), priority=1
    )

    env = MARLEnvironment(sim, max_steps=50)

    observations = env.get_observations()

    print("Initial observations:", observations)

    # Temporary manual actions.
    # Later these will come from the neural network.
    actions = {
        1: 1,
        2: 2
    }

    total_reward = 0

    for step in range(50):
        observations, reward, done, info = env.step(
            actions if step == 0 else {}
        )

        total_reward += reward

        if done:
            break

    print("\nFinal info:", info)
    print("Total reward:", round(total_reward, 3))

    assert info["success"]
    assert info["completed"] == 2
    assert info["collisions"] == 0

    print("\nMARL ENVIRONMENT TEST PASSED")


if __name__ == "__main__":
    main()
