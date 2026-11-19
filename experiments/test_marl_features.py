
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from simulation import Simulation
from marl_features import get_robot_task_features


def main():
    sim = Simulation(rows=20, cols=20)

    assert sim.add_robot(1, (0, 0))
    assert sim.add_robot(2, (19, 19))

    assert sim.add_task(
        1,
        pickup=(3, 5),
        delivery=(12, 14),
        priority=3
    )

    robot = sim.robots[0]
    task = sim.tasks[0]

    features = get_robot_task_features(
        robot,
        task,
        sim.robots,
        sim.tasks,
        sim.environment
    )

    print("MARL features:", features)
    print("Number of features:", len(features))

    assert len(features) == 12
    assert all(0 <= x <= 1 for x in features)

    print("MARL FEATURES TEST PASSED")


if __name__ == "__main__":
    main()
