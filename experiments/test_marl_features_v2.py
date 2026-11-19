
import sys
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from test_marl_episodes import create_episode
from marl_features_v2 import get_team_features


def main():
    env = create_episode(seed=42)
    sim = env.sim

    task = sim.tasks[0]

    print("MARL FEATURES V2")
    print("=" * 50)

    results = []

    for robot in sim.robots:
        features = get_team_features(
            robot,
            task,
            sim.robots,
            sim.tasks,
            sim.environment
        )

        assert len(features) == 17
        assert all(
            math.isfinite(value)
            and 0.0 <= value <= 1.0
            for value in features
        )

        results.append(features)

        print(
            f"Robot {robot.robot_id}: "
            f"{[round(v, 3) for v in features]}"
        )

    assert len(results) == 3

    # Different robot positions should produce
    # different candidate observations.
    assert len({
        tuple(features)
        for features in results
    }) > 1

    print("\nNumber of features: 17")
    print("Robots tested: 3")
    print("\nMARL FEATURES V2 TEST PASSED")


if __name__ == "__main__":
    main()
