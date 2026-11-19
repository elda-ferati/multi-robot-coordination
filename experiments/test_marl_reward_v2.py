
import sys
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from test_marl_episodes import (
    create_episode,
    baseline_actions
)
from marl_environment_v2 import MARLEnvironmentV2


def main():
    original_env = create_episode(seed=42)

    env = MARLEnvironmentV2(
        original_env.sim,
        max_steps=original_env.max_steps,
        max_candidates=original_env.max_candidates
    )

    total_reward = 0.0
    rewards = []

    for _ in range(env.max_steps):
        actions = baseline_actions(env)

        observations, reward, done, info = (
            env.step(actions)
        )

        assert math.isfinite(reward)

        for candidates in observations.values():
            for candidate in candidates:
                assert len(candidate["features"]) == 17

        rewards.append(reward)
        total_reward += reward

        if done:
            break

    print("\nMARL V2 REWARD TEST")
    print("=" * 50)

    print("Completed:", info["completed"])
    print("Total tasks:", info["total_tasks"])
    print("Steps:", info["time_step"])
    print("Distance:", info["distance"])
    print("Collisions:", info["collisions"])
    print("Total reward:", round(total_reward, 3))
    print("Minimum step reward:", round(min(rewards), 3))
    print("Maximum step reward:", round(max(rewards), 3))

    assert info["success"]
    assert info["completed"] == 8
    assert info["collisions"] == 0

    # Verify positive delivery rewards and
    # negative time/movement penalties exist.
    assert any(r > 0 for r in rewards)
    assert any(r < 0 for r in rewards)

    print("\nMARL V2 REWARD TEST PASSED")


if __name__ == "__main__":
    main()
