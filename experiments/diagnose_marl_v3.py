
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import torch

from test_marl_episodes import (
    create_episode,
    baseline_actions
)

from marl_environment_v2 import MARLEnvironmentV2
from marl_wait_policy import (
    MARLQNetworkV2Wait,
    select_actions_with_wait
)
from marl_wait_guard import WaitGuardPolicy


CHECKPOINT = (
    ROOT / "models_v3" / "marl_v3_episode_100.pt"
)


def load_model():
    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True
    )

    model = MARLQNetworkV2Wait()

    model.load_state_dict(
        checkpoint["model"]
    )

    model.eval()

    return model


def evaluate(seed, policy, model):
    original = create_episode(seed)

    env = MARLEnvironmentV2(
        original.sim,
        max_steps=original.max_steps,
        max_candidates=original.max_candidates
    )

    guard = WaitGuardPolicy(
        model,
        max_wait=2
    )

    waits = 0
    opportunities = 0

    for _ in range(env.max_steps):
        observations = env.get_observations()

        if policy == "greedy":
            actions = baseline_actions(env)

        elif policy == "original":
            actions = select_actions_with_wait(
                model, observations
            )

        else:
            actions = guard.select_actions(
                observations
            )

        for robot_id, candidates in observations.items():
            if not candidates:
                continue

            opportunities += 1

            if actions[robot_id] is None:
                waits += 1

        _, _, done, info = env.step(actions)

        if done:
            break

    return {
        "success": info["success"],
        "steps": info["time_step"],
        "distance": info["distance"],
        "collisions": info["collisions"],
        "waits": waits,
        "opportunities": opportunities
    }


def main():
    torch.set_num_threads(2)

    model = load_model()

    seeds = range(4100, 4130)

    print("\nMARL V3 DIAGNOSTIC")
    print("=" * 65)

    for policy in ["greedy", "original", "guarded"]:
        results = [
            evaluate(seed, policy, model)
            for seed in seeds
        ]

        n = len(results)

        success = sum(
            r["success"] for r in results
        )

        steps = sum(
            r["steps"] for r in results
        ) / n

        distance = sum(
            r["distance"] for r in results
        ) / n

        collisions = sum(
            r["collisions"] for r in results
        )

        waits = sum(
            r["waits"] for r in results
        )

        opportunities = sum(
            r["opportunities"] for r in results
        )

        wait_rate = (
            100 * waits / opportunities
            if opportunities else 0
        )

        print(
            f"{policy:10} "
            f"success={success}/{n} "
            f"steps={steps:.2f} "
            f"distance={distance:.2f} "
            f"wait={wait_rate:.2f}% "
            f"collisions={collisions}"
        )

    print("\nMARL V3 DIAGNOSTIC COMPLETE")


if __name__ == "__main__":
    main()
