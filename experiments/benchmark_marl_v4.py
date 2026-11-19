
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import torch

from test_marl_episodes import create_episode, baseline_actions
from marl_environment_v2 import MARLEnvironmentV2
from marl_model_v2 import MARLQNetworkV2
from marl_task_policy_v4 import select_task_actions


def create_env(seed):
    original = create_episode(seed)
    return MARLEnvironmentV2(
        original.sim,
        max_steps=original.max_steps,
        max_candidates=original.max_candidates
    )


def load_model():
    path = ROOT / "models_v4" / "marl_v4_episode_100.pt"

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    model = MARLQNetworkV2()
    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model


def evaluate(seed, policy, model):
    env = create_env(seed)

    for _ in range(env.max_steps):
        observations = env.get_observations()

        if policy == "greedy":
            actions = baseline_actions(env)
        else:
            actions = select_task_actions(model, observations)

        _, _, done, info = env.step(actions)

        if done:
            break

    return info


def average(results, key):
    return sum(r[key] for r in results) / len(results)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)

    model = load_model()
    seeds = range(7000, 7100)

    greedy_results = []
    v4_results = []

    for seed in seeds:
        greedy_results.append(evaluate(seed, "greedy", model))
        v4_results.append(evaluate(seed, "v4", model))

    print("\nMARL V4 - 100 SEED BENCHMARK")
    print("=" * 65)

    for name, results in [
        ("GREEDY", greedy_results),
        ("MARL V4", v4_results)
    ]:
        successes = sum(int(r["success"]) for r in results)
        collisions = sum(r["collisions"] for r in results)

        print(f"\n{name}")
        print(f"Success: {successes}/100")
        print(f"Average steps: {average(results, 'time_step'):.3f}")
        print(f"Average distance: {average(results, 'distance'):.3f}")
        print(f"Total collisions: {collisions}")

    # Paired comparisons on identical seeds.
    both_successful = [
        (g, v)
        for g, v in zip(greedy_results, v4_results)
        if g["success"] and v["success"]
    ]

    print("\nPAIRED COMPARISON")
    print("=" * 65)

    print("Both successful:", len(both_successful))

    if both_successful:
        distance_wins = sum(
            v["distance"] < g["distance"]
            for g, v in both_successful
        )

        step_wins = sum(
            v["time_step"] < g["time_step"]
            for g, v in both_successful
        )

        mean_distance_diff = sum(
            v["distance"] - g["distance"]
            for g, v in both_successful
        ) / len(both_successful)

        mean_steps_diff = sum(
            v["time_step"] - g["time_step"]
            for g, v in both_successful
        ) / len(both_successful)

        print("V4 distance wins:", distance_wins)
        print("V4 step wins:", step_wins)
        print("Mean distance difference:", round(mean_distance_diff, 3))
        print("Mean steps difference:", round(mean_steps_diff, 3))

    print("\nBENCHMARK COMPLETE")


if __name__ == "__main__":
    main()
