
import sys
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import torch

from benchmark_marl_v4 import (
    load_model,
    evaluate
)


def bootstrap_ci(differences, samples=5000):
    rng = random.Random(42)
    n = len(differences)

    averages = []

    for _ in range(samples):
        sample = [
            differences[rng.randrange(n)]
            for _ in range(n)
        ]
        averages.append(statistics.mean(sample))

    averages.sort()

    lower = averages[int(0.025 * samples)]
    upper = averages[int(0.975 * samples)]

    return lower, upper


def summarize(name, differences):
    wins = sum(d < 0 for d in differences)
    losses = sum(d > 0 for d in differences)
    ties = sum(d == 0 for d in differences)

    mean = statistics.mean(differences)
    median = statistics.median(differences)

    lower, upper = bootstrap_ci(differences)

    print(f"\n{name}")
    print("-" * 45)
    print("V4 wins:", wins)
    print("Greedy wins:", losses)
    print("Ties:", ties)
    print("Mean difference:", round(mean, 3))
    print("Median difference:", round(median, 3))
    print(
        "95% bootstrap CI:",
        f"[{lower:.3f}, {upper:.3f}]"
    )


def main():
    torch.set_num_threads(2)

    model = load_model()

    distance_diffs = []
    step_diffs = []

    for seed in range(7000, 7100):
        greedy = evaluate(seed, "greedy", model)
        v4 = evaluate(seed, "v4", model)

        if not (greedy["success"] and v4["success"]):
            raise RuntimeError(
                f"Unsuccessful episode: seed={seed}"
            )

        distance_diffs.append(
            v4["distance"] - greedy["distance"]
        )

        step_diffs.append(
            v4["time_step"] - greedy["time_step"]
        )

    print("\nMARL V4 STATISTICAL ANALYSIS")
    print("=" * 55)

    summarize("DISTANCE", distance_diffs)
    summarize("STEPS", step_diffs)

    print("\nANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
