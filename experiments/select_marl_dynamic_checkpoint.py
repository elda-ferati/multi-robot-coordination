
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import torch

from marl_model_v2 import MARLQNetworkV2
from marl_allocation_v4 import load_marl_v4
from auction_allocation import allocate_tasks_auction
from benchmark_v3_v4_stress import run_scenario
from train_marl_v4_dynamic import evaluate_model


SEEDS = range(32000, 32030)


def load_dynamic(episode):
    path = (
        ROOT / "models_v4_dynamic"
        / f"marl_v4_dynamic_episode_{episode}.pt"
    )

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    model = MARLQNetworkV2()
    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model


def is_success(result):
    return (
        result["error"] is None
        and result["completed"] == 40
        and result["total_tasks"] == 40
        and result["collisions"] == 0
    )


def mean(rows, key):
    values = [
        r[key] for r in rows
        if r[key] is not None
    ]

    return (
        sum(values) / len(values)
        if values else None
    )


def main():
    torch.set_num_threads(2)

    models = {
        "MARL Original": load_marl_v4(),
        "Dynamic 10": load_dynamic(10),
        "Dynamic 20": load_dynamic(20)
    }

    results = {
        name: [] for name in ["Auction", *models]
    }

    print("\nDYNAMIC CHECKPOINT VALIDATION")
    print("=" * 65)

    for index, seed in enumerate(SEEDS, 1):
        results["Auction"].append(
            run_scenario(
                seed,
                "Auction",
                allocate_tasks_auction
            )
        )

        for name, model in models.items():
            results[name].append(
                evaluate_model(model, seed)
            )

        if index % 10 == 0:
            print(f"Completed {index}/30 seeds")

    print("\nVALIDATION RESULTS")
    print("=" * 65)

    for name, rows in results.items():
        good = [r for r in rows if is_success(r)]

        print(f"\n{name}")
        print(f"Success: {len(good)}/30")

        if good:
            for key in (
                "finish", "distance",
                "p1_avg", "p3_avg"
            ):
                value = mean(good, key)
                if value is not None:
                    print(f"{key}: {value:.3f}")

    print("\nPAIRED COMPARISON VS AUCTION")
    print("=" * 65)

    auction = results["Auction"]

    for name in models:
        paired = [
            (a, b)
            for a, b in zip(auction, results[name])
            if is_success(a) and is_success(b)
        ]

        print(f"\n{name}")
        print("Both successful:", len(paired))

        for key in ("finish", "distance", "p3_avg"):
            differences = [
                b[key] - a[key]
                for a, b in paired
                if a[key] is not None
                and b[key] is not None
            ]

            if not differences:
                continue

            average = sum(differences) / len(differences)
            wins = sum(d < 0 for d in differences)

            print(
                f"{key}: "
                f"mean_diff={average:+.3f} "
                f"MARL_wins={wins}/{len(differences)}"
            )

    print("\nCHECKPOINT VALIDATION COMPLETE")


if __name__ == "__main__":
    main()
