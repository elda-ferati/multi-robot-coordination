
import sys
import csv
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


SEEDS = range(31000, 31050)


def load_dynamic():
    path = (
        ROOT / "models_v4_dynamic"
        / "marl_v4_dynamic_episode_20.pt"
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


def successful(result):
    return (
        result["error"] is None
        and result["completed"] == 40
        and result["total_tasks"] == 40
        and result["collisions"] == 0
    )


def summarize(name, results):
    good = [r for r in results if successful(r)]

    print(f"\n{name}")
    print("-" * 50)
    print(f"Success: {len(good)}/{len(results)}")

    if not good:
        return

    for key in ("finish", "distance", "p1_avg", "p3_avg"):
        values = [
            r[key] for r in good
            if r[key] is not None
        ]

        if values:
            avg = sum(values) / len(values)
            print(f"{key}: {avg:.3f}")


def paired_comparison(original, dynamic):
    pairs = [
        (a, b)
        for a, b in zip(original, dynamic)
        if successful(a) and successful(b)
    ]

    print("\nDYNAMIC VS ORIGINAL — PAIRED")
    print("=" * 60)
    print("Both successful:", len(pairs))

    for key in ("finish", "distance", "p3_avg"):
        differences = [
            b[key] - a[key]
            for a, b in pairs
            if a[key] is not None and b[key] is not None
        ]

        if not differences:
            continue

        mean_diff = sum(differences) / len(differences)

        wins = sum(d < 0 for d in differences)
        losses = sum(d > 0 for d in differences)
        ties = sum(d == 0 for d in differences)

        print(
            f"{key}: mean_diff={mean_diff:+.3f} "
            f"dynamic_wins={wins} "
            f"original_wins={losses} ties={ties}"
        )


def main():
    torch.set_num_threads(2)

    original_model = load_marl_v4()
    dynamic_model = load_dynamic()

    results = {
        "Auction": [],
        "MARL Original": [],
        "MARL Dynamic": []
    }

    print("\nMARL DYNAMIC COMPARISON")
    print("=" * 60)

    for index, seed in enumerate(SEEDS, 1):
        results["Auction"].append(
            run_scenario(
                seed, "Auction",
                allocate_tasks_auction
            )
        )

        results["MARL Original"].append(
            evaluate_model(original_model, seed)
        )

        results["MARL Dynamic"].append(
            evaluate_model(dynamic_model, seed)
        )

        if index % 10 == 0:
            print(f"Completed {index}/{len(SEEDS)} seeds")

    print("\nFINAL RESULTS")
    print("=" * 60)

    for name, rows in results.items():
        summarize(name, rows)

    paired_comparison(
        results["MARL Original"],
        results["MARL Dynamic"]
    )

    output = ROOT / "results"
    output.mkdir(exist_ok=True)

    path = output / "compare_marl_dynamic.csv"

    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "algorithm", "seed", "completed", "total_tasks",
            "finish", "distance", "collisions", "runtime",
            "error", "p1_count", "p1_completed", "p1_avg",
            "p3_count", "p3_completed", "p3_avg"
        ]

        writer = csv.DictWriter(
            f, fieldnames=fieldnames
        )
        writer.writeheader()

        for name, rows in results.items():
            for row in rows:
                writer.writerow({
                    **row,
                    "algorithm": name
                })

    print("\nSaved:", path)
    print("COMPARISON COMPLETE")


if __name__ == "__main__":
    main()
