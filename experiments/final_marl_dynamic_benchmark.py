
import sys
import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import torch

from marl_allocation_v4 import load_marl_v4
from select_marl_dynamic_checkpoint import load_dynamic
from train_marl_v4_dynamic import evaluate_model
from benchmark_v3_v4_stress import run_scenario
from auction_allocation import allocate_tasks_auction
from hungarian_v3 import allocate_tasks_hungarian_v3
from hungarian_v4 import allocate_tasks_hungarian_v4


SEEDS = range(33000, 33100)


def successful(row):
    return (
        row["error"] is None
        and row["completed"] == 40
        and row["total_tasks"] == 40
        and row["collisions"] == 0
    )


def mean(rows, key):
    values = [
        row[key] for row in rows
        if row.get(key) is not None
    ]
    return sum(values) / len(values) if values else None


def bootstrap_ci(values, n_boot=3000):
    rng = random.Random(42)
    n = len(values)

    estimates = []
    for _ in range(n_boot):
        sample = [
            values[rng.randrange(n)]
            for _ in range(n)
        ]
        estimates.append(sum(sample) / n)

    estimates.sort()
    return (
        estimates[int(n_boot * 0.025)],
        estimates[int(n_boot * 0.975)]
    )


def main():
    torch.set_num_threads(2)

    models = {
        "MARL Original": load_marl_v4(),
        "Dynamic 10": load_dynamic(10),
        "Dynamic 20": load_dynamic(20)
    }

    allocators = {
        "Auction": allocate_tasks_auction,
        "Hungarian V3": allocate_tasks_hungarian_v3,
        "Hungarian V4": allocate_tasks_hungarian_v4
    }

    names = list(allocators) + list(models)
    results = {name: {} for name in names}

    print("\nFINAL MARL DYNAMIC BENCHMARK")
    print("=" * 65)

    for index, seed in enumerate(SEEDS, start=1):
        for name, allocator in allocators.items():
            result = run_scenario(
                seed, name, allocator
            )
            results[name][seed] = result

        for name, model in models.items():
            result = evaluate_model(model, seed)
            result["algorithm"] = name
            results[name][seed] = result

        if index % 10 == 0:
            print(f"Completed {index}/100 seeds")

    print("\nFINAL RESULTS")
    print("=" * 65)

    for name in names:
        rows = list(results[name].values())
        good = [r for r in rows if successful(r)]

        print(f"\n{name}")
        print(f"Success: {len(good)}/100")
        print(
            "Total collisions:",
            sum(r["collisions"] for r in rows)
        )

        for key in (
            "finish", "distance",
            "p1_avg", "p3_avg"
        ):
            value = mean(good, key)
            if value is not None:
                print(f"{key}: {value:.3f}")

    print("\nDYNAMIC 20 VS AUCTION")
    print("=" * 65)

    paired = [
        (
            results["Auction"][seed],
            results["Dynamic 20"][seed]
        )
        for seed in SEEDS
        if successful(results["Auction"][seed])
        and successful(results["Dynamic 20"][seed])
    ]

    print("Both successful:", len(paired))

    for key in ("finish", "distance", "p3_avg"):
        diffs = [
            dynamic[key] - auction[key]
            for auction, dynamic in paired
            if auction.get(key) is not None
            and dynamic.get(key) is not None
        ]

        if not diffs:
            continue

        lower, upper = bootstrap_ci(diffs)

        print(f"\n{key}")
        print(f"Mean difference: {sum(diffs)/len(diffs):+.3f}")
        print(f"Dynamic wins: {sum(d < 0 for d in diffs)}")
        print(f"Auction wins: {sum(d > 0 for d in diffs)}")
        print(f"Ties: {sum(d == 0 for d in diffs)}")
        print(
            f"95% bootstrap CI: "
            f"[{lower:.3f}, {upper:.3f}]"
        )

    output = ROOT / "results"
    output.mkdir(exist_ok=True)

    path = output / "final_marl_dynamic_benchmark.csv"

    fieldnames = [
        "algorithm", "seed", "completed", "total_tasks",
        "finish", "distance", "collisions", "runtime",
        "error", "p1_count", "p1_completed", "p1_avg",
        "p3_count", "p3_completed", "p3_avg"
    ]

    with open(
        path, "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=fieldnames
        )
        writer.writeheader()

        for name in names:
            for seed in SEEDS:
                writer.writerow(results[name][seed])

    print("\nSaved:", path)
    print("FINAL BENCHMARK COMPLETE")


if __name__ == "__main__":
    main()
