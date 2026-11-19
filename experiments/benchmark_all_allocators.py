
import sys
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EXPERIMENTS = ROOT / "experiments"

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(EXPERIMENTS))

import torch

from auction_allocation import allocate_tasks_auction
from hungarian_v3 import allocate_tasks_hungarian_v3
from hungarian_v4 import allocate_tasks_hungarian_v4
from marl_allocation_v4 import (
    allocate_tasks_marl_v4,
    load_marl_v4
)
from benchmark_v3_v4_stress import run_scenario


ALGORITHMS = {
    "Auction": allocate_tasks_auction,
    "Hungarian V3": allocate_tasks_hungarian_v3,
    "Hungarian V4": allocate_tasks_hungarian_v4,
    "MARL V4": allocate_tasks_marl_v4,
}

SEEDS = range(100, 200)
EXPECTED_TASKS = 40


def successful(result):
    return (
        result["error"] is None
        and result["completed"] == EXPECTED_TASKS
        and result["total_tasks"] == EXPECTED_TASKS
        and result["collisions"] == 0
    )


def mean(rows, key):
    values = [
        row[key]
        for row in rows
        if row.get(key) is not None
    ]

    if not values:
        return None

    return sum(values) / len(values)


def print_summary(results):
    print("\n" + "=" * 70)
    print("FINAL BENCHMARK SUMMARY")
    print("=" * 70)

    for name in ALGORITHMS:
        rows = [
            r for r in results
            if r["algorithm"] == name
        ]

        valid = [r for r in rows if successful(r)]

        print(f"\n{name}")
        print("-" * 50)
        print(f"Success: {len(valid)}/{len(rows)}")

        failed_seeds = [
            r["seed"]
            for r in rows
            if not successful(r)
        ]

        if failed_seeds:
            print("Failed seeds:", failed_seeds)

        for key in [
            "finish",
            "distance",
            "p1_avg",
            "p3_avg",
            "runtime"
        ]:
            value = mean(valid, key)

            if value is not None:
                print(f"{key}: {value:.3f}")

        print(
            "Total collisions:",
            sum(r["collisions"] for r in rows)
        )


def compare_to_marl(results):
    print("\n" + "=" * 70)
    print("PAIRED COMPARISON VS MARL V4")
    print("=" * 70)

    by_algorithm = {
        name: {
            r["seed"]: r
            for r in results
            if r["algorithm"] == name
        }
        for name in ALGORITHMS
    }

    marl = by_algorithm["MARL V4"]

    for name in ALGORITHMS:
        if name == "MARL V4":
            continue

        baseline = by_algorithm[name]

        paired = [
            (baseline[seed], marl[seed])
            for seed in SEEDS
            if successful(baseline[seed])
            and successful(marl[seed])
        ]

        print(f"\nMARL V4 vs {name}")
        print("-" * 50)
        print("Both successful:", len(paired))

        if not paired:
            continue

        for key in [
            "finish",
            "distance",
            "p3_avg"
        ]:
            pairs = [
                (a[key], b[key])
                for a, b in paired
                if a.get(key) is not None
                and b.get(key) is not None
            ]

            if not pairs:
                continue

            differences = [
                b - a for a, b in pairs
            ]

            average_diff = (
                sum(differences) / len(differences)
            )

            wins = sum(
                difference < 0
                for difference in differences
            )

            ties = sum(
                difference == 0
                for difference in differences
            )

            print(
                f"{key}: "
                f"mean_diff={average_diff:+.3f} "
                f"MARL_wins={wins}/{len(pairs)} "
                f"ties={ties}"
            )


def main():
    torch.set_num_threads(2)

    # Warm-load the model before the benchmark.
    load_marl_v4()

    results = []

    print("\nMULTI-ROBOT ALLOCATION BENCHMARK")
    print("=" * 70)
    print("Algorithms:", ", ".join(ALGORITHMS))
    print("Seeds: 100-199")

    for index, seed in enumerate(SEEDS, start=1):
        for name, allocator in ALGORITHMS.items():
            try:
                result = run_scenario(
                    seed,
                    name,
                    allocator
                )

            except Exception as exc:
                result = {
                    "algorithm": name,
                    "seed": seed,
                    "completed": 0,
                    "total_tasks": EXPECTED_TASKS,
                    "finish": None,
                    "distance": None,
                    "collisions": 0,
                    "runtime": None,
                    "error": repr(exc),
                    "p1_avg": None,
                    "p3_avg": None,
                }

            results.append(result)

        if index % 10 == 0:
            print(
                f"Completed {index}/{len(SEEDS)} seeds"
            )

    print_summary(results)
    compare_to_marl(results)

    output = ROOT / "results"
    output.mkdir(exist_ok=True)

    path = output / "benchmark_all_allocators.csv"

    fieldnames = list(dict.fromkeys(
        key
        for result in results
        for key in result.keys()
    ))

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nResults saved to:", path)
    print("BENCHMARK COMPLETE")


if __name__ == "__main__":
    main()
