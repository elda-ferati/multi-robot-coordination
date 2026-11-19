
import sys
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))

from auction_allocation import allocate_tasks_auction
from hungarian_v3 import allocate_tasks_hungarian_v3
from hungarian_v4 import allocate_tasks_hungarian_v4

from benchmark_v3_v4_stress import run_scenario


ALGORITHMS = {
    "Auction": allocate_tasks_auction,
    "Hungarian V3": allocate_tasks_hungarian_v3,
    "Hungarian V4": allocate_tasks_hungarian_v4,
}

SEEDS = range(100, 200)


def average(rows, key):
    values = [
        row[key]
        for row in rows
        if row[key] is not None
    ]

    if not values:
        return None

    return sum(values) / len(values)


def main():
    results = []

    print("=" * 60)
    print("AUCTION VS HUNGARIAN V3 VS HUNGARIAN V4")
    print("=" * 60)

    for seed in SEEDS:
        for name, allocator in ALGORITHMS.items():
            result = run_scenario(
                seed,
                name,
                allocator
            )

            results.append(result)

            print(
                f"{name:15} seed={seed} "
                f"completed={result['completed']}/"
                f"{result['total_tasks']} "
                f"finish={result['finish']} "
                f"distance={result['distance']} "
                f"error={result['error']}"
            )

    print("\n" + "=" * 60)
    print("FINAL BENCHMARK SUMMARY")
    print("=" * 60)

    for name in ALGORITHMS:
        rows = [
            r for r in results
            if r["algorithm"] == name
        ]

        successful = [
            r for r in rows
            if r["error"] is None
            and r["completed"] == 40
            and r["total_tasks"] == 40
            and r["collisions"] == 0
        ]

        print(f"\n{name}")
        print("-" * 50)
        print(f"Successful: {len(successful)}/{len(rows)}")

        if len(successful) != len(rows):
            failed = [
                r["seed"] for r in rows
                if r not in successful
            ]
            print("Failed seeds:", failed)

        # Means over successful runs only.
        for key in [
            "finish",
            "distance",
            "p1_avg",
            "p3_avg",
            "runtime"
        ]:
            value = average(successful, key)

            if value is not None:
                print(f"{key}: {value:.3f}")

        print(
            "Total collisions:",
            sum(r["collisions"] for r in rows)
        )

    output = ROOT / "results"
    output.mkdir(exist_ok=True)

    filepath = (
        output / "benchmark_auction_vs_hungarian.csv"
    )

    with open(
        filepath,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nResults saved to:", filepath)


if __name__ == "__main__":
    main()
