r"""Iterative Best-So-Far tracking over existing multi-robot benchmark runs.

Run from project root:
    .\.venv\Scripts\python.exe best_so_far_tracking.py

Reads results/final_marl_dynamic_benchmark.csv (read-only).
Writes NEW outputs in results/best_so_far_tracking/.
No source file is overwritten or deleted.

The algorithm order is a PRESENTATION/COMPARISON order, not evidence of the
actual historical implementation order. A lower metric is considered better.
Each metric and seed is tracked separately; best-so-far is not a global optimum.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "results" / "final_marl_dynamic_benchmark.csv"
OUTPUT = ROOT / "results" / "best_so_far_tracking"
METRICS = {
    "finish": "Total completion steps",
    "distance": "Total robot distance",
    "p1_avg": "Mean normal-task completion steps",
    "p3_avg": "Mean urgent-task completion steps",
}
# The chosen comparison order is explicit and reproducible.
ORDER = ["Auction", "Hungarian V3", "Hungarian V4", "MARL Original", "Dynamic 10", "Dynamic 20"]
REQUIRED = {"algorithm", "seed", "completed", "total_tasks", "collisions", "error", *METRICS}


def seed_key(s):
    try:
        return (0, int(s))
    except ValueError:
        return (1, s)


def load_data():
    if not INPUT.is_file():
        raise FileNotFoundError(f"Missing input CSV: {INPUT}")
    data = defaultdict(dict)
    with INPUT.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
        for line, r in enumerate(reader, 2):
            seed, algo = (r["seed"] or "").strip(), (r["algorithm"] or "").strip()
            if not seed or not algo:
                raise ValueError(f"Missing seed/algorithm on line {line}")
            if algo in data[seed]:
                raise ValueError(f"Duplicate seed {seed} / algorithm {algo} on line {line}")
            try:
                successful = (int(r["completed"]) == int(r["total_tasks"])
                              and int(r["total_tasks"]) > 0
                              and int(r["collisions"]) == 0
                              and (r["error"] or "").strip() in ("", "None"))
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid status on line {line}: {e}") from e
            values = {}
            for metric in METRICS:
                try:
                    number = float(r[metric])
                except (ValueError, TypeError):
                    number = float("nan")
                values[metric] = number if math.isfinite(number) else None
            data[seed][algo] = {"successful": successful, **values}
    if not data:
        raise ValueError("The input CSV is empty")
    available = {algo for group in data.values() for algo in group}
    ordered = [algo for algo in ORDER if algo in available]
    ordered += sorted(available - set(ordered))
    if len(ordered) < 2:
        raise ValueError("Need at least two algorithms to compare")
    return data, ordered


def write_rows(filename, cols, rows):
    with (OUTPUT / filename).open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def main():
    data, order = load_data()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    details, summaries, overview = [], [], []
    for metric, title in METRICS.items():
        valid_seeds = [seed for seed in sorted(data, key=seed_key)
                       if all(algo in data[seed]
                              and data[seed][algo]["successful"]
                              and data[seed][algo][metric] is not None
                              for algo in order)]
        if not valid_seeds:
            raise ValueError(f"No comparable complete seeds for metric '{metric}'")
        grouped = defaultdict(list)
        for seed in valid_seeds:
            best_cost = None
            best_algo = None
            for index, algo in enumerate(order, 1):
                cost = data[seed][algo][metric]
                previous_best = best_cost
                delta = None if previous_best is None else cost - previous_best
                improved = previous_best is None or cost < previous_best - 1e-9
                if improved:
                    best_cost, best_algo = cost, algo
                improvement = None if delta is None else max(0.0, -delta)
                row = {
                    "metric": metric, "seed": seed, "iteration": index,
                    "algorithm": algo, "candidate_cost": cost,
                    "previous_best_cost": previous_best if previous_best is not None else "",
                    "delta_vs_previous_best": delta if delta is not None else "",
                    "improved_best": int(improved),
                    "improvement_amount": improvement if improvement is not None else "",
                    "best_so_far_cost": best_cost,
                    "best_so_far_algorithm": best_algo,
                    "gap_from_current_best_pct": (100 * (cost - best_cost) / abs(best_cost)
                                                  if best_cost != 0 else (0 if cost == 0 else "")),
                }
                details.append(row)
                grouped[index].append(row)
        for index, algo in enumerate(order, 1):
            rows = grouped[index]
            summaries.append({
                "metric": metric, "iteration": index, "algorithm": algo,
                "seeds_compared": len(rows),
                "candidate_mean_cost": mean(float(r["candidate_cost"]) for r in rows),
                "best_so_far_mean_cost": mean(float(r["best_so_far_cost"]) for r in rows),
                "new_best_count": sum(int(r["improved_best"]) for r in rows),
                "mean_improvement_when_positive":
                    (mean(float(r["improvement_amount"]) for r in rows
                          if r["improvement_amount"] != "" and float(r["improvement_amount"]) > 1e-9)
                     if any(r["improvement_amount"] != "" and float(r["improvement_amount"]) > 1e-9
                            for r in rows) else 0.0),
            })
        overview.append((metric, title, len(valid_seeds)))

    write_rows("best_so_far_history_by_seed.csv",
               ["metric", "seed", "iteration", "algorithm", "candidate_cost",
                "previous_best_cost", "delta_vs_previous_best", "improved_best",
                "improvement_amount", "best_so_far_cost", "best_so_far_algorithm",
                "gap_from_current_best_pct"], details)
    write_rows("best_so_far_progress_summary.csv",
               ["metric", "iteration", "algorithm", "seeds_compared",
                "candidate_mean_cost", "best_so_far_mean_cost", "new_best_count",
                "mean_improvement_when_positive"], summaries)
    lines = [
        "BEST-SO-FAR ITERATIVE TRACKING — MULTI-ROBOT COORDINATION", "=" * 72,
        f"Input: {INPUT.name}", f"Comparison order: {' -> '.join(order)}",
        "The order is chosen for analysis, NOT a verified development timeline.",
        "Each seed and metric has its OWN evolving best (minimum cost).",
        "Delta = candidate cost - previous best; negative Delta means improvement.",
        "First candidate initializes best and has no Delta.",
        "This is retrospective tracking over existing runs, NOT a new search optimizer.",
        "Best-so-far is empirical; it does not prove a global mathematical optimum.", "",
    ]
    for metric, title, nseeds in overview:
        lines.extend([f"{title.upper()} ({metric}) — {nseeds} comparable seeds",
                      f"{'Step':>4}  {'Algorithm':<19} {'Mean candidate':>15} {'Mean best so far':>17} {'New best':>9}"])
        for row in (r for r in summaries if r["metric"] == metric):
            lines.append(f"{row['iteration']:>4}  {row['algorithm']:<19} "
                         f"{row['candidate_mean_cost']:>15.3f} "
                         f"{row['best_so_far_mean_cost']:>17.3f} "
                         f"{row['new_best_count']:>9}")
        lines.append("")
    lines += ["Generated: best_so_far_history_by_seed.csv, best_so_far_progress_summary.csv",
              "The original benchmark and previous Best Found / Delta reports remain unchanged."]
    (OUTPUT / "best_so_far_readme.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nReports folder: {OUTPUT}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}")
        raise SystemExit(1)
