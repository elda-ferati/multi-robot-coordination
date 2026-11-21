r"""Best-found / delta analysis for Multi-Robot Coordination.

Run from the project root:
    .\.venv\Scripts\python.exe best_found_delta_analysis.py

Reads results/final_marl_dynamic_benchmark.csv and writes three reports to
results/best_found_analysis/.  Uses only Python's standard library.

IMPORTANT: 'best found' means the lowest observed cost among the compared
algorithms for the SAME seed. It does not certify a global optimum.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "results" / "final_marl_dynamic_benchmark.csv"
OUTPUT = ROOT / "results" / "best_found_analysis"
METRICS = {
    "finish": "Total completion steps",
    "distance": "Total robot distance",
    "p1_avg": "Mean normal-task completion steps",
    "p3_avg": "Mean urgent-task completion steps",
}
REQUIRED = {"seed", "algorithm", "completed", "total_tasks", "collisions", "error", *METRICS}


def load_results(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    by_seed = defaultdict(dict)
    all_algorithms = set()
    failed = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")
        for number, row in enumerate(reader, start=2):
            seed = str(row["seed"]).strip()
            algorithm = str(row["algorithm"]).strip()
            if not seed or not algorithm:
                raise ValueError(f"Missing seed or algorithm at CSV line {number}")
            if algorithm in by_seed[seed]:
                raise ValueError(f"Duplicate result for seed={seed}, algorithm={algorithm}")
            all_algorithms.add(algorithm)
            try:
                completed = int(row["completed"])
                total = int(row["total_tasks"])
                collisions = int(row["collisions"])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid counts at CSV line {number}: {exc}") from exc
            error = str(row["error"] or "").strip()
            success = (error in ("", "None") and total > 0
                       and completed == total and collisions == 0)
            values = {}
            for key in METRICS:
                raw = str(row[key] or "").strip()
                try:
                    value = float(raw) if raw else None
                except (ValueError, TypeError):
                    value = None
                values[key] = value if value is not None and math.isfinite(value) else None
            if not success:
                failed.append((seed, algorithm, f"completed={completed}/{total}, collisions={collisions}, error={error}"))
            by_seed[seed][algorithm] = {"valid": success, **values}
    if not all_algorithms:
        raise ValueError("CSV contains no algorithm results")
    return by_seed, sorted(all_algorithms), failed


def seed_sort(seed):
    try:
        return (0, int(seed))
    except ValueError:
        return (1, seed)


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def fmt(x, digits=3):
    return "N/A" if x is None else f"{x:.{digits}f}"


def analyze():
    by_seed, algorithms, failed = load_results(INPUT)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Only compare on complete, collision-free observations where every
    # algorithm is present and has a finite value for that metric.
    detail = []
    summaries = []
    coverage = {}
    for metric, title in METRICS.items():
        collected = defaultdict(list)
        usable_seeds = 0
        excluded = 0
        for seed in sorted(by_seed, key=seed_sort):
            entries = by_seed[seed]
            if any(a not in entries or not entries[a]["valid"] or
                   entries[a][metric] is None for a in algorithms):
                excluded += 1
                continue
            usable_seeds += 1
            values = {a: entries[a][metric] for a in algorithms}
            best = min(values.values())
            winners = sorted(a for a, val in values.items()
                             if math.isclose(val, best, rel_tol=1e-12, abs_tol=1e-9))
            for algorithm in algorithms:
                cost = values[algorithm]
                delta = max(0.0, cost - best)  # zero for tied best
                gap_pct = (100 * delta / abs(best)) if abs(best) > 1e-12 else (0.0 if delta == 0 else None)
                record = {
                    "metric": metric, "seed": seed, "algorithm": algorithm,
                    "cost": cost, "best_found_cost": best,
                    "best_found_algorithm_s": " | ".join(winners),
                    "delta": delta, "gap_percent": gap_pct,
                    "is_best_tie": int(algorithm in winners),
                }
                detail.append(record)
                collected[algorithm].append(record)
        coverage[metric] = (usable_seeds, excluded)
        for algorithm in algorithms:
            rows = collected[algorithm]
            gaps = [r["gap_percent"] for r in rows if r["gap_percent"] is not None]
            summaries.append({
                "metric": metric,
                "algorithm": algorithm,
                "seeds_compared": len(rows),
                "best_or_tied_count": sum(r["is_best_tie"] for r in rows),
                "mean_cost": mean(r["cost"] for r in rows) if rows else None,
                "mean_delta": mean(r["delta"] for r in rows) if rows else None,
                "mean_gap_percent": mean(gaps) if gaps else None,
                "median_gap_percent": median(gaps) if gaps else None,
                "max_gap_percent": max(gaps) if gaps else None,
            })

    detail_path = OUTPUT / "best_found_delta_by_seed.csv"
    summary_path = OUTPUT / "best_found_delta_summary.csv"
    text_path = OUTPUT / "best_found_readme.txt"
    write_csv(detail_path,
              ["metric", "seed", "algorithm", "cost", "best_found_cost",
               "best_found_algorithm_s", "delta", "gap_percent", "is_best_tie"], detail)
    write_csv(summary_path,
              ["metric", "algorithm", "seeds_compared", "best_or_tied_count",
               "mean_cost", "mean_delta", "mean_gap_percent", "median_gap_percent",
               "max_gap_percent"], summaries)

    lines = [
        "MULTI-ROBOT COORDINATION — BEST-FOUND + DELTA ANALYSIS",
        "=" * 66,
        f"Input: {INPUT.name}",
        f"Algorithms compared: {len(algorithms)} ({', '.join(algorithms)})",
        f"Unique seeds in input: {len(by_seed)}",
        f"Runs that did not complete successfully / had collisions: {len(failed)}",
        "",
        "DEFINITIONS (each seed, each metric, separately)",
        "best_found = minimum observed cost over compared algorithms",
        "delta = algorithm_cost - best_found  [lower is better]",
        "gap_percent = 100 * delta / abs(best_found) (when best_found != 0)",
        "Best/tied count credits ALL algorithms tied at minimum; sums may exceed seed count.",
        "Only seeds where every algorithm succeeded and has a usable value are included.",
        "Best-found is an empirical reference, NOT a proven global optimum.",
        "Metrics are independent: do not combine winners across metrics.",
        "",
    ]
    for metric, title in METRICS.items():
        used, excluded = coverage[metric]
        lines += [f"{title.upper()} ({metric})", f"Comparable seeds: {used}; excluded: {excluded}",
                  f"{'Algorithm':<22} {'Mean cost':>12} {'Best/ties':>12} {'Mean delta':>12} {'Mean gap %':>12}"]
        rows = [r for r in summaries if r["metric"] == metric]
        for r in sorted(rows, key=lambda x: x["mean_cost"] if x["mean_cost"] is not None else float("inf")):
            lines.append(f"{r['algorithm'][:22]:<22} {fmt(r['mean_cost']):>12} "
                         f"{r['best_or_tied_count']:>12} {fmt(r['mean_delta']):>12} "
                         f"{fmt(r['mean_gap_percent']):>12}")
        lines.append("")
    if failed:
        lines.append("Unsuccessful examples (up to 10):")
        for s, a, reason in failed[:10]:
            lines.append(f"  seed={s}, algorithm={a}: {reason}")
    lines += [
        "INTERPRETATION:",
        "Mean gap is the average per-seed relative deviation from best found.",
        "This differs from comparing overall averages across seeds.",
        "Gap and delta are not statistical significance tests.",
        "Because workloads are dynamic, a best-known assignment may not be globally optimal.",
        "For an optimality proof, an exact solver or a mathematically justified lower bound is required.",
    ]
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\nGenerated:")
    for path in (detail_path, summary_path, text_path):
        print(f"  {path}")


if __name__ == "__main__":
    try:
        analyze()
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nERROR: {exc}")
        raise SystemExit(1)
