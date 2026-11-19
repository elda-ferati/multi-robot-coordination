
import sys
import csv
import random
import time
from pathlib import Path
from datetime import datetime, timezone

# Paths relative to project folder
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from simulation import Simulation

RESULTS_DIR = ROOT / "results"
RESULTS_FILE = RESULTS_DIR / "experiments.csv"


def create_simulation(seed=42):
    random.seed(seed)

    sim = Simulation(20, 20)

    # Static warehouse shelves
    for row in [5, 10, 15]:
        for col in range(3, 17):
            if col not in [6, 13]:
                sim.environment.add_obstacle(row, col)

    # Dynamic obstacles
    assert sim.environment.add_dynamic_obstacle(7, 8)
    assert sim.environment.add_dynamic_obstacle(12, 12)

    # Robots
    assert sim.add_robot(1, (0, 0))
    assert sim.add_robot(2, (0, 19))
    assert sim.add_robot(3, (19, 10))

    # Delivery tasks
    assert sim.add_task(1, (2, 2), (18, 18), priority=1)
    assert sim.add_task(2, (2, 17), (18, 2), priority=3)
    assert sim.add_task(3, (17, 10), (1, 10), priority=2)
    assert sim.add_task(4, (8, 3), (12, 17), priority=1)
    assert sim.add_task(5, (12, 3), (3, 15), priority=3)
    assert sim.add_task(6, (17, 17), (2, 3), priority=2)

    return sim


def get_finish_step(sim):
    for state in sim.history:
        if len(state["tasks"]) == len(sim.tasks):
            if all(t["status"] == "completed" for t in state["tasks"]):
                return state["step"]
    return None


def save_results(result):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fields = [
        "experiment_id",
        "timestamp",
        "algorithm",
        "seed",
        "grid_size",
        "robots",
        "tasks",
        "completed_tasks",
        "completion_rate",
        "finish_step",
        "total_distance",
        "collisions",
        "astar_calls_simulation",
        "replanning_events",
        "total_waiting_steps",
        "runtime_seconds",
    ]

    file_exists = RESULTS_FILE.exists()

    with RESULTS_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)

        if not file_exists or RESULTS_FILE.stat().st_size == 0:
            writer.writeheader()

        writer.writerow(result)

    print(f"\nResults saved: {RESULTS_FILE}")


def run_experiment(seed=42, steps=150):
    sim = create_simulation(seed)

    start = time.perf_counter()
    sim.run(steps)
    runtime = time.perf_counter() - start

    completed = sum(t.status == "completed" for t in sim.tasks)
    total_tasks = len(sim.tasks)

    result = {
        "experiment_id": datetime.now(timezone.utc).strftime(
            "%Y%m%d_%H%M%S_%f"
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "algorithm": "Global Auction + Smart A*",
        "seed": seed,
        "grid_size": "20x20",
        "robots": len(sim.robots),
        "tasks": total_tasks,
        "completed_tasks": completed,
        "completion_rate": round(
            100 * completed / total_tasks, 2
        ) if total_tasks else 100,
        "finish_step": get_finish_step(sim),
        "total_distance": sim.total_distance,
        "collisions": sim.collision_count,
        "astar_calls_simulation": sim.astar_calls,
        "replanning_events": sim.replanning_events,
        "total_waiting_steps": sum(sim.robot_waiting.values()),
        "runtime_seconds": round(runtime, 6),
    }

    save_results(result)
    sim.print_metrics()

    return result


if __name__ == "__main__":
    run_experiment(seed=42, steps=150)
