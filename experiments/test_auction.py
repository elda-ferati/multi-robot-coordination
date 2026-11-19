
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))

import simulation
from auction_allocation import allocate_tasks_auction


def main():
    # Use Auction without changing simulation.py
    simulation.allocate_tasks = allocate_tasks_auction

    sim = simulation.Simulation(rows=10, cols=10)

    assert sim.add_robot(1, (0, 0))
    assert sim.add_robot(2, (9, 9))

    assert sim.add_task(
        task_id=1,
        pickup=(0, 2),
        delivery=(0, 7),
        priority=3
    )

    assert sim.add_task(
        task_id=2,
        pickup=(8, 8),
        delivery=(5, 5),
        priority=1
    )

    for _ in range(50):
        sim.step()

        if all(
            task.status == "completed"
            for task in sim.tasks
        ):
            break

    sim.print_metrics()

    assert all(
        task.status == "completed"
        for task in sim.tasks
    ), "Some tasks were not completed"

    assert sim.collision_count == 0

    print("\nAUCTION BASIC TEST PASSED")


if __name__ == "__main__":
    main()
