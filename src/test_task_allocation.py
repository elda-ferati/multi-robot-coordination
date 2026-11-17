
from simulation import Simulation


def test_task_allocation():
    sim = Simulation(10, 10)

    assert sim.add_robot(1, (0, 0))
    assert sim.add_robot(2, (9, 9))

    assert sim.add_task(
        1,
        pickup=(1, 1),
        delivery=(4, 4),
        priority=1
    )

    assert sim.add_task(
        2,
        pickup=(8, 8),
        delivery=(5, 5),
        priority=3
    )

    sim.run(30)

    for task in sim.tasks:
        print(
            f"Task {task.task_id}: "
            f"status={task.status}, "
            f"robot={task.assigned_robot}"
        )

    assert all(
        task.status == "completed"
        for task in sim.tasks
    )

    assert sum(
        robot.completed_tasks
        for robot in sim.robots
    ) == 2

    print("PASS: Task allocation and deliveries")


if __name__ == "__main__":
    test_task_allocation()
