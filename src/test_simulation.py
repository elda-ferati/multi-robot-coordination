
from simulation import Simulation


def test_multiple_robots():
    sim = Simulation(10, 10)

    assert sim.add_robot(1, (0, 0))
    assert sim.add_robot(2, (9, 9))
    assert sim.add_robot(3, (0, 9))

    assert sim.assign_goal(1, (8, 8))
    assert sim.assign_goal(2, (1, 1))
    assert sim.assign_goal(3, (9, 0))

    for _ in range(30):
        sim.step()

        positions = [robot.position for robot in sim.robots]

        assert len(positions) == len(set(positions))

    print("PASS: Robots do not occupy the same cell")


def test_duplicate_position():
    sim = Simulation(5, 5)

    assert sim.add_robot(1, (0, 0))
    assert not sim.add_robot(2, (0, 0))

    print("PASS: Duplicate robot positions rejected")


if __name__ == "__main__":
    test_multiple_robots()
    test_duplicate_position()

    print("\nALL SIMULATION TESTS PASSED!")
