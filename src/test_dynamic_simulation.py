
from simulation import Simulation


def test_dynamic_simulation():
    sim = Simulation(10, 10)

    # Add robots
    assert sim.add_robot(1, (0, 0))
    assert sim.add_robot(2, (9, 9))

    # Add dynamic obstacles
    assert sim.environment.add_dynamic_obstacle(4, 4)
    assert sim.environment.add_dynamic_obstacle(5, 5)

    # Assign destinations
    assert sim.assign_goal(1, (8, 8))
    assert sim.assign_goal(2, (1, 1))

    # Run simulation for 20 time steps
    for _ in range(20):
        sim.step()

    # Check simulation history
    assert len(sim.history) == 20

    for state in sim.history:
        positions = state["positions"]
        dynamic_obstacles = state["dynamic_obstacles"]

        # Robots must not share the same position
        assert len(positions) == len(set(positions))

        # Dynamic obstacles must not share positions
        assert len(dynamic_obstacles) == len(set(dynamic_obstacles))

    print("PASS: Dynamic simulation executed")
    print("Recorded steps:", len(sim.history))


if __name__ == "__main__":
    test_dynamic_simulation()
