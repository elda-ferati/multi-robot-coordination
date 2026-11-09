
from environment import GridEnvironment


def test_grid_creation():
    env = GridEnvironment(5, 5)

    assert len(env.grid) == 5
    assert all(len(row) == 5 for row in env.grid)
    assert all(cell == 0 for row in env.grid for cell in row)

    print("PASS: Grid creation")


def test_invalid_dimensions():
    try:
        GridEnvironment(0, 5)
    except ValueError:
        print("PASS: Invalid grid dimensions")
        return

    raise AssertionError("Invalid dimensions were accepted")


def test_static_obstacles():
    env = GridEnvironment(5, 5)

    env.add_obstacle(2, 2)
    assert env.grid[2][2] == 1

    env.add_random_obstacles(5)

    static_count = sum(row.count(1) for row in env.grid)
    assert static_count == 6

    print("PASS: Static obstacles")


def test_dynamic_movement():
    env = GridEnvironment(5, 5)

    env.add_dynamic_obstacle(2, 2)

    for step in range(100):
        env.move_dynamic_obstacles()

        assert len(env.dynamic_obstacles) == 1

        row, col = env.dynamic_obstacles[0]

        assert env.is_valid_position(row, col)
        assert env.grid[row][col] == 2

    print("PASS: Dynamic movement")


def test_trapped_obstacle():
    env = GridEnvironment(5, 5)

    env.add_obstacle(1, 2)
    env.add_obstacle(3, 2)
    env.add_obstacle(2, 1)
    env.add_obstacle(2, 3)

    env.add_dynamic_obstacle(2, 2)

    for step in range(10):
        env.move_dynamic_obstacles()
        assert env.dynamic_obstacles == [(2, 2)]

    print("PASS: Trapped obstacle")


def test_multiple_dynamic_obstacles():
    env = GridEnvironment(10, 10)

    for i in range(10):
        assert env.add_random_dynamic_obstacle()

    for step in range(100):
        env.move_dynamic_obstacles()

        positions = env.dynamic_obstacles

        assert len(positions) == 10
        assert len(set(positions)) == 10

        for row, col in positions:
            assert env.is_valid_position(row, col)
            assert env.grid[row][col] == 2

        dynamic_count = sum(row.count(2) for row in env.grid)
        assert dynamic_count == 10

    print("PASS: Multiple dynamic obstacles")


def test_add_remove():
    env = GridEnvironment(5, 5)

    assert env.add_random_dynamic_obstacle()
    assert len(env.dynamic_obstacles) == 1

    assert env.remove_random_dynamic_obstacle()
    assert len(env.dynamic_obstacles) == 0

    assert not env.remove_random_dynamic_obstacle()

    print("PASS: Adding and removing obstacles")


def test_full_grid():
    env = GridEnvironment(2, 2)

    env.add_random_obstacles(4)

    assert not env.add_random_dynamic_obstacle()

    static_count = sum(row.count(1) for row in env.grid)
    assert static_count == 4

    print("PASS: Full grid")


if __name__ == "__main__":
    test_grid_creation()
    test_invalid_dimensions()
    test_static_obstacles()
    test_dynamic_movement()
    test_trapped_obstacle()
    test_multiple_dynamic_obstacles()
    test_add_remove()
    test_full_grid()

    print("\nALL TESTS PASSED!")
