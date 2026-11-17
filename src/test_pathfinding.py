
from environment import GridEnvironment
from pathfinding import astar


def test_astar():
    env = GridEnvironment(10, 10)

    # Wall with one opening
    for col in range(10):
        if col != 5:
            env.add_obstacle(4, col)

    start = (0, 0)
    goal = (9, 9)

    path = astar(env, start, goal)

    assert path is not None
    assert path[0] == start
    assert path[-1] == goal
    assert len(path) - 1 == 18

    for row, col in path:
        assert env.grid[row][col] == 0

    print("PASS: A* finds the shortest path")


def test_no_path():
    env = GridEnvironment(3, 3)

    env.add_obstacle(0, 1)
    env.add_obstacle(1, 0)

    path = astar(env, (0, 0), (2, 2))

    assert path is None

    print("PASS: A* detects blocked paths")


def test_invalid_goal():
    env = GridEnvironment(5, 5)
    env.add_obstacle(4, 4)

    path = astar(env, (0, 0), (4, 4))

    assert path is None

    print("PASS: A* rejects blocked goals")


if __name__ == "__main__":
    test_astar()
    test_no_path()
    test_invalid_goal()

    print("\nALL A* TESTS PASSED!")
