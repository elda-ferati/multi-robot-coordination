
from environment import GridEnvironment
from robot import Robot


def test_robot_movement():
    env = GridEnvironment(5, 5)
    robot = Robot(1, (0, 0))

    assert robot.assign_goal(env, (4, 4))

    for step in range(8):
        robot.move(env)

    assert robot.position == (4, 4)
    assert robot.completed_tasks == 1

    print("PASS: Robot reaches destination")


def test_robot_avoids_obstacles():
    env = GridEnvironment(5, 5)

    env.add_obstacle(0, 1)
    env.add_obstacle(1, 1)

    robot = Robot(1, (0, 0))

    assert robot.assign_goal(env, (0, 4))

    for step in range(20):
        if robot.goal is None:
            break
        robot.move(env)

    assert robot.position == (0, 4)

    print("PASS: Robot avoids obstacles")


def test_unreachable_goal():
    env = GridEnvironment(3, 3)

    env.add_obstacle(0, 1)
    env.add_obstacle(1, 0)

    robot = Robot(1, (0, 0))

    assert not robot.assign_goal(env, (2, 2))

    print("PASS: Robot rejects unreachable goal")


if __name__ == "__main__":
    test_robot_movement()
    test_robot_avoids_obstacles()
    test_unreachable_goal()

    print("\nALL ROBOT TESTS PASSED!")
