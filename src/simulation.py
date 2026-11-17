
from environment import GridEnvironment
from robot import Robot
from pathfinding import astar


class Simulation:
    def __init__(self, rows=10, cols=10):
        self.environment = GridEnvironment(rows, cols)
        self.robots = []
        self.time_step = 0
        self.history = []

    def add_robot(self, robot_id, position):
        row, col = position

        if not self.environment.is_valid_position(row, col):
            return False

        if self.environment.grid[row][col] != 0:
            return False

        if any(r.position == position for r in self.robots):
            return False

        if any(r.robot_id == robot_id for r in self.robots):
            return False

        self.robots.append(Robot(robot_id, position))
        return True

    def assign_goal(self, robot_id, goal):
        for robot in self.robots:
            if robot.robot_id == robot_id:
                return robot.assign_goal(
                    self.environment, goal
                )

        return False

    def step(self):
        self.time_step += 1

        # Current robot positions
        occupied = {r.position for r in self.robots}

        # Move dynamic obstacles without entering robot cells
        self.environment.move_dynamic_obstacles(
            blocked_positions=occupied
        )

        # Rotate robot priority at every time step
        count = len(self.robots)

        if count:
            offset = (self.time_step - 1) % count
            ordered_robots = (
                self.robots[offset:] + self.robots[:offset]
            )
        else:
            ordered_robots = []

        # Move robots sequentially
        for robot in ordered_robots:
            if robot.goal is None:
                continue

            current = robot.position

            # Treat other robots as temporary obstacles
            blocked = {
                r.position
                for r in self.robots
                if r.robot_id != robot.robot_id
            }

            # Find a new path with A*
            path = astar(
                self.environment,
                current,
                robot.goal,
                blocked_positions=blocked
            )

            if path is None or len(path) < 2:
                robot.path = []
                continue

            next_position = path[1]

            # Move only one cell per time step
            robot.position = next_position
            robot.path = path[2:]

            # Check if destination is reached
            if robot.position == robot.goal:
                robot.completed_tasks += 1
                robot.goal = None
                robot.path = []

        # Record simulation history
        self.history.append({
            "step": self.time_step,
            "positions": [
                r.position for r in self.robots
            ],
            "dynamic_obstacles":
                self.environment.dynamic_obstacles.copy(),
            "completed_tasks": [
                r.completed_tasks for r in self.robots
            ]
        })

    def run(self, steps=20):
        for _ in range(steps):
            self.step()
