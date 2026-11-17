
from environment import GridEnvironment
from robot import Robot


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

        if any(robot.position == position for robot in self.robots):
            return False

        if any(robot.robot_id == robot_id for robot in self.robots):
            return False

        self.robots.append(Robot(robot_id, position))
        return True

    def assign_goal(self, robot_id, goal):
        for robot in self.robots:
            if robot.robot_id == robot_id:
                return robot.assign_goal(
                    self.environment,
                    goal
                )

        return False

    def step(self):
        self.time_step += 1

        # 1. Move dynamic obstacles
        self.environment.move_dynamic_obstacles()

        # 2. Reserve robot positions
        occupied = {robot.position for robot in self.robots}
        reserved = set()
        planned_moves = []

        # 3. Plan robot movements
        for robot in self.robots:
            current = robot.position
            next_position = current

            if robot.goal is not None:
                # Replan using updated obstacle positions
                robot.assign_goal(
                    self.environment,
                    robot.goal
                )

                if robot.path:
                    candidate = robot.path[0]
                    row, col = candidate

                    if (
                        self.environment.grid[row][col] == 0
                        and candidate not in occupied
                        and candidate not in reserved
                    ):
                        next_position = candidate

            planned_moves.append((robot, next_position))
            reserved.add(next_position)

        # 4. Apply robot movements
        for robot, next_position in planned_moves:
            if next_position != robot.position:
                robot.position = next_position

                if robot.path and robot.path[0] == next_position:
                    robot.path.pop(0)

                if robot.position == robot.goal:
                    robot.completed_tasks += 1
                    robot.goal = None
                    robot.path = []

        # 5. Save simulation history
        self.history.append({
            "step": self.time_step,
            "positions": [
                robot.position for robot in self.robots
            ],
            "dynamic_obstacles":
                self.environment.dynamic_obstacles.copy()
        })

    def run(self, steps=20):
        for _ in range(steps):
            self.step()
