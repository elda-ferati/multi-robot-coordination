
from pathfinding import astar


class Robot:
    def __init__(self, robot_id, start_position):
        self.robot_id = robot_id
        self.position = start_position
        self.goal = None
        self.path = []
        self.completed_tasks = 0

    def assign_goal(self, environment, goal):
        path = astar(environment, self.position, goal)

        if path is None:
            return False

        self.goal = goal
        self.path = path[1:]
        return True

    def move(self, environment):
        if not self.path:
            return False

        next_position = self.path[0]
        row, col = next_position

        if environment.grid[row][col] != 0:
            if not self.assign_goal(environment, self.goal):
                return False

            if not self.path:
                return False

            next_position = self.path[0]

        self.position = next_position
        self.path.pop(0)

        if self.position == self.goal:
            self.completed_tasks += 1
            self.goal = None
            self.path = []

        return True
