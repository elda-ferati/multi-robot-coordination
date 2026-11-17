
from environment import GridEnvironment
from robot import Robot
from task import Task
from task_allocation import allocate_tasks
from pathfinding import astar


class Simulation:
    def __init__(self, rows=10, cols=10):
        self.environment = GridEnvironment(rows, cols)

        self.robots = []
        self.tasks = []

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

        self.robots.append(
            Robot(robot_id, position)
        )
        return True

    def assign_goal(self, robot_id, goal):
        for robot in self.robots:
            if robot.robot_id == robot_id:
                return robot.assign_goal(
                    self.environment,
                    goal
                )

        return False

    def add_task(
        self,
        task_id,
        pickup,
        delivery,
        priority=1
    ):
        if any(t.task_id == task_id for t in self.tasks):
            return False

        for position in [pickup, delivery]:
            row, col = position

            if not self.environment.is_valid_position(row, col):
                return False

            if self.environment.grid[row][col] != 0:
                return False

        task = Task(
            task_id,
            pickup,
            delivery,
            priority
        )

        task.created_step = self.time_step
        self.tasks.append(task)

        return True

    def process_robot_task(self, robot):
        task = robot.current_task

        if task is None:
            return

        # Pickup
        if (
            not robot.carrying_package
            and robot.position == task.pickup
        ):
            robot.carrying_package = True
            task.status = "picked_up"

            robot.goal = task.delivery
            robot.path = []

        # Delivery
        if (
            robot.carrying_package
            and robot.position == task.delivery
        ):
            task.status = "completed"
            task.completed_step = self.time_step

            robot.completed_tasks += 1
            robot.carrying_package = False
            robot.current_task = None

            robot.goal = None
            robot.path = []

    def step(self):
        self.time_step += 1

        occupied = {
            robot.position
            for robot in self.robots
        }

        self.environment.move_dynamic_obstacles(
            blocked_positions=occupied
        )

        # Assign available tasks
        allocate_tasks(
            self.robots,
            self.tasks,
            self.environment
        )

        # Rotate movement priority
        count = len(self.robots)

        if count:
            offset = (self.time_step - 1) % count
            ordered_robots = (
                self.robots[offset:]
                + self.robots[:offset]
            )
        else:
            ordered_robots = []

        for robot in ordered_robots:
            self.process_robot_task(robot)

            if robot.goal is None:
                continue

            blocked = {
                r.position
                for r in self.robots
                if r.robot_id != robot.robot_id
            }

            path = astar(
                self.environment,
                robot.position,
                robot.goal,
                blocked_positions=blocked
            )

            if path is None or len(path) < 2:
                robot.path = []
                continue

            robot.position = path[1]
            robot.path = path[2:]

            self.process_robot_task(robot)

        self.history.append({
            "step": self.time_step,
            "positions": [
                r.position for r in self.robots
            ],
            "dynamic_obstacles":
                self.environment.dynamic_obstacles.copy(),
            "completed_tasks": [
                r.completed_tasks for r in self.robots
            ],
            "tasks": [
                {
                    "task_id": t.task_id,
                    "status": t.status,
                    "assigned_robot": t.assigned_robot
                }
                for t in self.tasks
            ]
        })

    def run(self, steps=20):
        for _ in range(steps):
            self.step()
