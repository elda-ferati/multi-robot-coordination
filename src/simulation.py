
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

        self.astar_calls = 0
        self.replanning_events = 0
        self.total_distance = 0
        self.collision_count = 0

        self.robot_distance = {}
        self.robot_waiting = {}
        self.blocked_steps = {}

        self.max_wait = 2

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

        robot = Robot(robot_id, position)
        self.robots.append(robot)

        self.robot_distance[robot_id] = 0
        self.robot_waiting[robot_id] = 0
        self.blocked_steps[robot_id] = 0

        return True

    def assign_goal(self, robot_id, goal):
        for robot in self.robots:
            if robot.robot_id == robot_id:
                return robot.assign_goal(
                    self.environment, goal
                )
        return False

    def add_task(
        self, task_id, pickup, delivery, priority=1
    ):
        if any(t.task_id == task_id for t in self.tasks):
            return False

        if priority < 1:
            return False

        for position in (pickup, delivery):
            row, col = position

            if not self.environment.is_valid_position(row, col):
                return False

            if self.environment.grid[row][col] != 0:
                return False

        task = Task(
            task_id, pickup, delivery, priority
        )
        task.created_step = self.time_step

        self.tasks.append(task)
        return True

    def process_robot_task(self, robot):
        task = robot.current_task

        if task is None:
            return

        if (
            not robot.carrying_package
            and robot.position == task.pickup
        ):
            robot.carrying_package = True
            task.status = "picked_up"

            robot.goal = task.delivery
            robot.path = []

            self.blocked_steps[robot.robot_id] = 0

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

            self.blocked_steps[robot.robot_id] = 0

    def plan_path(self, robot, blocked=None):
        self.astar_calls += 1
        self.replanning_events += 1

        path = astar(
            self.environment,
            robot.position,
            robot.goal,
            blocked_positions=blocked
        )

        if path is None:
            robot.path = []
            return False

        robot.path = path[1:]
        return True

    def step(self):
        self.time_step += 1

        occupied = {
            robot.position for robot in self.robots
        }

        self.environment.move_dynamic_obstacles(
            blocked_positions=occupied
        )

        allocate_tasks(
            self.robots,
            self.tasks,
            self.environment
        )

        count = len(self.robots)

        if count:
            offset = (self.time_step - 1) % count
            ordered = (
                self.robots[offset:]
                + self.robots[:offset]
            )
        else:
            ordered = []

        for robot in ordered:
            self.process_robot_task(robot)

            if robot.goal is None:
                continue

            robot_id = robot.robot_id

            blocked = {
                other.position
                for other in self.robots
                if other.robot_id != robot_id
            }

            # Plan a route if none is available
            if not robot.path:
                if not self.plan_path(robot):
                    self.robot_waiting[robot_id] += 1
                    continue

            if not robot.path:
                continue

            next_position = robot.path[0]
            row, col = next_position

            # Static or dynamic obstacle
            if self.environment.grid[row][col] != 0:
                self.blocked_steps[robot_id] += 1

                # Replan around the obstacle
                if not self.plan_path(robot, blocked):
                    self.robot_waiting[robot_id] += 1
                    continue

                if not robot.path:
                    continue

                next_position = robot.path[0]

            # Another robot is blocking the route
            if next_position in blocked:
                self.blocked_steps[robot_id] += 1

                if self.blocked_steps[robot_id] <= self.max_wait:
                    self.robot_waiting[robot_id] += 1
                    continue

                # Search for a route around other robots
                if not self.plan_path(robot, blocked):
                    self.robot_waiting[robot_id] += 1
                    continue

                if not robot.path:
                    continue

                next_position = robot.path[0]

            # Final collision check
            row, col = next_position

            if (
                next_position in blocked
                or self.environment.grid[row][col] != 0
            ):
                self.robot_waiting[robot_id] += 1
                continue

            previous = robot.position

            robot.position = next_position
            robot.path.pop(0)

            distance = (
                abs(previous[0] - next_position[0])
                + abs(previous[1] - next_position[1])
            )

            self.robot_distance[robot_id] += distance
            self.total_distance += distance

            self.blocked_steps[robot_id] = 0

            self.process_robot_task(robot)

        positions = [
            robot.position for robot in self.robots
        ]

        obstacles = set(
            self.environment.dynamic_obstacles
        )

        if (
            len(positions) != len(set(positions))
            or set(positions) & obstacles
        ):
            self.collision_count += 1
            raise RuntimeError("Collision detected")

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
            ],
            "total_distance": self.total_distance,
            "astar_calls": self.astar_calls
        })

    def run(self, steps=20):
        for _ in range(steps):
            self.step()

    def print_metrics(self):
        completed = sum(
            task.status == "completed"
            for task in self.tasks
        )

        finish_step = None

        if self.tasks:
            for state in self.history:
                if all(
                    task["status"] == "completed"
                    for task in state["tasks"]
                ) and len(state["tasks"]) == len(self.tasks):
                    finish_step = state["step"]
                    break

        print("\nSMART PLANNING METRICS")
        print("----------------------")
        print("Simulation steps:", self.time_step)
        print(
            "Completed deliveries:",
            f"{completed}/{len(self.tasks)}"
        )
        print("Total distance:", self.total_distance)
        print("A* calls (simulation):", self.astar_calls)
        print("Replanning events:", self.replanning_events)
        print("Recorded collisions:", self.collision_count)
        print("All deliveries finished at:", finish_step)

        for robot in self.robots:
            robot_id = robot.robot_id

            print(f"\nRobot {robot_id}")
            print(
                "Distance:",
                self.robot_distance[robot_id]
            )
            print(
                "Waiting steps:",
                self.robot_waiting[robot_id]
            )
            print(
                "Completed deliveries:",
                robot.completed_tasks
            )
