
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

        # Performance metrics
        self.astar_calls = 0
        self.replanning_events = 0
        self.total_distance = 0
        self.collision_count = 0

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

    def add_task(
        self,
        task_id,
        pickup,
        delivery,
        priority=1
    ):
        if any(t.task_id == task_id for t in self.tasks):
            return False

        if priority < 1:
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

    def plan_path(self, robot, blocked):
        self.astar_calls += 1

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
            robot.position
            for robot in self.robots
        }

        # Move dynamic obstacles safely
        self.environment.move_dynamic_obstacles(
            blocked_positions=occupied
        )

        # Allocate pending tasks
        allocate_tasks(
            self.robots,
            self.tasks,
            self.environment
        )

        # Rotate robot priority
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

            # Check whether current path is usable
            needs_replanning = False

            if not robot.path:
                needs_replanning = True

            else:
                next_position = robot.path[0]
                row, col = next_position

                if (
                    self.environment.grid[row][col] != 0
                    or next_position in blocked
                ):
                    needs_replanning = True

            if needs_replanning:
                self.replanning_events += 1

                if not self.plan_path(robot, blocked):
                    continue

            if not robot.path:
                continue

            next_position = robot.path[0]

            # Final safety check
            if (
                next_position in blocked
                or self.environment.grid[
                    next_position[0]
                ][next_position[1]] != 0
            ):
                continue

            old_position = robot.position
            robot.position = next_position
            robot.path.pop(0)

            movement = (
                abs(next_position[0] - old_position[0])
                + abs(next_position[1] - old_position[1])
            )

            self.total_distance += movement

            self.process_robot_task(robot)

        # Safety validation
        positions = [r.position for r in self.robots]
        dynamic_positions = self.environment.dynamic_obstacles

        if (
            len(positions) != len(set(positions))
            or set(positions) & set(dynamic_positions)
        ):
            self.collision_count += 1
            raise RuntimeError(
                "Collision detected in simulation"
            )

        # Save history
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
            t.status == "completed"
            for t in self.tasks
        )

        print("\nPERFORMANCE METRICS")
        print("-------------------")
        print("Simulation steps:", self.time_step)
        print("Completed deliveries:", completed)
        print("Total distance:", self.total_distance)
        print("A* calls:", self.astar_calls)
        print("Replanning events:", self.replanning_events)
        print("Recorded collisions:", self.collision_count)
