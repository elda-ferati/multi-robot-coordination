
from collections import deque

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

        # Deadlock resolution metrics
        self.yield_events = 0
        self.yield_moves = 0
        self.failed_yield_attempts = 0

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

    def is_free_cell(self, position):
        row, col = position

        return (
            self.environment.is_valid_position(row, col)
            and self.environment.grid[row][col] == 0
        )

    def neighbors(self, position):
        row, col = position

        for dr, dc in [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1)
        ]:
            candidate = (row + dr, col + dc)

            if self.is_free_cell(candidate):
                yield candidate

    def move_robot(self, robot, destination, moved_ids):
        """
        Move one robot by one grid cell.

        Every robot may move at most once per simulation
        step. This also applies to yielding movements.
        """
        robot_id = robot.robot_id

        if robot_id in moved_ids:
            return False

        if not self.is_free_cell(destination):
            return False

        if any(
            other.position == destination
            for other in self.robots
            if other.robot_id != robot_id
        ):
            return False

        distance = (
            abs(robot.position[0] - destination[0])
            + abs(robot.position[1] - destination[1])
        )

        if distance != 1:
            return False

        robot.position = destination

        self.robot_distance[robot_id] += distance
        self.total_distance += distance

        self.blocked_steps[robot_id] = 0
        moved_ids.add(robot_id)

        return True

    def find_yield_chain(
        self, blocker, requester, moved_ids
    ):
        """
        Find a chain of occupied cells leading from
        the blocking robot to an empty cell.

        The requester stays in its current position.

        Returns positions from blocker to empty cell.
        """
        occupied = {
            robot.position: robot
            for robot in self.robots
        }

        start = blocker.position
        forbidden = requester.position

        queue = deque([start])
        parent = {start: None}

        while queue:
            current = queue.popleft()

            for neighbor in self.neighbors(current):
                if neighbor in parent:
                    continue

                if neighbor == forbidden:
                    continue

                occupying_robot = occupied.get(neighbor)

                if (
                    occupying_robot is not None
                    and occupying_robot.robot_id in moved_ids
                ):
                    continue

                parent[neighbor] = current

                if occupying_robot is None:
                    chain = []
                    position = neighbor

                    while position is not None:
                        chain.append(position)
                        position = parent[position]

                    chain.reverse()
                    return chain

                queue.append(neighbor)

        return None

    def resolve_blockage(
        self, requester, blocker, moved_ids
    ):
        """
        Move the blocker, or a chain of blockers,
        toward an available empty cell.

        The entire chain is planned before any robot
        is moved. Moves are then executed backwards.
        """
        if blocker.robot_id in moved_ids:
            return False

        chain = self.find_yield_chain(
            blocker, requester, moved_ids
        )

        if chain is None or len(chain) < 2:
            self.failed_yield_attempts += 1
            return False

        occupied = {
            robot.position: robot
            for robot in self.robots
        }

        # Validate the full chain before moving.
        robots_to_move = []

        for position in chain[:-1]:
            robot = occupied.get(position)

            if (
                robot is None
                or robot.robot_id in moved_ids
            ):
                self.failed_yield_attempts += 1
                return False

            robots_to_move.append(robot)

        # Execute from the empty cell backwards.
        for index in range(len(robots_to_move) - 1, -1, -1):
            robot = robots_to_move[index]
            destination = chain[index + 1]

            if not self.move_robot(
                robot, destination, moved_ids
            ):
                raise RuntimeError(
                    "Unexpected yielding movement failure"
                )

            # A displaced robot must replan later.
            robot.path = []

            self.yield_moves += 1

            # Process pickup/delivery if the robot
            # was displaced onto its task location.
            self.process_robot_task(robot)

        self.yield_events += 1
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

        moved_ids = set()

        for robot in ordered:
            self.process_robot_task(robot)

            if robot.goal is None:
                continue

            robot_id = robot.robot_id

            # Robot already moved during yielding.
            if robot_id in moved_ids:
                continue

            blocked = {
                other.position
                for other in self.robots
                if other.robot_id != robot_id
            }

            # Initial path planning
            if not robot.path:
                if not self.plan_path(robot):
                    self.robot_waiting[robot_id] += 1
                    continue

            if not robot.path:
                continue

            next_position = robot.path[0]

            # Replan when a static/dynamic obstacle
            # blocks the next movement.
            if not self.is_free_cell(next_position):
                self.blocked_steps[robot_id] += 1

                if not self.plan_path(robot, blocked):
                    self.robot_waiting[robot_id] += 1
                    continue

                if not robot.path:
                    continue

                next_position = robot.path[0]

            # Another robot blocks the next cell.
            if next_position in blocked:
                self.blocked_steps[robot_id] += 1

                if self.blocked_steps[robot_id] <= self.max_wait:
                    self.robot_waiting[robot_id] += 1
                    continue

                # First try a normal alternative route.
                alternative = self.plan_path(
                    robot, blocked
                )

                if alternative and robot.path:
                    next_position = robot.path[0]

                else:
                    # No alternative route:
                    # ask the blocking robot to yield.
                    unrestricted = astar(
                        self.environment,
                        robot.position,
                        robot.goal
                    )

                    if unrestricted is None or len(unrestricted) < 2:
                        self.robot_waiting[robot_id] += 1
                        continue

                    next_position = unrestricted[1]

                    blocker = next(
                        (
                            other for other in self.robots
                            if other.position == next_position
                            and other.robot_id != robot_id
                        ),
                        None
                    )

                    if blocker is not None:
                        resolved = self.resolve_blockage(
                            robot,
                            blocker,
                            moved_ids
                        )

                        if not resolved:
                            self.robot_waiting[robot_id] += 1
                            continue

                    # The requester can now use the
                    # first step of the unrestricted path.
                    robot.path = unrestricted[1:]
                    next_position = robot.path[0]

            if not robot.path:
                continue

            # Final collision prevention.
            if not self.is_free_cell(next_position):
                self.robot_waiting[robot_id] += 1
                robot.path = []
                continue

            if any(
                other.position == next_position
                for other in self.robots
                if other.robot_id != robot_id
            ):
                self.robot_waiting[robot_id] += 1
                robot.path = []
                continue

            moved = self.move_robot(
                robot, next_position, moved_ids
            )

            if not moved:
                self.robot_waiting[robot_id] += 1
                continue

            robot.path.pop(0)

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
            "astar_calls": self.astar_calls,
            "yield_events": self.yield_events,
            "yield_moves": self.yield_moves
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
                if (
                    len(state["tasks"]) == len(self.tasks)
                    and all(
                        task["status"] == "completed"
                        for task in state["tasks"]
                    )
                ):
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

        print("\nDEADLOCK RESOLUTION METRICS")
        print("---------------------------")
        print("Yield events:", self.yield_events)
        print("Yield moves:", self.yield_moves)
        print(
            "Failed yield attempts:",
            self.failed_yield_attempts
        )

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
