
import random

from pathfinding import astar


class DynamicTaskGenerator:
    def __init__(
        self,
        simulation,
        seed=42,
        interval=10,
        tasks_per_interval=2,
        urgent_probability=0.25,
        max_new_tasks=20
    ):
        if interval < 1:
            raise ValueError("interval must be >= 1")

        if tasks_per_interval < 1:
            raise ValueError(
                "tasks_per_interval must be >= 1"
            )

        if not 0 <= urgent_probability <= 1:
            raise ValueError(
                "urgent_probability must be between 0 and 1"
            )

        if max_new_tasks < 0:
            raise ValueError(
                "max_new_tasks must be >= 0"
            )

        self.simulation = simulation
        self.rng = random.Random(seed)

        self.interval = interval
        self.tasks_per_interval = tasks_per_interval
        self.urgent_probability = urgent_probability
        self.max_new_tasks = max_new_tasks

        self.generated_tasks = 0
        self.events = []

        existing_ids = [
            task.task_id
            for task in simulation.tasks
            if isinstance(task.task_id, int)
        ]

        self.next_task_id = (
            max(existing_ids, default=0) + 1
        )

    def available_positions(self):
        environment = self.simulation.environment

        robot_positions = {
            robot.position
            for robot in self.simulation.robots
        }

        positions = []

        for row in range(environment.rows):
            for col in range(environment.cols):
                position = (row, col)

                if environment.grid[row][col] != 0:
                    continue

                if position in robot_positions:
                    continue

                positions.append(position)

        return positions

    def generate_task(self):
        if self.generated_tasks >= self.max_new_tasks:
            return None

        positions = self.available_positions()

        if len(positions) < 2:
            return None

        environment = self.simulation.environment

        # Limit random attempts to avoid endless loops.
        for _ in range(100):
            pickup, delivery = self.rng.sample(
                positions, 2
            )

            # Check whether a route exists when
            # the task is created.
            route = astar(
                environment,
                pickup,
                delivery
            )

            if route is None:
                continue

            priority = (
                3
                if self.rng.random()
                < self.urgent_probability
                else 1
            )

            # Avoid IDs already used by tasks.
            existing_ids = {
                task.task_id
                for task in self.simulation.tasks
            }

            while self.next_task_id in existing_ids:
                self.next_task_id += 1

            task_id = self.next_task_id

            success = self.simulation.add_task(
                task_id=task_id,
                pickup=pickup,
                delivery=delivery,
                priority=priority
            )

            if not success:
                continue

            self.next_task_id += 1
            self.generated_tasks += 1

            event = {
                "step": self.simulation.time_step,
                "task_id": task_id,
                "pickup": pickup,
                "delivery": delivery,
                "priority": priority
            }

            self.events.append(event)

            return event

        return None

    def update(self):
        """
        Call once after each simulation step.

        New tasks appear at configured intervals.
        """
        step = self.simulation.time_step

        if step == 0:
            return []

        if step % self.interval != 0:
            return []

        new_events = []

        for _ in range(self.tasks_per_interval):
            event = self.generate_task()

            if event is None:
                break

            new_events.append(event)

        return new_events
