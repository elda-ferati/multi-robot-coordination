
import simulation
from marl_features import get_robot_task_features


def no_automatic_allocation(robots, tasks, environment):
    pass


class MARLEnvironment:
    def __init__(self, sim, max_steps=300, max_candidates=8):
        self.sim = sim
        self.max_steps = max_steps
        self.max_candidates = max_candidates

    def get_observations(self):
        """
        Return candidate tasks and their features
        for each available robot.
        """
        observations = {}

        pending = sorted(
            [
                t for t in self.sim.tasks
                if t.status == "pending"
            ],
            key=lambda t: (-t.priority, t.task_id)
        )

        for robot in self.sim.robots:
            available = (
                robot.current_task is None
                and robot.goal is None
            )

            candidates = (
                pending[:self.max_candidates]
                if available else []
            )

            observations[robot.robot_id] = [
                {
                    "task_id": task.task_id,
                    "features": get_robot_task_features(
                        robot,
                        task,
                        self.sim.robots,
                        self.sim.tasks,
                        self.sim.environment
                    )
                }
                for task in candidates
            ]

        return observations

    def step(self, actions):
        """
        actions = {robot_id: task_id or None}

        One decision round followed by one
        simulation movement step.
        """

        before_distance = self.sim.total_distance

        before_waiting = sum(
            self.sim.robot_waiting.values()
        )

        completed_before = {
            task.task_id
            for task in self.sim.tasks
            if task.status == "completed"
        }

        robots_by_id = {
            robot.robot_id: robot
            for robot in self.sim.robots
        }

        tasks_by_id = {
            task.task_id: task
            for task in self.sim.tasks
        }

        observations = self.get_observations()

        reward = -0.05
        assignments = 0

        for robot_id in sorted(actions):
            task_id = actions[robot_id]

            if task_id is None:
                continue

            if robot_id not in robots_by_id:
                raise ValueError("Unknown robot ID")

            allowed = {
                candidate["task_id"]
                for candidate in observations[robot_id]
            }

            if task_id not in allowed:
                raise ValueError(
                    f"Invalid action: robot={robot_id}, "
                    f"task={task_id}"
                )

            robot = robots_by_id[robot_id]
            task = tasks_by_id[task_id]

            if task.status != "pending":
                continue

            if robot.assign_task(
                task, self.sim.environment
            ):
                assignments += 1

        # Disable the simulation's default allocator
        # for this step only.
        original = simulation.allocate_tasks

        try:
            simulation.allocate_tasks = (
                no_automatic_allocation
            )
            self.sim.step()
        finally:
            simulation.allocate_tasks = original

        distance_delta = (
            self.sim.total_distance - before_distance
        )

        waiting_delta = (
            sum(self.sim.robot_waiting.values())
            - before_waiting
        )

        newly_completed = [
            task for task in self.sim.tasks
            if task.status == "completed"
            and task.task_id not in completed_before
        ]

        # Shared team reward
        reward += 0.2 * assignments
        reward -= 0.01 * distance_delta
        reward -= 0.02 * max(0, waiting_delta)

        for task in newly_completed:
            reward += 20.0 if task.priority >= 3 else 10.0

        all_completed = (
            bool(self.sim.tasks)
            and all(
                task.status == "completed"
                for task in self.sim.tasks
            )
        )

        time_limit = (
            self.sim.time_step >= self.max_steps
        )

        done = all_completed or time_limit

        info = {
            "time_step": self.sim.time_step,
            "completed": sum(
                task.status == "completed"
                for task in self.sim.tasks
            ),
            "total_tasks": len(self.sim.tasks),
            "distance": self.sim.total_distance,
            "collisions": self.sim.collision_count,
            "success": all_completed,
        }

        return (
            self.get_observations(),
            reward,
            done,
            info
        )
