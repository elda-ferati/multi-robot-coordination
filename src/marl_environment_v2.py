
from marl_environment import MARLEnvironment
from marl_features_v2 import get_team_features


class MARLEnvironmentV2(MARLEnvironment):
    """
    Team-aware observations and shared team reward.
    """

    def get_observations(self):
        observations = {}

        pending = sorted(
            [
                task for task in self.sim.tasks
                if task.status == "pending"
            ],
            key=lambda task: (
                -task.priority,
                task.task_id
            )
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
                    "features": get_team_features(
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
        Execute team actions and compute a shared reward.
        """

        before_completed = {
            task.task_id
            for task in self.sim.tasks
            if task.status == "completed"
        }

        before_distance = self.sim.total_distance
        before_collisions = self.sim.collision_count
        before_waiting = sum(
            self.sim.robot_waiting.values()
        )

        observations, _, done, info = super().step(
            actions
        )

        distance_delta = max(
            0,
            self.sim.total_distance - before_distance
        )

        collision_delta = max(
            0,
            self.sim.collision_count - before_collisions
        )

        waiting_delta = max(
            0,
            sum(self.sim.robot_waiting.values())
            - before_waiting
        )

        newly_completed = [
            task for task in self.sim.tasks
            if task.status == "completed"
            and task.task_id not in before_completed
        ]

        incomplete = [
            task for task in self.sim.tasks
            if task.status != "completed"
        ]

        # Shared reward for the whole robot team.
        reward = -0.10

        # Minimize total movement and waiting.
        reward -= 0.03 * distance_delta
        reward -= 0.05 * waiting_delta

        # Discourage leaving tasks unfinished.
        for task in incomplete:
            reward -= (
                0.03 if task.priority >= 3
                else 0.01
            )

        # Reward deliveries, with urgent priority.
        for task in newly_completed:
            if task.priority >= 3:
                reward += 24.0
            else:
                reward += 12.0

        # Strong penalty for recorded collisions.
        reward -= 20.0 * collision_delta

        return observations, reward, done, info
