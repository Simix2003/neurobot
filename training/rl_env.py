from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from config import FPS, WINDOW_WIDTH, WINDOW_HEIGHT
from world.environment import Environment
from world.food import Food
from agents.sensors import compute_sensor_readings
from .rewards import compute_step_reward


class NeuroBotRLEnv:
    """Gym-like wrapper around the NeuroBot Environment for RL training.

    This wrapper exposes:
      - reset() -> obs
      - step(action) -> (obs, reward, done, info)

    Observations are the 5 sensor values:
      [food_distance, food_angle, wall_front, wall_left, wall_right]

    Actions are 2D continuous vectors in [-1, 1]^2:
      [forward_value, turn_value]
    """

    def __init__(
        self,
        episode_seconds: float = 30.0,
        dt: float | None = None,
        easy_mode: bool = True,
        easy_radius: float | None = None,
    ) -> None:
        self._dt: float = dt if dt is not None else 1.0 / float(FPS)

        # Underlying simulation environment
        self._env = Environment()
        self._env.control_mode = "manual"  # actions come from RL policy

        # Configure episode duration based on requested length
        self._env.max_episode_time_seconds = episode_seconds
        self._env.max_episode_steps = int(episode_seconds * FPS)

        # Keep episodes time-based/step-based for now
        self._env.food_per_episode_target = 0

        # Training curriculum / easier setup flags
        self._easy_mode: bool = easy_mode
        # Default radius: quarter of the smaller world dimension
        default_radius = min(WINDOW_WIDTH, WINDOW_HEIGHT) / 4.0
        self._easy_radius: float = easy_radius if easy_radius is not None else default_radius

        # Cached quantities for reward shaping
        self._last_food_distance: float = 0.0
        self._last_distance_traveled: float = 0.0
        self._last_score: int = 0

    @property
    def env(self) -> Environment:
        """Access to the underlying Environment (for rendering or inspection)."""
        return self._env

    def reset(self) -> np.ndarray:
        """Reset the episode and return the initial observation."""
        self._env.reset_episode()
        self._env.control_mode = "manual"

        # In easy mode, respawn food closer to the robot to make early learning easier
        if self._easy_mode:
            cx, cy = self._env.robot.x, self._env.robot.y
            angle = float(np.random.uniform(-np.pi, np.pi))
            radius = float(np.random.uniform(0.0, self._easy_radius))
            x = cx + np.cos(angle) * radius
            y = cy + np.sin(angle) * radius
            margin = 20.0
            x = float(np.clip(x, margin, WINDOW_WIDTH - margin))
            y = float(np.clip(y, margin, WINDOW_HEIGHT - margin))
            self._env.food = Food(x=x, y=y)

        sensors = compute_sensor_readings(self._env.robot, self._env.food)
        self._env.sensors = sensors

        self._last_food_distance = sensors.food_distance
        self._last_distance_traveled = self._env.distance_traveled
        self._last_score = self._env.score

        return self._obs_from_sensors(sensors)

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Apply an action and advance the simulation by one step."""
        if action.shape != (2,):
            raise ValueError(f"Expected action shape (2,), got {action.shape}")

        # Clip action into the valid control range
        forward = float(np.clip(action[0], -1.0, 1.0))
        turn = float(np.clip(action[1], -1.0, 1.0))

        prev_food_distance = self._last_food_distance
        prev_distance_traveled = self._env.distance_traveled
        prev_score = self._env.score

        # Drive the environment via the manual-control path
        self._env.handle_manual_controls(forward, turn)
        done, episode_metrics = self._env.update(self._dt)

        # Sensors after the step; ensure they are available
        sensors = self._env.sensors or compute_sensor_readings(
            self._env.robot, self._env.food
        )
        self._env.sensors = sensors
        obs = self._obs_from_sensors(sensors)

        # Reward shaping based on movement and food progress
        step_distance = self._env.distance_traveled - prev_distance_traveled
        food_delta = self._env.score - prev_score
        reward = compute_step_reward(
            prev_food_distance=prev_food_distance,
            new_food_distance=sensors.food_distance,
            step_distance=step_distance,
            food_eaten=int(food_delta),
            wall_front=sensors.wall_front,
            wall_left=sensors.wall_left,
            wall_right=sensors.wall_right,
        )

        # Update cached values for the next step
        self._last_food_distance = sensors.food_distance
        self._last_distance_traveled = self._env.distance_traveled
        self._last_score = self._env.score

        info: Dict[str, Any] = {}
        if episode_metrics is not None:
            info["episode_metrics"] = episode_metrics

        return obs, float(reward), bool(done), info

    @staticmethod
    def _obs_from_sensors(sensors) -> np.ndarray:
        return np.asarray(
            [
                sensors.food_distance,
                sensors.food_angle,
                sensors.wall_front,
                sensors.wall_left,
                sensors.wall_right,
            ],
            dtype=np.float32,
        )

