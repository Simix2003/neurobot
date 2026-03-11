from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import math
import numpy as np
import pygame

from agents import Brain, Robot
from agents.heuristic_controller import compute_heuristic_action
from agents.sensors import SensorReadings, compute_sensor_readings
from config import (
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    WORLD_BORDER_COLOR,
    WORLD_BORDER_THICKNESS,
    FPS,
)
from world.food import Food, spawn_food_random


@dataclass
class Environment:
    """Minimal world container for Step 1A.

    For now it only represents the rectangular world area and exposes
    update/draw hooks that will host entities in later steps.
    """

    width: int = WINDOW_WIDTH
    height: int = WINDOW_HEIGHT
    robot: Robot = field(default_factory=Robot)
    food: Food = field(default_factory=spawn_food_random)
    score: int = 0

    brain: Brain = field(default_factory=Brain)
    control_mode: str = "brain"  # "brain", "manual", or "heuristic"

    sensors: SensorReadings | None = None

    # Pending inputs for manual mode
    _pending_forward: float = 0.0
    _pending_turn: float = 0.0

    # Episode state
    episode_index: int = 0
    steps_in_episode: int = 0
    time_in_episode: float = 0.0
    distance_traveled: float = 0.0
    food_collected_in_episode: int = 0

    # Internal tracking for distance computation
    _last_robot_x: float = field(default=WINDOW_WIDTH / 2.0)
    _last_robot_y: float = field(default=WINDOW_HEIGHT / 2.0)

    # Episode configuration
    max_episode_time_seconds: float = 60.0
    max_episode_steps: int = FPS * 60  # ~60 seconds at default FPS
    food_per_episode_target: int = 0  # 0 disables this condition

    # Reward shaping parameters
    food_reward: float = 10.0
    distance_penalty: float = 0.01

    def __post_init__(self) -> None:
        # Start the first episode
        self.reset_episode()

    def handle_manual_controls(self, forward: float, turn: float) -> None:
        """Store current frame's normalized control inputs."""
        self._pending_forward = max(-1.0, min(1.0, forward))
        self._pending_turn = max(-1.0, min(1.0, turn))

    def reset_episode(self) -> None:
        """Reset episode-specific state and respawn entities."""
        self.episode_index += 1

        # Reset robot and food
        self.robot = Robot()
        self.food = spawn_food_random()
        self.score = 0

        # Reset episode metrics
        self.steps_in_episode = 0
        self.time_in_episode = 0.0
        self.distance_traveled = 0.0
        self.food_collected_in_episode = 0

        # Reset helpers
        self._last_robot_x = self.robot.x
        self._last_robot_y = self.robot.y
        self.sensors = None
        self._pending_forward = 0.0
        self._pending_turn = 0.0

    def update(self, dt: float) -> Tuple[bool, Dict[str, float | int | str] | None]:
        """Advance world simulation by dt seconds.

        Returns:
            done: True if the current episode has terminated.
            metrics: Episode metrics dict when done is True, else None.
        """
        if dt <= 0.0:
            return False, None

        # Decide control inputs based on current mode
        if self.control_mode == "brain":
            # Compute sensors and query brain for movement
            self.sensors = compute_sensor_readings(self.robot, self.food)
            inputs = np.array(
                [
                    self.sensors.food_distance,
                    self.sensors.food_angle,
                    self.sensors.wall_front,
                    self.sensors.wall_left,
                    self.sensors.wall_right,
                ],
                dtype=float,
            )
            forward, turn = self.brain.forward(inputs)
        elif self.control_mode == "heuristic":
            # Heuristic controller uses the same sensor API
            self.sensors = compute_sensor_readings(self.robot, self.food)
            forward, turn = compute_heuristic_action(self.sensors)
        else:
            # Manual mode (Step 1B behavior)
            forward = self._pending_forward
            turn = self._pending_turn

        # Apply movement
        prev_x, prev_y = self.robot.x, self.robot.y
        self.robot.apply_manual_input(forward, turn, dt)

        # Reset manual controls so we do not accumulate stale input
        self._pending_forward = 0.0
        self._pending_turn = 0.0

        # Compute per-step distance
        step_distance = math.hypot(self.robot.x - prev_x, self.robot.y - prev_y)

        # Collision check between robot and food
        prev_score = self.score
        self._check_robot_food_collision()

        # Update episode aggregates
        self.steps_in_episode += 1
        self.time_in_episode += dt
        self.distance_traveled += step_distance
        score_delta = self.score - prev_score
        if score_delta > 0:
            self.food_collected_in_episode += score_delta

        # Refresh sensors after movement for manual mode so HUD stays informative
        if self.control_mode == "manual":
            self.sensors = compute_sensor_readings(self.robot, self.food)

        done, metrics = self._check_episode_done()
        if done:
            # Prepare metrics before resetting
            episode_metrics = self._build_episode_metrics()
            self.reset_episode()
            return True, episode_metrics

        return False, metrics

    def _check_robot_food_collision(self) -> None:
        dx = self.robot.x - self.food.x
        dy = self.robot.y - self.food.y
        min_distance = self.robot.radius + self.food.radius
        if dx * dx + dy * dy <= min_distance * min_distance:
            self.score += 1
            self.food = spawn_food_random()

    def _check_episode_done(self) -> Tuple[bool, Dict[str, float | int | str] | None]:
        """Check whether the current episode should terminate."""
        time_limit_reached = self.time_in_episode >= self.max_episode_time_seconds
        step_limit_reached = (
            self.max_episode_steps > 0
            and self.steps_in_episode >= self.max_episode_steps
        )
        food_target_reached = (
            self.food_per_episode_target > 0
            and self.food_collected_in_episode >= self.food_per_episode_target
        )

        if time_limit_reached or step_limit_reached or food_target_reached:
            return True, self._build_episode_metrics()

        return False, None

    def _build_episode_metrics(self) -> Dict[str, float | int | str]:
        """Assemble a simple metrics dict for logging/training."""
        reward = (
            self.food_reward * float(self.food_collected_in_episode)
            - self.distance_penalty * float(self.distance_traveled)
        )
        return {
            "episode": self.episode_index,
            "mode": self.control_mode,
            "food_collected": int(self.food_collected_in_episode),
            "distance_traveled": float(self.distance_traveled),
            "episode_length_seconds": float(self.time_in_episode),
            "steps": int(self.steps_in_episode),
            "reward": float(reward),
        }

    def draw(self, surface: pygame.Surface) -> None:
        """Draw world-level visuals (currently just the border)."""
        rect = pygame.Rect(0, 0, self.width, self.height)
        pygame.draw.rect(
            surface,
            WORLD_BORDER_COLOR,
            rect,
            WORLD_BORDER_THICKNESS,
        )
        self.food.draw(surface)
        self.robot.draw(surface)

