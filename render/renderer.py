from __future__ import annotations

import math

import pygame

from config import (
    BACKGROUND_COLOR,
    SCORE_TEXT_COLOR,
    SENSOR_RAY_COLOR_FOOD,
    SENSOR_RAY_COLOR_WALL,
)
from world.environment import Environment


class Renderer:
    """Minimal renderer for Step 1A.

    Responsible for clearing the screen, asking the environment to draw itself,
    and flipping the display buffer.
    """

    def __init__(self, screen: pygame.Surface, environment: Environment) -> None:
        self._screen = screen
        self._environment = environment
        self._font: pygame.font.Font | None = None

    def render_frame(self) -> None:
        """Render a single frame of the world."""
        self._screen.fill(BACKGROUND_COLOR)
        self._environment.draw(self._screen)
        self._draw_hud()
        pygame.display.flip()

    def _draw_hud(self) -> None:
        if self._font is None:
            self._font = pygame.font.SysFont("consolas", 18)

        # Score
        score_text = f"Score: {self._environment.score}"
        surf = self._font.render(score_text, True, SCORE_TEXT_COLOR)
        self._screen.blit(surf, (10, 10))

        # Control mode
        mode_text = f"Mode: {self._environment.control_mode.upper()} (press M to toggle)"
        surf_mode = self._font.render(mode_text, True, SCORE_TEXT_COLOR)
        self._screen.blit(surf_mode, (10, 32))

        # Sensor values (if available)
        sensors = getattr(self._environment, "sensors", None)
        if sensors is not None:
            sensor_text = (
                f"dF={sensors.food_distance:.2f} "
                f"aF={sensors.food_angle:.2f} "
                f"wF={sensors.wall_front:.2f} "
                f"wL={sensors.wall_left:.2f} "
                f"wR={sensors.wall_right:.2f}"
            )
            surf_sensors = self._font.render(sensor_text, True, SCORE_TEXT_COLOR)
            self._screen.blit(surf_sensors, (10, 54))

        self._draw_sensor_rays()

    def _draw_sensor_rays(self) -> None:
        sensors = getattr(self._environment, "sensors", None)
        robot = self._environment.robot
        if sensors is None:
            return

        cx, cy = int(robot.x), int(robot.y)

        # Food direction ray
        # Reconstruct approximate absolute angle to food from relative angle
        absolute_angle = robot.angle + sensors.food_angle * math.pi
        length_food = 150
        fx = cx + math.cos(absolute_angle) * length_food
        fy = cy + math.sin(absolute_angle) * length_food
        pygame.draw.line(
            self._screen,
            SENSOR_RAY_COLOR_FOOD,
            (cx, cy),
            (int(fx), int(fy)),
            1,
        )

        # Wall direction rays (front, left, right) with lengths based on normalized distance
        for offset, value in (
            (0.0, sensors.wall_front),
            (math.pi / 2.0, sensors.wall_left),
            (-math.pi / 2.0, sensors.wall_right),
        ):
            angle = robot.angle + offset
            # Invert so that smaller distance -> shorter line, purely visual
            length = 200 * (1.0 - value)
            wx = cx + math.cos(angle) * length
            wy = cy + math.sin(angle) * length
            pygame.draw.line(
                self._screen,
                SENSOR_RAY_COLOR_WALL,
                (cx, cy),
                (int(wx), int(wy)),
                1,
            )

