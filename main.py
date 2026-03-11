import os
import csv

import pygame

from config import WINDOW_WIDTH, WINDOW_HEIGHT, FPS
from world.environment import Environment
from render.renderer import Renderer


def main() -> None:
    pygame.init()
    pygame.display.set_caption("NeuroBot - Simulation Sandbox (Steps 1-2)")

    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    # Prepare runs/ directory and episode log file
    base_dir = os.path.dirname(__file__)
    runs_dir = os.path.join(base_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    episodes_log_path = os.path.join(runs_dir, "episodes.csv")

    # Ensure CSV has a header
    if not os.path.exists(episodes_log_path) or os.path.getsize(episodes_log_path) == 0:
        with open(episodes_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "episode",
                    "mode",
                    "food_collected",
                    "distance_traveled",
                    "episode_length_seconds",
                    "steps",
                    "reward",
                ]
            )

    env = Environment(WINDOW_WIDTH, WINDOW_HEIGHT)
    renderer = Renderer(screen, env)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                # Explicitly select control modes
                if event.key == pygame.K_m:
                    env.control_mode = "manual"
                elif event.key == pygame.K_b:
                    env.control_mode = "brain"
                elif event.key == pygame.K_h:
                    env.control_mode = "heuristic"

        # Manual keyboard controls are only applied in manual mode
        if env.control_mode == "manual":
            keys = pygame.key.get_pressed()
            forward = 0.0
            turn = 0.0

            if keys[pygame.K_UP] or keys[pygame.K_w]:
                forward += 1.0
            if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                forward -= 1.0
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                turn -= 1.0
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                turn += 1.0

            env.handle_manual_controls(forward, turn)

        done, metrics = env.update(dt)

        if done and metrics is not None:
            # Console summary
            print(
                "Episode {episode} | mode={mode} | food={food_collected} | "
                "dist={distance_traveled:.1f} | length={episode_length_seconds:.1f}s | "
                "steps={steps} | reward={reward:.2f}".format(**metrics)
            )

            # Append to CSV log
            with open(episodes_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        metrics["episode"],
                        metrics["mode"],
                        metrics["food_collected"],
                        metrics["distance_traveled"],
                        metrics["episode_length_seconds"],
                        metrics["steps"],
                        metrics["reward"],
                    ]
                )

        renderer.render_frame()

    pygame.quit()


if __name__ == "__main__":
    main()

