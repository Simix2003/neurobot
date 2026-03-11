import pygame

from config import WINDOW_WIDTH, WINDOW_HEIGHT, FPS
from world.environment import Environment
from render.renderer import Renderer


def main() -> None:
    pygame.init()
    pygame.display.set_caption("NeuroBot - Simulation Sandbox (Steps 1-2)")

    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    env = Environment(WINDOW_WIDTH, WINDOW_HEIGHT)
    renderer = Renderer(screen, env)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                # Toggle between brain and manual control
                if event.key == pygame.K_m:
                    env.control_mode = "manual" if env.control_mode == "brain" else "brain"

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
        env.update(dt)
        renderer.render_frame()

    pygame.quit()


if __name__ == "__main__":
    main()

