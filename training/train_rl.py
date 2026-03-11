from __future__ import annotations

import argparse
import csv
import os
from typing import List

import numpy as np
import torch
import torch.optim as optim

from training.rl_env import NeuroBotRLEnv
from training.policy import PolicyNetwork
from training.io import ensure_dir, save_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NeuroBot with REINFORCE.")
    parser.add_argument(
        "--episodes",
        type=int,
        required=True,
        help="Number of training episodes.",
    )
    parser.add_argument(
        "--episode-seconds",
        type=float,
        default=30.0,
        help="Target duration of each episode in seconds (default: 30).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor for returns (default: 0.99).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for the policy optimizer (default: 1e-3).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Save a policy checkpoint every N episodes (default: 50).",
    )
    return parser.parse_args()


def compute_returns(rewards: List[float], gamma: float) -> torch.Tensor:
    """Compute discounted returns G_t from a list of rewards."""
    returns: List[float] = []
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        returns.append(g)
    returns.reverse()
    returns_t = torch.tensor(returns, dtype=torch.float32)
    # Normalize for stability
    if returns_t.std() > 1e-8:
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
    return returns_t


def main() -> None:
    args = parse_args()

    # Reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    base_dir = os.path.dirname(__file__)
    runs_dir = os.path.join(base_dir, "..", "runs")
    runs_dir = os.path.normpath(runs_dir)
    ensure_dir(runs_dir)

    train_log_path = os.path.join(runs_dir, "train_log.csv")
    policies_dir = os.path.join(runs_dir, "policies")
    ensure_dir(policies_dir)

    # Prepare train_log.csv with header if needed
    if not os.path.exists(train_log_path) or os.path.getsize(train_log_path) == 0:
        with open(train_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "episode",
                    "return",
                    "loss",
                    "mean_reward",
                    "length_steps",
                    "best_return_so_far",
                ]
            )

    env = NeuroBotRLEnv(episode_seconds=args.episode_seconds)
    policy = PolicyNetwork()
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    best_return = float("-inf")

    for ep in range(1, args.episodes + 1):
        obs_np = env.reset()
        log_probs: List[torch.Tensor] = []
        rewards: List[float] = []

        done = False
        steps = 0

        while not done:
            obs_t = torch.from_numpy(obs_np.astype(np.float32))
            action_t, log_prob_t = policy.act(obs_t, deterministic=False)

            next_obs_np, reward, done, info = env.step(action_t.detach().numpy())

            log_probs.append(log_prob_t)
            rewards.append(float(reward))

            obs_np = next_obs_np
            steps += 1

        returns_t = compute_returns(rewards, args.gamma)
        log_probs_t = torch.stack(log_probs)

        loss = -(log_probs_t * returns_t).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_return = float(returns_t[0].item())
        mean_reward = float(np.mean(rewards)) if rewards else 0.0
        best_return = max(best_return, episode_return)

        print(
            f"Ep {ep}/{args.episodes} | "
            f"return={episode_return:.3f} | "
            f"mean_r={mean_reward:.3f} | "
            f"len={steps} | "
            f"loss={loss.item():.3f} | "
            f"best_return={best_return:.3f}"
        )

        # Append training metrics to CSV
        with open(train_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    ep,
                    episode_return,
                    float(loss.item()),
                    mean_reward,
                    steps,
                    best_return,
                ]
            )

        # Periodic checkpointing
        if ep % args.checkpoint_every == 0 or ep == args.episodes:
            ckpt_path = os.path.join(policies_dir, f"policy_ep{ep:04d}.pt")
            save_policy(policy, ckpt_path)


if __name__ == "__main__":
    main()

