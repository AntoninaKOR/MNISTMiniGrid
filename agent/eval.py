"""Evaluate a trained PPO policy: plot learning curves and record a GIF.

Reads a policy checkpoint produced by :mod:`agent.train` (which also stores
the layout used during training and a per-rollout metrics CSV next to it),
re-builds the same env, rolls the policy out greedily or stochastically, and
writes:

* an MP-style learning-curves figure (PNG),
* a GIF showing the trained agent in action.

Example:
    python -m agent.eval --policy checkpoints/policy_10x10.pt
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.distributions import Categorical

from agent.mnist_classifier import load_classifier
from agent.policy import GRUPolicy
from env import MNISTMazeVecEnv, load_mnist_by_class


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", type=Path, required=True, help="Path to policy checkpoint (*.pt).")
    p.add_argument("--gif", type=Path, default=None, help="Output GIF path (default: alongside checkpoint).")
    p.add_argument("--plot", type=Path, default=None, help="Output PNG path for learning curves.")
    p.add_argument("--metrics", type=Path, default=None, help="Override metrics CSV path.")
    p.add_argument("--num-episodes", type=int, default=3, help="Episodes to record in the GIF.")
    p.add_argument("--max-frames", type=int, default=400, help="Hard cap on total GIF frames.")
    p.add_argument("--deterministic", action="store_true", help="Greedy argmax actions instead of sampling.")
    p.add_argument("--cell-size", type=int, default=28)
    p.add_argument("--fps", type=float, default=3.0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--mnist-cache", type=Path, default=Path(".mnist_cache"))
    p.add_argument("--seed", type=int, default=12345, help="Seed for evaluation rollouts.")
    return p.parse_args()


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def plot_learning_curves(metrics_path: Path, out_path: Path) -> None:
    with metrics_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"No rows in {metrics_path}; skipping plot.")
        return

    steps = np.array([float(r["env_steps"]) for r in rows])

    def col(name: str) -> np.ndarray:
        return np.array([float(r[name]) if r[name] not in ("", "nan") else np.nan for r in rows])

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    (ax_ret, ax_len), (ax_succ, ax_ent) = axes

    ax_ret.plot(steps, col("ep_return_mean"))
    ax_ret.set_title("Mean episode return")
    ax_ret.set_ylabel("return")

    ax_len.plot(steps, col("ep_length_mean"))
    ax_len.set_title("Mean episode length")
    ax_len.set_ylabel("steps")

    ax_succ.plot(steps, col("ep_success_rate"))
    ax_succ.set_title("Success rate (reached goal)")
    ax_succ.set_ylabel("share")
    ax_succ.set_xlabel("env steps")

    ax_ent.plot(steps, col("entropy"), label="policy entropy")
    ax_ent.plot(steps, col("approx_kl"), label="approx KL")
    ax_ent.set_title("Policy entropy & KL")
    ax_ent.set_xlabel("env steps")
    ax_ent.legend(loc="best")

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Learning curves: {metrics_path.name}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved learning curves to {out_path}")


def record_gif(
    env: MNISTMazeVecEnv,
    classifier,
    policy: GRUPolicy,
    *,
    num_episodes: int,
    max_frames: int,
    cell_size: int,
    fps: float,
    deterministic: bool,
    device: torch.device,
    seed: int,
    out_path: Path,
) -> None:
    obs, _ = env.reset(seed=seed)
    hidden = policy.initial_hidden(env.num_envs, device)
    episode_start = torch.ones(env.num_envs, dtype=torch.bool, device=device)

    frames: list[Image.Image] = [Image.fromarray(env.render_frame(env_idx=0, cell_size=cell_size))]
    finished = 0
    while finished < num_episodes and len(frames) < max_frames:
        with torch.no_grad():
            hidden = hidden * (~episode_start).float().unsqueeze(-1)
            image_t = torch.from_numpy(obs["image"]).to(device)
            goal_t = torch.from_numpy(obs["goal"]).to(device)
            digit = classifier.predict(image_t)
            logits, _, hidden = policy.step(digit, goal_t, hidden)
            if deterministic:
                action = logits.argmax(-1)
            else:
                action = Categorical(logits=logits).sample()

        obs, _, terminated, truncated, _ = env.step(action.cpu().numpy())
        done = np.logical_or(terminated, truncated)
        episode_start = torch.from_numpy(done).to(device)
        if done[0]:
            finished += 1
        frames.append(Image.fromarray(env.render_frame(env_idx=0, cell_size=cell_size)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(round(1000.0 / fps))
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )
    print(f"Saved GIF ({len(frames)} frames, {finished}/{num_episodes} episodes) to {out_path}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    ckpt = load_checkpoint(args.policy)
    train_args = ckpt["args"]
    obstacles = ckpt["obstacle_mask"]
    colors = ckpt["color_map"]
    height = int(train_args["size"])
    width = int(train_args["size"])
    n_colors = int(train_args["n_colors"])
    max_episode_steps = int(train_args["max_episode_steps"] or 4 * height)

    classifier = load_classifier(train_args["mnist_checkpoint"], device=device)
    policy = GRUPolicy(
        n_digits=10,
        goal_dim=height + width,
        n_actions=4,
        hidden_dim=int(train_args["hidden_dim"]),
    ).to(device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    mnist_banks = load_mnist_by_class(cache_dir=args.mnist_cache)
    env = MNISTMazeVecEnv(
        num_envs=1,
        height=height,
        width=width,
        obstacle_mask=obstacles,
        color_map=colors,
        n_colors=n_colors,
        max_steps=max_episode_steps,
        mnist_images_by_class=mnist_banks,
        seed=args.seed,
    )

    # --- Learning curves ---
    metrics_path = args.metrics or args.policy.with_name(args.policy.stem + "_metrics.csv")
    plot_path = args.plot or args.policy.with_name(args.policy.stem + "_curves.png")
    if metrics_path.exists():
        plot_learning_curves(metrics_path, plot_path)
    else:
        print(f"No metrics CSV at {metrics_path}; skipping learning-curve plot.")

    # --- GIF ---
    gif_path = args.gif or args.policy.with_name(args.policy.stem + "_rollout.gif")
    record_gif(
        env, classifier, policy,
        num_episodes=args.num_episodes,
        max_frames=args.max_frames,
        cell_size=args.cell_size,
        fps=args.fps,
        deterministic=args.deterministic,
        device=device,
        seed=args.seed,
        out_path=gif_path,
    )


if __name__ == "__main__":
    main()
