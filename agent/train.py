"""Train a recurrent PPO agent on :class:`env.MNISTMazeVecEnv`.

The pipeline is:

1. Load a pre-trained MNIST classifier from disk (see
   :mod:`agent.pretrain_mnist`). It is used as a frozen observation encoder:
   the policy never sees raw pixels, only the predicted digit class.
2. Build a vectorised maze of the requested ``--size`` with a fresh random
   layout (obstacles + coloring) and a :class:`agent.policy.GRUPolicy`.
3. Run PPO + GAE.

One run trains a single board size; use ``--size 10 / 20 / 30`` for the three
requested scales.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from agent.mnist_classifier import load_classifier
from agent.policy import GRUPolicy
from agent.ppo import PPOConfig, collect_rollout, ppo_update
from env import MNISTMazeVecEnv, load_mnist_by_class, random_color_map, random_obstacle_mask


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=int, default=10, help="Maze side length (10, 20, or 30).")
    p.add_argument("--n-colors", type=int, default=10)
    p.add_argument("--obstacle-fraction", type=float, default=0.12)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--rollout-length", type=int, default=128)
    p.add_argument("--total-steps", type=int, default=300_000, help="Total environment steps.")
    p.add_argument("--max-episode-steps", type=int, default=None, help="Defaults to 4 * size.")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument(
        "--value-clip-eps",
        type=float,
        default=0.2,
        help="Per-epoch value-clip range. Set to a negative number to disable.",
    )
    # γ=0.95 (horizon ≈ 20 steps) matches our short episodes (max 4 * size).
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument(
        "--curriculum-start",
        type=int,
        default=2,
        help="Initial max Manhattan distance from start to goal at episode reset. "
        "Set equal to --curriculum-end (or to the board diameter) to disable.",
    )
    p.add_argument(
        "--curriculum-end",
        type=int,
        default=None,
        help="Final max Manhattan distance; defaults to the full board diameter (height + width).",
    )
    p.add_argument(
        "--curriculum-fraction",
        type=float,
        default=0.5,
        help="Fraction of --total-steps over which max_goal_distance grows linearly "
        "from --curriculum-start to --curriculum-end. After that it stays at the end value.",
    )
    p.add_argument(
        "--mnist-checkpoint",
        type=Path,
        default=Path("checkpoints/mnist_classifier.pt"),
    )
    p.add_argument("--mnist-cache", type=Path, default=Path(".mnist_cache"))
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save the trained policy state_dict (defaults to checkpoints/policy_<size>x<size>.pt).",
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=1, help="Print stats every N rollouts.")
    return p.parse_args()


def curriculum_max_distance(step: int, total_steps: int, fraction: float, start: int, end: int) -> int:
    """Linear ramp from ``start`` to ``end`` over the first ``fraction`` of training."""
    ramp_steps = max(1, int(round(total_steps * fraction)))
    if step >= ramp_steps:
        return end
    t = step / ramp_steps
    return int(round(start + t * (end - start)))


def main() -> None:
    args = parse_args()
    assert args.num_envs % args.minibatches == 0
    height = width = args.size
    max_episode_steps = args.max_episode_steps or 4 * args.size
    diameter = height + width
    curriculum_end = args.curriculum_end if args.curriculum_end is not None else diameter

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # Layout (fixed for the whole run, per task spec).
    layout_rng = np.random.default_rng(args.seed)
    obstacles = random_obstacle_mask(height, width, args.obstacle_fraction, layout_rng)
    colors = random_color_map(height, width, args.n_colors, layout_rng)
    mnist_banks = load_mnist_by_class(cache_dir=args.mnist_cache)

    env = MNISTMazeVecEnv(
        num_envs=args.num_envs,
        height=height,
        width=width,
        obstacle_mask=obstacles,
        color_map=colors,
        n_colors=args.n_colors,
        max_steps=max_episode_steps,
        mnist_images_by_class=mnist_banks,
        seed=args.seed,
        max_goal_distance=args.curriculum_start,
    )

    classifier = load_classifier(args.mnist_checkpoint, device=device)
    policy = GRUPolicy(
        n_digits=10,
        goal_dim=height + width,
        n_actions=4,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    cfg = PPOConfig(
        rollout_length=args.rollout_length,
        epochs=args.epochs,
        minibatches=args.minibatches,
        lr=args.lr,
        clip_eps=args.clip_eps,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        value_clip_eps=(None if args.value_clip_eps < 0 else args.value_clip_eps),
    )

    obs, _ = env.reset(seed=args.seed)
    episode_start = np.ones(env.num_envs, dtype=bool)
    hidden = policy.initial_hidden(env.num_envs, device)
    prev_action = policy.initial_prev_action(env.num_envs, device)

    steps_per_rollout = args.rollout_length * args.num_envs
    n_rollouts = max(1, args.total_steps // steps_per_rollout)
    env_steps = 0
    t0 = time.time()

    metric_keys = [
        "env_steps",
        "ep_return_mean",
        "ep_length_mean",
        "ep_success_rate",
        "n_episodes",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_frac",
    ]
    metrics_log: list[dict[str, float]] = []

    print(
        f"Starting training: size={args.size}x{args.size}  "
        f"num_envs={args.num_envs}  rollout_len={args.rollout_length}  "
        f"total_steps={args.total_steps}  device={device}"
    )
    print(
        f"Curriculum: max_goal_distance ramps {args.curriculum_start} -> {curriculum_end} "
        f"over the first {int(args.curriculum_fraction * 100)}% of training."
    )

    for rollout_idx in range(n_rollouts):
        env.max_goal_distance = curriculum_max_distance(
            env_steps,
            args.total_steps,
            args.curriculum_fraction,
            args.curriculum_start,
            curriculum_end,
        )
        rollout, hidden, prev_action, obs, episode_start, info = collect_rollout(
            env, classifier, policy,
            rollout_length=cfg.rollout_length,
            hidden=hidden,
            prev_action=prev_action,
            last_obs=obs,
            last_episode_start=episode_start,
            device=device,
        )
        update_stats = ppo_update(policy, optimizer, rollout, cfg)
        env_steps += steps_per_rollout
        metrics_log.append({
            "env_steps": env_steps,
            "ep_return_mean": info["ep_return_mean"],
            "ep_length_mean": info["ep_length_mean"],
            "ep_success_rate": info["ep_success_rate"],
            "n_episodes": info["n_episodes"],
            **update_stats,
        })

        if (rollout_idx + 1) % args.log_every == 0 or rollout_idx == n_rollouts - 1:
            sps = env_steps / max(1e-6, time.time() - t0)
            print(
                f"[{env_steps:>8d}/{args.total_steps}]  "
                f"d_max={env.max_goal_distance:>3d}  "
                f"ep_ret={info['ep_return_mean']:.2f}  "
                f"ep_len={info['ep_length_mean']:.1f}  "
                f"succ={info['ep_success_rate']:.2f}  "
                f"n_ep={info['n_episodes']:>3d}  "
                f"pi_loss={update_stats['policy_loss']:+.3f}  "
                f"v_loss={update_stats['value_loss']:.3f}  "
                f"H={update_stats['entropy']:.3f}  "
                f"kl={update_stats['approx_kl']:+.3f}  "
                f"sps={sps:.0f}"
            )

    out = args.output or Path(f"checkpoints/policy_{args.size}x{args.size}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy": policy.state_dict(),
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "color_map": colors,
            "obstacle_mask": obstacles,
        },
        out,
    )

    metrics_path = out.with_name(out.stem + "_metrics.csv")
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metric_keys)
        writer.writeheader()
        writer.writerows(metrics_log)

    print(f"Saved policy to   {out}")
    print(f"Saved metrics to  {metrics_path}")


if __name__ == "__main__":
    main()
