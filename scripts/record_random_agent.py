"""Record a GIF of a scripted agent acting in :class:`MNISTMazeVecEnv`.

Each frame shows the MNIST observation the agent currently sees on the left
and the maze with the start cell (green box), the goal (yellow box) and the
agent (white circle) on the right.

Three policies are available via ``--policy``:

* ``random`` (default) — uniformly samples from the four directional actions.
* ``stay``  — the agent never acts; the env is not stepped, only the MNIST
  observation is resampled each frame so the floor-class digits keep changing.
* ``wall``  — the agent is teleported next to a wall and keeps sending the
  action that bumps into it; the maze view stays still but the MNIST shows
  fresh samples of the wall class on every step.

Example:
    python scripts/record_random_agent.py --size 20 --num-steps 300 \
        --output rollout_20x20.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from env import (
    MNISTMazeVecEnv,
    load_mnist_by_class,
    random_color_map,
    random_obstacle_mask,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=int, default=20, help="Side length of the square maze.")
    p.add_argument("--height", type=int, default=None, help="Maze height (overrides --size).")
    p.add_argument("--width", type=int, default=None, help="Maze width (overrides --size).")
    p.add_argument("--obstacle-fraction", type=float, default=0.12, help="Fraction of cells turned into obstacles.")
    p.add_argument("--n-colors", type=int, default=10, help="Total number of colors (1 wall + 1 obstacle + the rest floor).")
    p.add_argument("--num-steps", type=int, default=300, help="Number of environment steps to record.")
    p.add_argument("--max-steps", type=int, default=None, help="Episode length limit (defaults to 4 * h * w).")
    p.add_argument(
        "--policy",
        choices=("random", "stay", "wall"),
        default="random",
        help="Agent behaviour: random actions, stand still, or repeatedly bump into a wall.",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed for env, layout, and the random policy.")
    p.add_argument("--cell-size", type=int, default=24, help="Pixel size of one maze cell.")
    p.add_argument("--fps", type=float, default=3.0, help="GIF frames-per-second.")
    p.add_argument("--output", type=Path, default=Path("rollout.gif"), help="Output GIF path.")
    p.add_argument(
        "--mnist-cache",
        type=Path,
        default=Path(".mnist_cache"),
        help="Directory used to cache the downloaded MNIST files.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    height = args.height if args.height is not None else args.size
    width = args.width if args.width is not None else args.size
    max_steps = args.max_steps if args.max_steps is not None else 4 * height * width

    layout_rng = np.random.default_rng(args.seed)
    policy_rng = np.random.default_rng(args.seed + 1)

    obstacles = random_obstacle_mask(height, width, args.obstacle_fraction, layout_rng)
    colors = random_color_map(height, width, args.n_colors, layout_rng)
    mnist_banks = load_mnist_by_class(cache_dir=args.mnist_cache)

    env = MNISTMazeVecEnv(
        num_envs=1,
        height=height,
        width=width,
        obstacle_mask=obstacles,
        color_map=colors,
        n_colors=args.n_colors,
        max_steps=max_steps,
        mnist_images_by_class=mnist_banks,
        seed=args.seed,
    )

    obs, _ = env.reset(seed=args.seed)

    if args.policy == "wall":
        # Teleport the agent onto a top-row floor cell so action 0 (up) always
        # hits the surrounding wall and the agent never moves.
        top_free = [c for c in range(width) if not obstacles[0, c]]
        if not top_free:
            raise SystemExit("--policy wall requires at least one free cell in the top row")
        start_col = top_free[len(top_free) // 2]
        env.pos_agent[:] = np.array([[0, start_col]])
        env.pos_start[:] = env.pos_agent
        env.last_color[:] = env.color_map[0, start_col]
        env.step_count[:] = 0
        env.need_reset[:] = False
        env._build_obs()  # refresh last_image so the very first frame is consistent

    frames = [Image.fromarray(env.render_frame(env_idx=0, cell_size=args.cell_size))]
    episode_returns: list[float] = []
    current_return = 0.0

    for _ in range(args.num_steps):
        if args.policy == "random":
            actions = policy_rng.integers(0, 4, size=env.num_envs)
        elif args.policy == "wall":
            actions = np.zeros(env.num_envs, dtype=np.int64)  # always "up"
        else:  # "stay" -- never step, only refresh the MNIST observation.
            actions = None

        if actions is not None:
            obs, reward, terminated, truncated, _ = env.step(actions)
            current_return += float(reward[0])
            if bool(terminated[0]) or bool(truncated[0]):
                episode_returns.append(current_return)
                current_return = 0.0
        else:
            env._build_obs()  # re-sample MNIST without advancing the env

        frames.append(Image.fromarray(env.render_frame(env_idx=0, cell_size=args.cell_size)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(round(1000.0 / args.fps))
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )

    print(f"Saved {len(frames)} frames to {args.output} ({height}x{width}, {args.num_steps} steps, {args.fps} fps).")
    if episode_returns:
        print(
            f"Completed {len(episode_returns)} episode(s) during the rollout; "
            f"returns: {episode_returns}"
        )
    else:
        print("Random agent did not finish a single episode during the rollout.")


if __name__ == "__main__":
    main()
