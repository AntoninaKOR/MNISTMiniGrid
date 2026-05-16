"""Vectorized MNIST-based GridWorld-style maze environment.

The maze is a ``(height, width)`` rectangle surrounded by walls. Obstacles are
placed inside via a binary mask; floor cells are colored from a fixed colormap.
At each step the agent observes a single cell: an MNIST handwritten digit
randomly sampled from the class corresponding to the perceived cell color.
Walls and obstacles use one dedicated color each; the remaining
``n_colors - 2`` colors are assigned to floor cells.

The environment implements the ``gymnasium.vector.VectorEnv`` interface with
``AutoresetMode.NEXT_STEP``: when a sub-environment terminates or truncates,
the next call to ``step`` resets it before applying any action.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space

from env.mnist_data import load_mnist_by_class

# Actions are numbered clockwise starting from "up".
# Coordinates are (row, col); row grows downward.
ACTION_DELTAS = np.array(
    [
        [-1, 0],  # 0: up
        [0, 1],   # 1: right
        [1, 0],   # 2: down
        [0, -1],  # 3: left
    ],
    dtype=np.int64,
)
NUM_ACTIONS = 4

# Visualisation palette (one RGB triple per color index, up to 10 colors).
# Indices 0 and 1 are the conventional defaults for wall and obstacle.
DEFAULT_PALETTE = np.array(
    [
        (40, 44, 52),     # 0 wall     -- dark slate
        (12, 12, 14),     # 1 obstacle -- near black
        (231, 76, 60),    # 2 red
        (46, 204, 113),   # 3 green
        (52, 152, 219),   # 4 blue
        (241, 196, 15),   # 5 yellow
        (155, 89, 182),   # 6 purple
        (230, 126, 34),   # 7 orange
        (26, 188, 156),   # 8 teal
        (236, 64, 122),   # 9 pink
    ],
    dtype=np.uint8,
)


def random_obstacle_mask(
    height: int,
    width: int,
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a binary ``(height, width)`` obstacle mask.

    Exactly ``round(height * width * fraction)`` cells are marked as obstacles.
    """
    assert 0.0 <= fraction < 1.0, "fraction must be in [0, 1)"
    n_cells = height * width
    n_obstacles = int(round(n_cells * fraction))
    flat = np.zeros(n_cells, dtype=bool)
    flat[rng.choice(n_cells, size=n_obstacles, replace=False)] = True
    return flat.reshape(height, width)


def random_color_map(
    height: int,
    width: int,
    n_colors: int,
    rng: np.random.Generator,
    wall_color: int = 0,
    obstacle_color: int = 1,
) -> np.ndarray:
    """Sample a random ``(height, width)`` coloring using the floor palette.

    Floor cells use the ``n_colors - 2`` colors that remain after removing
    ``wall_color`` and ``obstacle_color`` from ``range(n_colors)``.
    """
    assert 2 < n_colors <= 10, "need at least one floor color and at most 10 classes"
    assert 0 <= wall_color < n_colors
    assert 0 <= obstacle_color < n_colors
    assert wall_color != obstacle_color
    floor_palette = np.array(
        [c for c in range(n_colors) if c not in (wall_color, obstacle_color)],
        dtype=np.int64,
    )
    return floor_palette[rng.integers(0, floor_palette.size, size=(height, width))]


def two_hot_encode(positions: np.ndarray, height: int, width: int) -> np.ndarray:
    """Two-hot encode ``(N, 2)`` integer ``(row, col)`` positions.

    The returned ``(N, height + width)`` ``float32`` array has a 1 at the row
    index in the first ``height`` slots and a 1 at the column index in the
    last ``width`` slots.
    """
    encoded = np.zeros((positions.shape[0], height + width), dtype=np.float32)
    rows = np.arange(positions.shape[0])
    encoded[rows, positions[:, 0]] = 1.0
    encoded[rows, height + positions[:, 1]] = 1.0
    return encoded


class MNISTMazeVecEnv(VectorEnv):
    """Vectorized MNIST maze environment.

    Parameters
    ----------
    num_envs:
        Number of parallel sub-environments.
    height, width:
        Size of the rectangular maze area (excluding the surrounding wall).
    obstacle_mask:
        ``(height, width)`` boolean mask marking obstacle cells.
    color_map:
        ``(height, width)`` integer array with a floor color for every cell.
        Values must lie in ``range(n_colors) \\ {wall_color, obstacle_color}``.
        The colormap is fixed for the lifetime of the environment.
    n_colors:
        Total number of colors (and MNIST classes). Must satisfy
        ``2 < n_colors <= 10``.
    max_steps:
        Episode length limit. Episodes that do not reach the goal within
        ``max_steps`` are truncated.
    wall_color, obstacle_color:
        Color indices reserved for walls and obstacles.
    mnist_images_by_class:
        Optional pre-loaded MNIST images grouped by class. When omitted the
        dataset is downloaded and cached automatically.
    seed:
        Seed for the internal RNG used for position/MNIST sampling.
    """

    metadata = {"autoreset_mode": AutoresetMode.NEXT_STEP}

    def __init__(
        self,
        num_envs: int,
        height: int,
        width: int,
        obstacle_mask: np.ndarray,
        color_map: np.ndarray,
        n_colors: int = 10,
        max_steps: int = 100,
        wall_color: int = 0,
        obstacle_color: int = 1,
        mnist_images_by_class: list[np.ndarray] | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        assert obstacle_mask.shape == (height, width)
        assert color_map.shape == (height, width)
        assert 2 < n_colors <= 10
        assert 0 <= wall_color < n_colors
        assert 0 <= obstacle_color < n_colors
        assert wall_color != obstacle_color

        floor_palette = {c for c in range(n_colors) if c not in (wall_color, obstacle_color)}
        cmap_values = set(np.unique(color_map).tolist())
        assert cmap_values.issubset(floor_palette), (
            f"color_map uses colors {sorted(cmap_values)} outside the floor "
            f"palette {sorted(floor_palette)}"
        )

        self.num_envs = int(num_envs)
        self.height = int(height)
        self.width = int(width)
        self.n_colors = int(n_colors)
        self.max_steps = int(max_steps)
        self.wall_color = int(wall_color)
        self.obstacle_color = int(obstacle_color)
        self.obstacle_mask = obstacle_mask.astype(bool)
        self.color_map = color_map.astype(np.int64)

        self._rng = np.random.default_rng(seed)

        if mnist_images_by_class is None:
            mnist_images_by_class = load_mnist_by_class()
        assert len(mnist_images_by_class) >= n_colors, (
            f"need MNIST images for at least {n_colors} classes, got {len(mnist_images_by_class)}"
        )
        self._images_by_class = [np.ascontiguousarray(a, dtype=np.uint8) for a in mnist_images_by_class]

        self._free_indices = np.flatnonzero(~self.obstacle_mask.ravel())
        assert self._free_indices.size >= 2, "need at least two free cells (agent + goal)"

        self.single_observation_space = spaces.Dict(
            {
                "image": spaces.Box(0, 255, shape=(28, 28), dtype=np.uint8),
                "goal": spaces.Box(0.0, 1.0, shape=(self.height + self.width,), dtype=np.float32),
            }
        )
        self.single_action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self.action_space = batch_space(self.single_action_space, self.num_envs)

        # Per-env state.
        self.pos_agent = np.zeros((self.num_envs, 2), dtype=np.int64)
        self.pos_goal = np.zeros((self.num_envs, 2), dtype=np.int64)
        # Position the agent started the current episode from (used for rendering).
        self.pos_start = np.zeros((self.num_envs, 2), dtype=np.int64)
        self.step_count = np.zeros(self.num_envs, dtype=np.int64)
        self.episode_return = np.zeros(self.num_envs, dtype=np.float32)
        self.need_reset = np.zeros(self.num_envs, dtype=bool)
        self.last_color = np.zeros(self.num_envs, dtype=np.int64)
        # Most recently emitted MNIST observation per env, used for rendering.
        self.last_image = np.zeros((self.num_envs, 28, 28), dtype=np.uint8)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _sample_positions(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Sample ``n`` disjoint ``(agent, goal)`` pairs on free cells."""
        agent_flat = self._rng.choice(self._free_indices, size=n)
        goal_flat = self._rng.choice(self._free_indices, size=n)
        collisions = goal_flat == agent_flat
        while collisions.any():
            goal_flat[collisions] = self._rng.choice(
                self._free_indices, size=int(collisions.sum())
            )
            collisions = goal_flat == agent_flat
        agent = np.stack(np.unravel_index(agent_flat, (self.height, self.width)), axis=-1)
        goal = np.stack(np.unravel_index(goal_flat, (self.height, self.width)), axis=-1)
        return agent.astype(np.int64), goal.astype(np.int64)

    def _sample_mnist(self, color_indices: np.ndarray) -> np.ndarray:
        """Sample one MNIST image per element in ``color_indices``."""
        out = np.empty((color_indices.shape[0], 28, 28), dtype=np.uint8)
        for i, c in enumerate(color_indices):
            bank = self._images_by_class[int(c)]
            out[i] = bank[self._rng.integers(0, bank.shape[0])]
        return out

    def _build_obs(self) -> dict[str, np.ndarray]:
        images = self._sample_mnist(self.last_color)
        self.last_image = images
        return {
            "image": images,
            "goal": two_hot_encode(self.pos_goal, self.height, self.width),
        }

    def _reset_envs(self, indices: np.ndarray) -> None:
        n = indices.size
        if n == 0:
            return
        agent, goal = self._sample_positions(n)
        self.pos_agent[indices] = agent
        self.pos_start[indices] = agent
        self.pos_goal[indices] = goal
        self.step_count[indices] = 0
        self.episode_return[indices] = 0.0
        self.need_reset[indices] = False
        self.last_color[indices] = self.color_map[agent[:, 0], agent[:, 1]]

    # ------------------------------------------------------------------
    # Gymnasium VectorEnv API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._reset_envs(np.arange(self.num_envs))
        return self._build_obs(), {}

    def step(
        self, actions: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        actions = np.asarray(actions, dtype=np.int64)
        assert actions.shape == (self.num_envs,)

        # NEXT_STEP autoreset: envs that finished last step are reset now and
        # their action is ignored.
        just_reset = self.need_reset.copy()
        self._reset_envs(np.flatnonzero(just_reset))
        active = ~just_reset

        deltas = ACTION_DELTAS[actions]
        proposed = self.pos_agent + deltas
        in_bounds = (
            (proposed[:, 0] >= 0)
            & (proposed[:, 0] < self.height)
            & (proposed[:, 1] >= 0)
            & (proposed[:, 1] < self.width)
        )
        # Look up obstacle status using a safe (clamped) position so that
        # out-of-bounds moves do not trigger an array index error.
        safe = np.where(in_bounds[:, None], proposed, self.pos_agent)
        is_obstacle = in_bounds & self.obstacle_mask[safe[:, 0], safe[:, 1]]
        blocked = ~in_bounds | is_obstacle
        movable = active & ~blocked

        self.pos_agent[movable] = proposed[movable]

        # Determine what the agent sees: wall, obstacle, or the floor color at
        # its new position.
        seen = np.where(
            ~in_bounds,
            self.wall_color,
            np.where(
                is_obstacle,
                self.obstacle_color,
                self.color_map[safe[:, 0], safe[:, 1]],
            ),
        )
        self.last_color[active] = seen[active]
        self.step_count[active] += 1

        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        rewards = np.zeros(self.num_envs, dtype=np.float32)

        reached = active & np.all(self.pos_agent == self.pos_goal, axis=-1)
        terminated[reached] = True
        rewards[reached] = 1.0
        self.episode_return[active] += rewards[active]

        timed_out = active & ~terminated & (self.step_count >= self.max_steps)
        truncated[timed_out] = True

        self.need_reset = terminated | truncated
        return self._build_obs(), rewards, terminated, truncated, {}

    def close(self) -> None:
        return None

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render_frame(
        self,
        env_idx: int = 0,
        cell_size: int = 24,
        wall_thickness: int | None = None,
        pad: int = 12,
        bg_color: tuple[int, int, int] = (24, 24, 24),
        text_color: tuple[int, int, int] = (235, 235, 235),
        palette: np.ndarray | None = None,
    ) -> np.ndarray:
        """Render a single ``(H, W, 3)`` ``uint8`` RGB frame for one sub-env.

        Layout: the MNIST image the agent is currently observing on the left,
        a "color -> digit" legend in the middle, the colored maze (with start,
        goal and agent markers) on the right, and a status bar with score and
        step count along the bottom.
        """
        from PIL import Image, ImageDraw, ImageFont

        assert 0 <= env_idx < self.num_envs
        palette = DEFAULT_PALETTE if palette is None else np.asarray(palette, dtype=np.uint8)
        assert palette.shape[0] >= self.n_colors
        if wall_thickness is None:
            wall_thickness = cell_size

        # -- Maze panel ------------------------------------------------------
        maze_h_inner = self.height * cell_size
        maze_w_inner = self.width * cell_size
        maze_h = maze_h_inner + 2 * wall_thickness
        maze_w = maze_w_inner + 2 * wall_thickness
        maze_img = Image.new("RGB", (maze_w, maze_h), tuple(int(c) for c in palette[self.wall_color]))
        draw = ImageDraw.Draw(maze_img)

        for r in range(self.height):
            for c in range(self.width):
                color_idx = self.obstacle_color if self.obstacle_mask[r, c] else int(self.color_map[r, c])
                color = tuple(int(v) for v in palette[color_idx])
                x0 = wall_thickness + c * cell_size
                y0 = wall_thickness + r * cell_size
                draw.rectangle(
                    [x0, y0, x0 + cell_size - 1, y0 + cell_size - 1],
                    fill=color,
                )

        marker_border = max(3, cell_size // 5)

        sr, sc = int(self.pos_start[env_idx, 0]), int(self.pos_start[env_idx, 1])
        sx, sy = wall_thickness + sc * cell_size, wall_thickness + sr * cell_size
        draw.rectangle(
            [sx + 1, sy + 1, sx + cell_size - 2, sy + cell_size - 2],
            outline=(76, 175, 80),
            width=marker_border,
        )

        gr, gc = int(self.pos_goal[env_idx, 0]), int(self.pos_goal[env_idx, 1])
        gx, gy = wall_thickness + gc * cell_size, wall_thickness + gr * cell_size
        draw.rectangle(
            [gx + 1, gy + 1, gx + cell_size - 2, gy + cell_size - 2],
            outline=(255, 235, 59),
            width=marker_border,
        )

        ar, ac = int(self.pos_agent[env_idx, 0]), int(self.pos_agent[env_idx, 1])
        ax, ay = wall_thickness + ac * cell_size, wall_thickness + ar * cell_size
        radius = max(3, cell_size // 2 - marker_border - max(1, cell_size // 16))
        cx, cy = ax + cell_size // 2, ay + cell_size // 2
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=(245, 245, 245),
            outline=(10, 10, 10),
            width=max(1, cell_size // 12),
        )

        maze_rgb = np.array(maze_img)

        # -- MNIST panel -----------------------------------------------------
        mnist_scale = max(1, maze_h // 28)
        mnist = np.repeat(np.repeat(self.last_image[env_idx], mnist_scale, axis=0), mnist_scale, axis=1)
        mnist_rgb = np.stack([mnist] * 3, axis=-1)

        # -- Legend ----------------------------------------------------------
        legend_font_size = max(14, cell_size)
        legend_font = ImageFont.load_default(size=legend_font_size)
        swatch = max(16, cell_size)
        row_gap = max(3, swatch // 4)
        # Digit label width: use the widest single-digit advance to keep rows aligned.
        label_advance = max(legend_font.getbbox(str(d))[2] for d in range(self.n_colors))
        legend_w = swatch + max(6, swatch // 3) + label_advance
        legend_h = self.n_colors * swatch + (self.n_colors - 1) * row_gap
        legend_img = Image.new("RGB", (legend_w, legend_h), bg_color)
        ldraw = ImageDraw.Draw(legend_img)
        for i in range(self.n_colors):
            y = i * (swatch + row_gap)
            color = tuple(int(v) for v in palette[i])
            ldraw.rectangle([0, y, swatch - 1, y + swatch - 1], fill=color, outline=(80, 80, 80), width=1)
            # Vertically centre the digit relative to the swatch.
            label = str(i)
            lbb = legend_font.getbbox(label)
            ly = y + (swatch - (lbb[3] - lbb[1])) // 2 - lbb[1]
            ldraw.text((swatch + max(6, swatch // 3), ly), label, fill=text_color, font=legend_font)
        legend_rgb = np.array(legend_img)

        # -- Compose top row -------------------------------------------------
        top_h = max(mnist_rgb.shape[0], legend_rgb.shape[0], maze_rgb.shape[0]) + 2 * pad
        total_w = pad + mnist_rgb.shape[1] + pad + legend_rgb.shape[1] + pad + maze_rgb.shape[1] + pad

        # -- Status bar ------------------------------------------------------
        status_font_size = max(16, cell_size + 4)
        status_font = ImageFont.load_default(size=status_font_size)
        score = float(self.episode_return[env_idx])
        step = int(self.step_count[env_idx])
        status_text = f"Step: {step:>4d} / {self.max_steps}     Score: {score:.0f}"
        sbb = status_font.getbbox(status_text)
        status_h = (sbb[3] - sbb[1]) + 2 * pad

        # -- Assemble --------------------------------------------------------
        total_h = top_h + status_h
        canvas = Image.new("RGB", (total_w, total_h), bg_color)
        mnist_x = pad
        legend_x = mnist_x + mnist_rgb.shape[1] + pad
        maze_x = legend_x + legend_rgb.shape[1] + pad
        canvas.paste(Image.fromarray(mnist_rgb), (mnist_x, (top_h - mnist_rgb.shape[0]) // 2))
        canvas.paste(Image.fromarray(legend_rgb), (legend_x, (top_h - legend_rgb.shape[0]) // 2))
        canvas.paste(Image.fromarray(maze_rgb), (maze_x, (top_h - maze_rgb.shape[0]) // 2))

        cdraw = ImageDraw.Draw(canvas)
        cdraw.line([(pad, top_h), (total_w - pad, top_h)], fill=(60, 60, 60), width=1)
        cdraw.text(
            (pad, top_h + pad - sbb[1]),
            status_text,
            fill=text_color,
            font=status_font,
        )
        return np.array(canvas)
