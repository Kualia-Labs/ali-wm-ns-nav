from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class QTINavigationBaselineEnv(gym.Env):
    """Continual non-stationary 2D grid navigation (BASELINE / external-only sibling).

    This environment implements a rho-controlled non-stationary navigation task on a
    discrete grid. The agent must reach a goal while avoiding static and dynamic
    obstacles. A single scalar ``rho`` governs how frequently and strongly the layout,
    obstacle motion, and goal change (change-point structure). The episode follows a
    fixed five-regime continual sequence: static -> dynamic -> blocked -> goal-relocation
    -> unseen-layout.

    Observation space (fixed 59-D, identical shape across ALL siblings A0-A6):
        Layout order L1(15) -> L2(21) -> L3(11) -> L4(12).
        In THIS baseline sibling, ONLY L1 (external state) is populated; L2/L3/L4 are
        present but ZERO-MASKED in place (never resized) so the network input shape is
        identical across siblings.

        L1 (indices 0-14):
            0-1   robot position normalised to [0,1] by grid size
            2-3   goal position normalised to [0,1]
            4     euclidean distance to goal normalised by grid diagonal -> [0,1]
            5-12  egocentric 3x3 occupancy (8 cells, center excluded), out-of-bounds=1
            13-14 heading sin/cos in [-1,1]
        L2 (15-35), L3 (36-46), L4 (47-58): zero-masked.

    Action space:
        Discrete(5): {0=up, 1=down, 2=left, 3=right, 4=stay}.

    Reward (byte-identical across siblings):
        +1.0 goal reached; +0.05 per unit distance reduction; -0.25 collision;
        -0.05 repeated failed/blocked action; -0.002 per step; -0.1 failure-to-recover
        within the post-change window. Final reward clipped to [-10, 10].

    Termination: goal reached, or collision budget (5) exhausted.
    Truncation: step_count >= max_steps (200).
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    # --- grid / episode constants ---
    GRID_SIZE: int = 12
    MAX_STEPS: int = 200
    COLLISION_BUDGET: int = 5
    N_STATIC_OBSTACLES: int = 10
    N_DYNAMIC_OBSTACLES: int = 4
    RECOVERY_WINDOW: int = 15
    HISTORY_LEN: int = 5
    WINDOW_W: int = 20
    GAMMA: float = 0.99
    REGIME_COUNT: int = 5

    # --- observation layout ---
    OBS_DIM: int = 59
    L1_DIM: int = 15
    L2_DIM: int = 21
    L3_DIM: int = 11
    L4_DIM: int = 12

    # --- reward magnitudes ---
    R_GOAL: float = 1.0
    R_PROGRESS: float = 0.05
    R_COLLISION: float = -0.25
    R_REPEATED_FAIL: float = -0.05
    R_STEP: float = -0.002
    R_FAIL_RECOVER: float = -0.1
    REWARD_MIN: float = -10.0
    REWARD_MAX: float = 10.0
    SHAPING_CAP: float = 0.9

    # --- non-stationarity ---
    DEFAULT_RHO: float = 0.1

    # action deltas: up, down, left, right, stay
    _ACTION_DELTAS: tuple[tuple[int, int], ...] = (
        (0, -1),
        (0, 1),
        (-1, 0),
        (1, 0),
        (0, 0),
    )

    def __init__(self, render_mode: str | None = None, rho: float = DEFAULT_RHO) -> None:
        super().__init__()

        self.render_mode = render_mode
        self.rho = float(np.clip(rho, 0.0, 1.0))

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.OBS_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(5)

        self._diag = float(np.sqrt(2.0) * self.GRID_SIZE)

        self.robot_pos = np.zeros(2, dtype=np.int32)
        self.goal_pos = np.zeros(2, dtype=np.int32)
        self.static_obstacles: set[tuple[int, int]] = set()
        self.dynamic_obstacles: list[np.ndarray] = []
        self.dynamic_dirs: list[int] = []
        self.heading: float = 0.0

        self.step_count: int = 0
        self.collisions: int = 0
        self.regime: int = 0

        self._dist_history: list[float] = []
        self._reward_history: list[float] = []
        self._blocked_history: list[float] = []
        self._consecutive_blocked: int = 0
        self._last_blocked_pos: tuple[int, int] | None = None
        self._cum_shaping: float = 0.0
        self._prev_dist: float = 0.0

        self._steps_since_change: int = 0
        self._recover_penalty_applied: bool = False
        self._success_since_change: bool = False

    # ------------------------------------------------------------------ helpers
    def _euclidean(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a.astype(np.float32) - b.astype(np.float32)))

    def _is_blocked_cell(self, x: int, y: int) -> bool:
        if x < 0 or y < 0 or x >= self.GRID_SIZE or y >= self.GRID_SIZE:
            return True
        if (x, y) in self.static_obstacles:
            return True
        return False

    def _dynamic_at(self, x: int, y: int) -> bool:
        for d in self.dynamic_obstacles:
            if int(d[0]) == x and int(d[1]) == y:
                return True
        return False

    def _random_free_cell(self, occupied: set[tuple[int, int]]) -> np.ndarray:
        for _ in range(200):
            c = self.np_random.integers(0, self.GRID_SIZE, size=2)
            t = (int(c[0]), int(c[1]))
            if t not in occupied:
                return c.astype(np.int32)
        for x in range(self.GRID_SIZE):
            for y in range(self.GRID_SIZE):
                if (x, y) not in occupied:
                    return np.array([x, y], dtype=np.int32)
        return np.array([0, 0], dtype=np.int32)

    def _build_layout(self) -> None:
        occupied: set[tuple[int, int]] = set()

        n_static = self.N_STATIC_OBSTACLES
        n_dynamic = self.N_DYNAMIC_OBSTACLES
        if self.regime == 0:
            n_static = self.N_STATIC_OBSTACLES // 2
            n_dynamic = 0
        elif self.regime == 1:
            n_static = self.N_STATIC_OBSTACLES // 2
            n_dynamic = self.N_DYNAMIC_OBSTACLES
        elif self.regime == 2:
            n_static = self.N_STATIC_OBSTACLES + 4
            n_dynamic = 1
        elif self.regime == 3:
            n_static = self.N_STATIC_OBSTACLES
            n_dynamic = self.N_DYNAMIC_OBSTACLES // 2
        else:
            n_static = self.N_STATIC_OBSTACLES + 2
            n_dynamic = self.N_DYNAMIC_OBSTACLES

        self.static_obstacles = set()
        for _ in range(n_static):
            c = self._random_free_cell(occupied)
            t = (int(c[0]), int(c[1]))
            self.static_obstacles.add(t)
            occupied.add(t)

        self.robot_pos = self._random_free_cell(occupied)
        occupied.add((int(self.robot_pos[0]), int(self.robot_pos[1])))

        best_goal = None
        best_d = -1.0
        for _ in range(20):
            g = self._random_free_cell(occupied)
            d = self._euclidean(g, self.robot_pos)
            if d > best_d:
                best_d = d
                best_goal = g
        self.goal_pos = (
            best_goal if best_goal is not None else self._random_free_cell(occupied)
        )
        occupied.add((int(self.goal_pos[0]), int(self.goal_pos[1])))

        self.dynamic_obstacles = []
        self.dynamic_dirs = []
        for _ in range(n_dynamic):
            c = self._random_free_cell(occupied)
            self.dynamic_obstacles.append(c.astype(np.int32))
            self.dynamic_dirs.append(int(self.np_random.integers(0, 4)))
            occupied.add((int(c[0]), int(c[1])))

        self.heading = 0.0

    def _move_dynamic_obstacles(self) -> None:
        for i, d in enumerate(self.dynamic_obstacles):
            direction = self.dynamic_dirs[i]
            dx, dy = self._ACTION_DELTAS[direction]
            nx, ny = int(d[0]) + dx, int(d[1]) + dy
            if self._is_blocked_cell(nx, ny) or (nx, ny) == (
                int(self.goal_pos[0]),
                int(self.goal_pos[1]),
            ):
                self.dynamic_dirs[i] = int(self.np_random.integers(0, 4))
            else:
                self.dynamic_obstacles[i] = np.array([nx, ny], dtype=np.int32)

    def _maybe_trigger_change(self) -> bool:
        if self.rho <= 0.0:
            return False
        if self.np_random.random() < self.rho:
            kind = self.np_random.random()
            occupied: set[tuple[int, int]] = set(self.static_obstacles)
            occupied.add((int(self.robot_pos[0]), int(self.robot_pos[1])))
            for d in self.dynamic_obstacles:
                occupied.add((int(d[0]), int(d[1])))
            if kind < 0.4:
                self.goal_pos = self._random_free_cell(occupied)
            elif kind < 0.7:
                if self.static_obstacles:
                    old = next(iter(self.static_obstacles))
                    self.static_obstacles.discard(old)
                    new_cell = self._random_free_cell(occupied)
                    self.static_obstacles.add((int(new_cell[0]), int(new_cell[1])))
            else:
                for i in range(len(self.dynamic_dirs)):
                    self.dynamic_dirs[i] = int(self.np_random.integers(0, 4))
            return True
        return False

    def _egocentric_occupancy(self) -> np.ndarray:
        cells = []
        rx, ry = int(self.robot_pos[0]), int(self.robot_pos[1])
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                x, y = rx + dx, ry + dy
                occ = 1.0 if (self._is_blocked_cell(x, y) or self._dynamic_at(x, y)) else 0.0
                cells.append(occ)
        return np.asarray(cells, dtype=np.float32)

    def _build_observation(self) -> np.ndarray:
        obs = np.zeros(self.OBS_DIM, dtype=np.float32)

        gs = float(self.GRID_SIZE)
        obs[0] = float(self.robot_pos[0]) / gs
        obs[1] = float(self.robot_pos[1]) / gs
        obs[2] = float(self.goal_pos[0]) / gs
        obs[3] = float(self.goal_pos[1]) / gs

        dist = self._euclidean(self.robot_pos, self.goal_pos)
        obs[4] = float(np.clip(dist / self._diag, 0.0, 1.0))

        obs[5:13] = self._egocentric_occupancy()

        obs[13] = float(np.sin(self.heading))
        obs[14] = float(np.cos(self.heading))

        # L2/L3/L4 zero-masked in place (baseline sibling)

        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        return obs.astype(self.observation_space.dtype)

    # ------------------------------------------------------------------ gym api
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        if options is not None:
            if "rho" in options:
                self.rho = float(np.clip(options["rho"], 0.0, 1.0))
            if "regime" in options:
                self.regime = int(options["regime"]) % self.REGIME_COUNT
            else:
                self.regime = int(self.np_random.integers(0, self.REGIME_COUNT))
        else:
            self.regime = int(self.np_random.integers(0, self.REGIME_COUNT))

        self.step_count = 0
        self.collisions = 0

        self._build_layout()

        self._prev_dist = self._euclidean(self.robot_pos, self.goal_pos)
        self._cum_shaping = 0.0

        self._dist_history = []
        self._reward_history = []
        self._blocked_history = []
        self._consecutive_blocked = 0
        self._last_blocked_pos = None

        self._steps_since_change = 0
        self._recover_penalty_applied = False
        self._success_since_change = False

        obs = self._build_observation()
        info: dict[str, Any] = {
            "regime": int(self.regime),
            "rho": float(self.rho),
            "distance_to_goal": float(self._prev_dist),
            "collisions": int(self.collisions),
            "reward_components": {
                "goal": 0.0,
                "progress": 0.0,
                "collision": 0.0,
                "repeated_failed": 0.0,
                "step": 0.0,
                "failure_to_recover": 0.0,
            },
        }
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        try:
            a = int(action)
        except (TypeError, ValueError):
            a = 4
        if a < 0 or a > 4:
            a = 4

        self.step_count += 1
        self._steps_since_change += 1

        reward_components = {
            "goal": 0.0,
            "progress": 0.0,
            "collision": 0.0,
            "repeated_failed": 0.0,
            "step": self.R_STEP,
            "failure_to_recover": 0.0,
        }

        dx, dy = self._ACTION_DELTAS[a]
        rx, ry = int(self.robot_pos[0]), int(self.robot_pos[1])
        nx, ny = rx + dx, ry + dy

        blocked = False
        collided = False
        terminated = False

        if a != 4:
            self.heading = float(np.arctan2(float(dy), float(dx)))

        if a != 4 and self._dynamic_at(nx, ny):
            collided = True
            self.collisions += 1
            reward_components["collision"] += self.R_COLLISION
        elif a != 4 and self._is_blocked_cell(nx, ny):
            blocked = True
            if self._last_blocked_pos == (nx, ny):
                self._consecutive_blocked += 1
            else:
                self._consecutive_blocked = 1
                self._last_blocked_pos = (nx, ny)
            reward_components["repeated_failed"] = self.R_REPEATED_FAIL
        else:
            if a != 4:
                self.robot_pos = np.array([nx, ny], dtype=np.int32)
                self._consecutive_blocked = 0
                self._last_blocked_pos = None

        self._move_dynamic_obstacles()

        if self._dynamic_at(int(self.robot_pos[0]), int(self.robot_pos[1])):
            if not collided:
                collided = True
                self.collisions += 1
                reward_components["collision"] += self.R_COLLISION

        new_dist = self._euclidean(self.robot_pos, self.goal_pos)
        delta = self._prev_dist - new_dist
        progress = 0.0
        if abs(delta) > 1e-6:
            progress = self.R_PROGRESS * float(delta)
            remaining = self.SHAPING_CAP - self._cum_shaping
            if progress > 0.0:
                progress = min(progress, max(0.0, remaining))
            self._cum_shaping += progress
        reward_components["progress"] = progress
        self._prev_dist = new_dist

        if int(self.robot_pos[0]) == int(self.goal_pos[0]) and int(
            self.robot_pos[1]
        ) == int(self.goal_pos[1]):
            reward_components["goal"] = self.R_GOAL
            terminated = True
            self._success_since_change = True

        changed = self._maybe_trigger_change()
        if changed:
            self._steps_since_change = 0
            self._recover_penalty_applied = False
            self._success_since_change = False
            self._prev_dist = self._euclidean(self.robot_pos, self.goal_pos)

        if (
            not self._recover_penalty_applied
            and self._steps_since_change >= self.RECOVERY_WINDOW
            and not self._success_since_change
        ):
            reward_components["failure_to_recover"] = self.R_FAIL_RECOVER
            self._recover_penalty_applied = True

        if self.collisions >= self.COLLISION_BUDGET:
            terminated = True

        truncated = bool(self.step_count >= self.MAX_STEPS)

        total_reward = float(sum(reward_components.values()))
        if not np.isfinite(total_reward):
            total_reward = 0.0
        total_reward = float(np.clip(total_reward, self.REWARD_MIN, self.REWARD_MAX))

        self._dist_history.append(float(new_dist))
        self._reward_history.append(float(total_reward))
        self._blocked_history.append(1.0 if (blocked or collided) else 0.0)
        if len(self._dist_history) > self.WINDOW_W:
            self._dist_history.pop(0)
        if len(self._reward_history) > self.WINDOW_W:
            self._reward_history.pop(0)
        if len(self._blocked_history) > self.WINDOW_W:
            self._blocked_history.pop(0)

        obs = self._build_observation()

        info: dict[str, Any] = {
            "regime": int(self.regime),
            "rho": float(self.rho),
            "distance_to_goal": float(new_dist),
            "collisions": int(self.collisions),
            "blocked": bool(blocked),
            "collided": bool(collided),
            "changed": bool(changed),
            "steps_since_change": int(self._steps_since_change),
            "reward_components": reward_components,
        }

        return obs, total_reward, bool(terminated), bool(truncated), info

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None
        grid = [["." for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        for (x, y) in self.static_obstacles:
            if 0 <= x < self.GRID_SIZE and 0 <= y < self.GRID_SIZE:
                grid[y][x] = "#"
        for d in self.dynamic_obstacles:
            x, y = int(d[0]), int(d[1])
            if 0 <= x < self.GRID_SIZE and 0 <= y < self.GRID_SIZE:
                grid[y][x] = "o"
        gx, gy = int(self.goal_pos[0]), int(self.goal_pos[1])
        if 0 <= gx < self.GRID_SIZE and 0 <= gy < self.GRID_SIZE:
            grid[gy][gx] = "G"
        rx, ry = int(self.robot_pos[0]), int(self.robot_pos[1])
        if 0 <= rx < self.GRID_SIZE and 0 <= ry < self.GRID_SIZE:
            grid[ry][rx] = "R"
        return "\n".join("".join(row) for row in grid)

    def close(self) -> None:
        return None
