from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class QTIMultiScaleNavigationTreatment(gym.Env):
    """Continual non-stationary 2D grid navigation (QTI TREATMENT sibling).

    The agent navigates a grid to a goal while avoiding static and dynamic
    obstacles. A scalar ``rho`` governs the frequency/magnitude of change
    events. Change-event timing defines ground-truth change-points used only
    for reward bookkeeping (the failure-to-recover penalty); these are NOT
    exposed in the observation.

    Observation (59-D float32 in [-1, 1]):
      L1 External State (15): robot_pos(2), goal_pos(2), dist_to_goal(1),
        egocentric 3x3 occupancy excl. center (8, oob=1), heading sin/cos(2).
      L2 Micro (21): last-5 actions, rewards, distance-deltas, collision/blocked
        flags, plus a step-level surprise (|TD-proxy|).
      L3 Mezo (11): deterministic sliding-window (W) statistics of L1/L2 history.
      L4 Macro (12): deterministic cross-regime functions; the change-point score
        is a CUSUM on observable statistics, NOT the ground-truth flag.

    Action space: Discrete(5) = {0 up, 1 down, 2 left, 3 right, 4 stay}.

    Reward (clipped to [-10, 10]):
      +1.0 goal; +0.05 per net signed distance reduction (potential-based,
      episode-capped below goal); -0.25 collision; -0.05 repeated blocked
      action; -0.002 per step; -0.1 failure-to-recover after a change event.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    # --- grid / episode constants ---
    GRID_SIZE: int = 12
    MAX_STEPS: int = 200
    COLLISION_BUDGET: int = 5
    N_STATIC_OBSTACLES: int = 10
    N_DYNAMIC_OBSTACLES: int = 3

    # --- buffer / window constants ---
    HIST_LEN: int = 5
    WINDOW_W: int = 20
    RECOVERY_WINDOW: int = 15

    # --- action constants ---
    N_ACTIONS: int = 5
    ACTION_DELTAS: tuple[tuple[int, int], ...] = (
        (0, -1),  # 0 up
        (0, 1),   # 1 down
        (-1, 0),  # 2 left
        (1, 0),   # 3 right
        (0, 0),   # 4 stay
    )

    # --- reward constants ---
    R_GOAL: float = 1.0
    R_PROGRESS_SCALE: float = 0.05
    R_COLLISION: float = -0.25
    R_REPEATED_BLOCK: float = -0.05
    R_STEP: float = -0.002
    R_FAIL_RECOVER: float = -0.1
    REWARD_CLIP: float = 10.0
    SHAPING_CAP: float = 0.9

    # --- td-proxy / cusum constants ---
    GAMMA: float = 0.99
    VALUE_LR: float = 0.05
    CUSUM_K: float = 0.05
    CUSUM_THRESH: float = 1.0

    # --- regime sequence ---
    REGIMES: tuple[str, ...] = ("static", "dynamic", "blocked", "goal_relocation", "unseen")

    # --- rho mapping: per-step change probability ---
    RHO_LEVELS: dict[str, float] = {
        "very_low": 0.005,
        "low": 0.02,
        "medium": 0.06,
        "high": 0.15,
        "per_step": 1.0,
    }

    OBS_DIM: int = 59

    def __init__(
        self,
        render_mode: str | None = None,
        rho: str | float = "medium",
        regime: str | None = None,
        grid_size: int | None = None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode

        self.grid_size = int(grid_size) if grid_size is not None else self.GRID_SIZE

        if isinstance(rho, str):
            self._rho_prob = float(self.RHO_LEVELS.get(rho, self.RHO_LEVELS["medium"]))
        else:
            self._rho_prob = float(np.clip(rho, 0.0, 1.0))

        self._fixed_regime = regime if regime in self.REGIMES else None

        self.action_space = spaces.Discrete(self.N_ACTIONS)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.OBS_DIM,), dtype=np.float32
        )

        self._diag = float(np.sqrt(2.0) * (self.grid_size - 1))
        self._max_disp = 1.0

        # cross-episode persistent memory (macro scale).
        # These are CLEARED whenever reset() is called with an explicit seed so
        # that identical seeds reproduce identical trajectories (determinism).
        self._visit_memory: dict[bytes, int] = {}
        self._skill_memory: dict[str, float] = {r: 0.0 for r in self.REGIMES}
        self._episode_counter: int = 0
        self._regime_idx: int = 0

        # internal state placeholders (set in reset)
        self._robot = np.zeros(2, dtype=np.int64)
        self._goal = np.zeros(2, dtype=np.int64)
        self._static_obs: set[tuple[int, int]] = set()
        self._dyn_obs: list[np.ndarray] = []
        self._regime: str = self.REGIMES[0]

        self._reset_episode_buffers()

    # ------------------------------------------------------------------
    # persistent (cross-episode) memory reset
    # ------------------------------------------------------------------
    def _reset_persistent_memory(self) -> None:
        self._visit_memory = {}
        self._skill_memory = {r: 0.0 for r in self.REGIMES}
        self._episode_counter = 0
        self._regime_idx = 0

    # ------------------------------------------------------------------
    # buffer / state init
    # ------------------------------------------------------------------
    def _reset_episode_buffers(self) -> None:
        self._actions_buf: deque[int] = deque([0] * self.HIST_LEN, maxlen=self.HIST_LEN)
        self._rewards_buf: deque[float] = deque([0.0] * self.HIST_LEN, maxlen=self.HIST_LEN)
        self._delta_buf: deque[float] = deque([0.0] * self.HIST_LEN, maxlen=self.HIST_LEN)
        self._flag_buf: deque[float] = deque([0.0] * self.HIST_LEN, maxlen=self.HIST_LEN)

        self._win_actions: deque[int] = deque(maxlen=self.WINDOW_W)
        self._win_rewards: deque[float] = deque(maxlen=self.WINDOW_W)
        self._win_deltas: deque[float] = deque(maxlen=self.WINDOW_W)
        self._win_dists: deque[float] = deque(maxlen=self.WINDOW_W)
        self._win_flags: deque[float] = deque(maxlen=self.WINDOW_W)
        self._win_success: deque[float] = deque(maxlen=self.WINDOW_W)
        self._win_cells: deque[bytes] = deque(maxlen=self.WINDOW_W)

        self._step_count: int = 0
        self._collisions: int = 0
        self._blocked_streak: int = 0
        self._prev_dist: float = 0.0
        self._cum_shaping: float = 0.0
        self._heading: float = 0.0
        self._speed: float = 0.0
        self._surprise: float = 0.0
        self._value_est: float = 0.0

        # change-point bookkeeping (ground truth — reward only)
        self._steps_since_change: int = 10_000
        self._max_dist_since_change: float = 0.0
        self._recover_penalised: bool = True
        self._success_at_change: bool = True

        # CUSUM accumulators (observable change detection)
        self._cusum_pos: float = 0.0
        self._cusum_neg: float = 0.0
        self._reward_run_mean: float = 0.0
        self._cp_score: float = 0.0
        self._recovery_rate_est: float = 0.0

        self._regime_onehot = np.zeros(4, dtype=np.float32)

    # ------------------------------------------------------------------
    # layout generation
    # ------------------------------------------------------------------
    def _rand_free_cell(self, blocked: set[tuple[int, int]]) -> np.ndarray:
        for _ in range(500):
            c = self.np_random.integers(0, self.grid_size, size=2)
            if (int(c[0]), int(c[1])) not in blocked:
                return c.astype(np.int64)
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if (x, y) not in blocked:
                    return np.array([x, y], dtype=np.int64)
        return np.zeros(2, dtype=np.int64)

    def _generate_layout(self) -> None:
        self._static_obs = set()
        n_static = self.N_STATIC_OBSTACLES
        if self._regime == "blocked":
            n_static += 5
        elif self._regime == "unseen":
            n_static += int(self.np_random.integers(-3, 6))
        n_static = int(np.clip(n_static, 0, self.grid_size * self.grid_size - 4))

        for _ in range(n_static):
            c = self._rand_free_cell(self._static_obs)
            self._static_obs.add((int(c[0]), int(c[1])))

        self._robot = self._rand_free_cell(self._static_obs)
        block = set(self._static_obs)
        block.add((int(self._robot[0]), int(self._robot[1])))
        self._goal = self._rand_free_cell(block)

        self._dyn_obs = []
        n_dyn = 0
        if self._regime in ("dynamic", "goal_relocation", "unseen"):
            n_dyn = self.N_DYNAMIC_OBSTACLES
            if self._regime == "unseen":
                n_dyn = int(np.clip(n_dyn + self.np_random.integers(-1, 3), 0, 8))
        for _ in range(n_dyn):
            block.add((int(self._goal[0]), int(self._goal[1])))
            c = self._rand_free_cell(block)
            self._dyn_obs.append(c.astype(np.int64))
            block.add((int(c[0]), int(c[1])))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _euclid(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sqrt(float((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)))

    def _is_dynamic(self, x: int, y: int) -> bool:
        for d in self._dyn_obs:
            if int(d[0]) == x and int(d[1]) == y:
                return True
        return False

    def _move_dynamic_obstacles(self) -> None:
        if not self._dyn_obs:
            return
        volatility = 0.5
        if self._regime == "dynamic":
            volatility = 0.9
        elif self._regime == "unseen":
            volatility = float(np.clip(0.4 + self._rho_prob, 0.0, 1.0))
        occupied = set(self._static_obs)
        occupied.add((int(self._robot[0]), int(self._robot[1])))
        occupied.add((int(self._goal[0]), int(self._goal[1])))
        for d in self._dyn_obs:
            occupied.add((int(d[0]), int(d[1])))
        for i, d in enumerate(self._dyn_obs):
            if self.np_random.random() >= volatility:
                continue
            choice = int(self.np_random.integers(0, 5))
            dx, dy = self.ACTION_DELTAS[choice]
            nx = int(d[0]) + dx
            ny = int(d[1]) + dy
            if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
                continue
            if (nx, ny) in occupied:
                continue
            occupied.discard((int(d[0]), int(d[1])))
            self._dyn_obs[i] = np.array([nx, ny], dtype=np.int64)
            occupied.add((nx, ny))

    def _apply_change_event(self) -> None:
        block = set(self._static_obs)
        block.add((int(self._robot[0]), int(self._robot[1])))

        if self._regime == "goal_relocation":
            self._goal = self._rand_free_cell(block)
        elif self._regime == "blocked":
            c = self._rand_free_cell(block)
            self._static_obs.add((int(c[0]), int(c[1])))
        elif self._regime == "dynamic":
            if self._dyn_obs:
                idx = int(self.np_random.integers(0, len(self._dyn_obs)))
                block.add((int(self._goal[0]), int(self._goal[1])))
                self._dyn_obs[idx] = self._rand_free_cell(block).astype(np.int64)
        elif self._regime == "unseen":
            if self.np_random.random() < 0.5:
                self._goal = self._rand_free_cell(block)
            else:
                c = self._rand_free_cell(block)
                self._static_obs.add((int(c[0]), int(c[1])))
        else:
            if self._static_obs:
                removable = list(self._static_obs)
                ridx = int(self.np_random.integers(0, len(removable)))
                self._static_obs.discard(removable[ridx])
                c = self._rand_free_cell(block)
                self._static_obs.add((int(c[0]), int(c[1])))

        # mark ground-truth change-point
        self._steps_since_change = 0
        self._max_dist_since_change = self._euclid(self._robot, self._goal)
        self._recover_penalised = False

    def _update_value_and_surprise(self, reward: float, new_dist: float) -> None:
        norm_dist = new_dist / max(self._diag, 1e-8)
        target = reward + self.GAMMA * (-norm_dist)
        td = target - self._value_est
        self._value_est += self.VALUE_LR * td
        self._surprise = float(np.clip(abs(td), 0.0, 1.0))

    def _push_buffers(
        self,
        action: int,
        reward: float,
        norm_delta: float,
        flag: float,
        new_dist: float,
    ) -> None:
        self._actions_buf.append(int(action))
        self._rewards_buf.append(float(np.clip(reward, -1.0, 1.0)))
        self._delta_buf.append(float(np.clip(norm_delta, -1.0, 1.0)))
        self._flag_buf.append(float(np.clip(flag, 0.0, 1.0)))

        self._win_actions.append(int(action))
        self._win_rewards.append(float(reward))
        self._win_deltas.append(float(np.clip(norm_delta, -1.0, 1.0)))
        self._win_dists.append(float(new_dist))
        self._win_flags.append(float(np.clip(flag, 0.0, 1.0)))
        reached = bool(np.array_equal(self._robot, self._goal))
        self._win_success.append(1.0 if reached else 0.0)
        cell = np.array([int(self._robot[0]), int(self._robot[1])], dtype=np.int64).tobytes()
        self._win_cells.append(cell)
        self._visit_memory[cell] = self._visit_memory.get(cell, 0) + 1

    def _update_cusum(self, reward: float) -> None:
        prev_mean = self._reward_run_mean
        self._reward_run_mean = 0.95 * prev_mean + 0.05 * reward
        dev = reward - prev_mean
        self._cusum_pos = max(0.0, self._cusum_pos + dev - self.CUSUM_K)
        self._cusum_neg = max(0.0, self._cusum_neg - dev - self.CUSUM_K)
        cs = max(self._cusum_pos, self._cusum_neg) / max(self.CUSUM_THRESH, 1e-8)
        self._cp_score = float(np.clip(cs, 0.0, 1.0))
        if self._cp_score >= 1.0:
            self._cusum_pos = 0.0
            self._cusum_neg = 0.0

    # ------------------------------------------------------------------
    # observation construction
    # ------------------------------------------------------------------
    def _build_l1(self) -> np.ndarray:
        gs = float(self.grid_size - 1)
        rx = float(self._robot[0]) / max(gs, 1e-8)
        ry = float(self._robot[1]) / max(gs, 1e-8)
        gx = float(self._goal[0]) / max(gs, 1e-8)
        gy = float(self._goal[1]) / max(gs, 1e-8)
        dist = self._euclid(self._robot, self._goal) / max(self._diag, 1e-8)

        occ: list[float] = []
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                if ox == 0 and oy == 0:
                    continue
                cx = int(self._robot[0]) + ox
                cy = int(self._robot[1]) + oy
                if not (0 <= cx < self.grid_size and 0 <= cy < self.grid_size):
                    occ.append(1.0)
                elif (cx, cy) in self._static_obs or self._is_dynamic(cx, cy):
                    occ.append(1.0)
                else:
                    occ.append(0.0)

        h_sin = float(np.sin(self._heading))
        h_cos = float(np.cos(self._heading))

        l1 = np.array(
            [rx, ry, gx, gy, dist] + occ + [h_sin, h_cos],
            dtype=np.float32,
        )
        return l1

    def _build_l2(self) -> np.ndarray:
        acts = [float(a) / float(self.N_ACTIONS - 1) for a in self._actions_buf]
        rews = [float(r) for r in self._rewards_buf]
        dels = [float(d) for d in self._delta_buf]
        flags = [float(f) for f in self._flag_buf]
        l2 = np.array(acts + rews + dels + flags + [self._surprise], dtype=np.float32)
        return l2

    def _build_l3(self) -> np.ndarray:
        wa = np.array(self._win_actions, dtype=np.float32)
        wr = np.array(self._win_rewards, dtype=np.float32)
        wdist = np.array(self._win_dists, dtype=np.float32)
        wf = np.array(self._win_flags, dtype=np.float32)
        ws = np.array(self._win_success, dtype=np.float32)
        n = len(wr)

        if n >= 2 and np.std(wa) > 1e-6 and np.std(wr) > 1e-6:
            corr = float(np.corrcoef(wa, wr)[0, 1])
        else:
            corr = 0.0
        corr = float(np.clip(corr, -1.0, 1.0))

        if n >= 2:
            t = np.arange(n, dtype=np.float32)
            tm = t - t.mean()
            dm = wdist - wdist.mean()
            denom = float(np.sum(tm * tm))
            slope = float(np.sum(tm * dm) / denom) if denom > 1e-8 else 0.0
            slope_trend = float(np.clip(-slope / max(self._diag, 1e-8) * 5.0, -1.0, 1.0))
        else:
            slope_trend = 0.0

        if n >= 4:
            half = n // 2
            momentum = float(np.mean(wr[half:]) - np.mean(wr[:half]))
        else:
            momentum = 0.0
        momentum = float(np.clip(momentum * 2.0, -1.0, 1.0))

        if n >= 1:
            counts = np.bincount(wa.astype(np.int64), minlength=self.N_ACTIONS).astype(np.float32)
            probs = counts / max(float(counts.sum()), 1e-8)
            ent = -np.sum(probs * np.log(probs + 1e-8))
            max_ent = float(np.log(self.N_ACTIONS))
            consistency = float(np.clip(1.0 - ent / max(max_ent, 1e-8), 0.0, 1.0))
        else:
            consistency = 0.0

        success_rate = float(np.clip(np.mean(ws) if n > 0 else 0.0, 0.0, 1.0))
        collision_rate = float(np.clip(np.mean(wf) if n > 0 else 0.0, 0.0, 1.0))

        if n >= 1:
            min_d = float(np.min(wdist))
            cur_d = float(wdist[-1])
            regression = float(np.clip((cur_d - min_d) / max(self._diag, 1e-8), 0.0, 1.0))
        else:
            regression = 0.0

        unique_cells = len(set(self._win_cells))
        coverage = float(np.clip(unique_cells / max(self.WINDOW_W, 1), 0.0, 1.0))

        repeated_fail = float(np.clip(np.mean(wf[-5:]) if n >= 1 else 0.0, 0.0, 1.0))

        if n >= 2:
            flips = float(np.mean((wa[1:] != wa[:-1]).astype(np.float32)))
        else:
            flips = 0.0
        oscillation = float(np.clip(flips, 0.0, 1.0))

        if self._steps_since_change < self.RECOVERY_WINDOW:
            adapt = float(np.clip(self._steps_since_change / max(self.RECOVERY_WINDOW, 1), 0.0, 1.0))
        else:
            adapt = 1.0

        l3 = np.array(
            [
                corr,
                slope_trend,
                momentum,
                consistency,
                success_rate,
                collision_rate,
                regression,
                coverage,
                repeated_fail,
                oscillation,
                adapt,
            ],
            dtype=np.float32,
        )
        return l3

    def _build_l4(self) -> np.ndarray:
        wf = np.array(self._win_flags, dtype=np.float32)
        wr = np.array(self._win_rewards, dtype=np.float32)
        wdist = np.array(self._win_dists, dtype=np.float32)
        n = len(wf)
        n_dyn = float(len(self._dyn_obs))
        n_static = float(len(self._static_obs))
        clutter = n_static / max(self.grid_size * self.grid_size, 1)
        block_sig = float(np.mean(wf) if n > 0 else 0.0)
        dyn_sig = float(np.clip(n_dyn / max(self.N_DYNAMIC_OBSTACLES, 1), 0.0, 1.0))

        logits = np.array(
            [
                1.0 - clutter - dyn_sig,
                dyn_sig,
                block_sig + clutter,
                self._cp_score,
            ],
            dtype=np.float32,
        )
        logits = logits - np.max(logits)
        exps = np.exp(logits)
        regime_descriptor = (exps / max(float(np.sum(exps)), 1e-8)).astype(np.float32)

        strat_match = float(np.clip(self._skill_memory.get(self._regime, 0.0), 0.0, 1.0))

        cp_det = float(np.clip(self._cp_score, 0.0, 1.0))

        if self._steps_since_change < self.RECOVERY_WINDOW and n > 0:
            recent = wr[-self.RECOVERY_WINDOW:]
            recovery = float(np.clip((np.mean(recent) + 1.0) / 2.0, 0.0, 1.0))
        else:
            recovery = float(np.clip(self._recovery_rate_est, 0.0, 1.0))

        coverage_now = len(set(self._win_cells))
        coverage_ratio = float(np.clip(coverage_now / max(self.WINDOW_W, 1), 0.0, 1.0))
        over_explore = float(np.clip(coverage_ratio, 0.0, 1.0))
        under_explore = float(np.clip(1.0 - coverage_ratio, 0.0, 1.0))

        if n > 0:
            mean_r = float(np.mean(wr))
            std_r = float(np.std(wr))
            risk_adj = float(np.clip(mean_r / max(std_r + 1e-3, 1e-3), -1.0, 1.0))
        else:
            risk_adj = 0.0

        cell = np.array([int(self._robot[0]), int(self._robot[1])], dtype=np.int64).tobytes()
        visits = float(self._visit_memory.get(cell, 0))
        familiarity = float(np.clip(visits / 10.0, 0.0, 1.0))

        avg_skill = float(np.mean(list(self._skill_memory.values()))) if self._skill_memory else 0.0
        retained_skill = float(np.clip(avg_skill, 0.0, 1.0))

        l4 = np.array(
            list(regime_descriptor)
            + [
                strat_match,
                cp_det,
                recovery,
                over_explore,
                under_explore,
                risk_adj,
                familiarity,
                retained_skill,
            ],
            dtype=np.float32,
        )
        return l4

    def _build_observation(self) -> np.ndarray:
        l1 = self._build_l1()
        l2 = self._build_l2()
        l3 = self._build_l3()
        l4 = self._build_l4()
        obs = np.concatenate([l1, l2, l3, l4]).astype(np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        return obs.astype(np.float32)

    # ------------------------------------------------------------------
    # gym API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        # Determinism: when an explicit seed is provided, fully clear all
        # cross-episode persistent state so the same seed reproduces the same
        # trajectory regardless of how many episodes preceded this reset.
        if seed is not None:
            self._reset_persistent_memory()

        if self._fixed_regime is not None:
            self._regime = self._fixed_regime
        else:
            self._regime = self.REGIMES[self._episode_counter % len(self.REGIMES)]
        self._regime_idx = self.REGIMES.index(self._regime)
        self._episode_counter += 1

        self._reset_episode_buffers()
        self._generate_layout()

        self._prev_dist = self._euclid(self._robot, self._goal)
        self._value_est = -self._prev_dist / max(self._diag, 1e-8)

        obs = self._build_observation()
        info: dict[str, Any] = {
            "regime": self._regime,
            "rho": self._rho_prob,
            "reward_components": {
                "goal": 0.0,
                "progress": 0.0,
                "collision": 0.0,
                "repeated_block": 0.0,
                "step": 0.0,
                "fail_recover": 0.0,
            },
        }
        return obs, info

    def step(
        self, action: Any
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        try:
            action = int(action)
        except (TypeError, ValueError):
            action = self.N_ACTIONS - 1
        action = int(np.clip(action, 0, self.N_ACTIONS - 1))

        self._step_count += 1
        self._steps_since_change += 1

        # possibly trigger change event
        if self.np_random.random() < self._rho_prob:
            self._apply_change_event()

        self._move_dynamic_obstacles()

        dx, dy = self.ACTION_DELTAS[action]
        nx = int(self._robot[0]) + dx
        ny = int(self._robot[1]) + dy

        collided = False
        blocked = False
        if action != 4:
            if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
                collided = True
                blocked = True
            elif (nx, ny) in self._static_obs or self._is_dynamic(nx, ny):
                collided = True
                blocked = True
            else:
                self._robot = np.array([nx, ny], dtype=np.int64)
                self._heading = float(np.arctan2(float(dy), float(dx)))
                self._speed = 1.0

        if blocked:
            self._blocked_streak += 1
        else:
            self._blocked_streak = 0

        new_dist = self._euclid(self._robot, self._goal)
        net_reduction = self._prev_dist - new_dist
        norm_delta = float(np.clip(net_reduction, -1.0, 1.0))

        # reward components
        comp_goal = 0.0
        comp_progress = 0.0
        comp_collision = 0.0
        comp_repeat = 0.0
        comp_step = self.R_STEP
        comp_fail = 0.0

        reached = bool(np.array_equal(self._robot, self._goal))
        if reached:
            comp_goal = self.R_GOAL

        shaping = self.R_PROGRESS_SCALE * net_reduction
        if shaping > 0.0:
            allowed = max(0.0, self.SHAPING_CAP - self._cum_shaping)
            shaping = min(shaping, allowed)
            self._cum_shaping += shaping
        comp_progress = shaping

        if collided:
            comp_collision = self.R_COLLISION
            self._collisions += 1

        if blocked and self._blocked_streak >= 2:
            comp_repeat = self.R_REPEATED_BLOCK

        # failure-to-recover: regression after a change event
        if 0 < self._steps_since_change <= self.RECOVERY_WINDOW:
            self._max_dist_since_change = max(self._max_dist_since_change, new_dist)
            if (
                self._steps_since_change == self.RECOVERY_WINDOW
                and not self._recover_penalised
            ):
                if new_dist >= self._max_dist_since_change - 1e-6:
                    comp_fail = self.R_FAIL_RECOVER
                self._recover_penalised = True

        reward = (
            comp_goal
            + comp_progress
            + comp_collision
            + comp_repeat
            + comp_step
            + comp_fail
        )
        reward = float(np.clip(reward, -self.REWARD_CLIP, self.REWARD_CLIP))

        flag = 1.0 if collided else 0.0
        self._update_value_and_surprise(reward, new_dist)
        self._push_buffers(action, reward, norm_delta, flag, new_dist)
        self._update_cusum(reward)

        # update skill memory (macro) toward windowed success/progress
        prog_signal = float(np.clip((norm_delta + 1.0) / 2.0, 0.0, 1.0))
        old = self._skill_memory.get(self._regime, 0.0)
        self._skill_memory[self._regime] = float(
            np.clip(0.99 * old + 0.01 * prog_signal, 0.0, 1.0)
        )
        self._recovery_rate_est = float(
            np.clip(0.95 * self._recovery_rate_est + 0.05 * prog_signal, 0.0, 1.0)
        )

        self._prev_dist = new_dist

        terminated = False
        truncated = False
        if reached:
            terminated = True
        elif self._collisions >= self.COLLISION_BUDGET:
            terminated = True
        if self._step_count >= self.MAX_STEPS:
            truncated = True

        obs = self._build_observation()
        info: dict[str, Any] = {
            "regime": self._regime,
            "steps_since_change": int(self._steps_since_change),
            "collisions": int(self._collisions),
            "cp_score": float(self._cp_score),
            "reward_components": {
                "goal": float(comp_goal),
                "progress": float(comp_progress),
                "collision": float(comp_collision),
                "repeated_block": float(comp_repeat),
                "step": float(comp_step),
                "fail_recover": float(comp_fail),
            },
        }

        return obs, reward, bool(terminated), bool(truncated), info

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for (ox, oy) in self._static_obs:
            grid[oy][ox] = "#"
        for d in self._dyn_obs:
            grid[int(d[1])][int(d[0])] = "o"
        gx, gy = int(self._goal[0]), int(self._goal[1])
        grid[gy][gx] = "G"
        rx, ry = int(self._robot[0]), int(self._robot[1])
        grid[ry][rx] = "A"
        lines = ["".join(row) for row in grid]
        out = "\n".join(lines)
        return out

    def close(self) -> None:
        return None
