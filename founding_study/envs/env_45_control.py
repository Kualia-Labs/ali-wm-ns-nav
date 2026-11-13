import math
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class QTINavMacroAblationControlEnv(gym.Env):
    """Continual non-stationary grid navigation (macro-ablation control sibling).

    QTI-aligned 59-D observation. In this CONTROL variant the L1 External State (15),
    L2 Micro (21), and L3 Mezo (11) layers are populated; the L4 Macro (12) layer is
    PRESENT but ZERO-MASKED IN PLACE (never resized) so the network input shape is
    identical across all siblings.

    Observation layout (indices):
        [0:15)   L1 External State
        [15:36)  L2 Micro
        [36:47)  L3 Mezo
        [47:59)  L4 Macro  (forced to 0.0 in this variant)

    Action space:
        Discrete(5): 0=up, 1=down, 2=left, 3=right, 4=stay

    Reward (byte-identical across siblings):
        +1.0 goal reached
        +0.05 per unit net signed distance-to-goal reduction (potential-based, capped)
        -0.25 collision (also decrements collision budget)
        -0.05 repeated consecutive failed/blocked action
        -0.002 per step
        -0.1 failure-to-recover within fixed window after a change event (once per event)
        Final reward clipped to [-10.0, 10.0].
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    GRID_SIZE: int = 12
    N_STATIC_OBSTACLES: int = 8
    N_DYNAMIC_OBSTACLES: int = 3
    COLLISION_BUDGET: int = 5
    MAX_STEPS: int = 200
    REGIME_COUNT: int = 5

    DEFAULT_RHO: float = 0.3
    RECOVERY_WINDOW: int = 15

    HISTORY_LEN: int = 5
    WINDOW_W: int = 20

    R_GOAL: float = 1.0
    R_PROGRESS_SCALE: float = 0.05
    R_COLLISION: float = -0.25
    R_REPEATED_FAIL: float = -0.05
    R_STEP: float = -0.002
    R_FAIL_RECOVER: float = -0.1
    REWARD_CLIP: float = 10.0
    SHAPING_CAP: float = 0.9

    GAMMA: float = 0.99

    OBS_DIM: int = 59
    L1_DIM: int = 15
    L2_DIM: int = 21
    L3_DIM: int = 11
    L4_DIM: int = 12
    L1_START: int = 0
    L2_START: int = 15
    L3_START: int = 36
    L4_START: int = 47

    N_ACTIONS: int = 5

    def __init__(self, render_mode: Optional[str] = None, rho: float = 0.3) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.rho: float = float(np.clip(rho, 0.0, 1.0))

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.N_ACTIONS)

        self._diag: float = math.sqrt(2.0) * (self.GRID_SIZE - 1)

        self._visitation_memory: dict[tuple[int, int], int] = {}

        self._deltas = np.array(
            [[0, 1], [0, -1], [-1, 0], [1, 0], [0, 0]], dtype=np.int32
        )

        self._init_episode_state()

    # ------------------------------------------------------------------ #
    def _init_episode_state(self) -> None:
        self.robot = np.zeros(2, dtype=np.int32)
        self.goal = np.zeros(2, dtype=np.int32)
        self.static_obs: list[tuple[int, int]] = []
        self.dynamic_obs: np.ndarray = np.full((self.N_DYNAMIC_OBSTACLES, 2), -1, dtype=np.int32)
        self.dynamic_dir: np.ndarray = np.zeros((self.N_DYNAMIC_OBSTACLES, 2), dtype=np.int32)

        self.step_count: int = 0
        self.collision_count: int = 0
        self.regime: int = 0
        self.heading: float = 0.0
        self.prev_dist: float = 0.0
        self.cum_shaping: float = 0.0
        self.consec_fail: int = 0
        self.running_reward_mean: float = 0.0
        self.reward_count: int = 0

        self.h_actions: list[int] = [0] * self.HISTORY_LEN
        self.h_rewards: list[float] = [0.0] * self.HISTORY_LEN
        self.h_dist_deltas: list[float] = [0.0] * self.HISTORY_LEN
        self.h_collision: list[float] = [0.0] * self.HISTORY_LEN

        self.w_actions: list[int] = []
        self.w_rewards: list[float] = []
        self.w_dist: list[float] = []
        self.w_success: list[float] = []
        self.w_collision: list[float] = []
        self.w_cells: list[tuple[int, int]] = []

        self.steps_since_change: int = 9999
        self.recover_penalty_applied: bool = True
        self.success_since_change: bool = True

        self._visitation_memory = {}

    # ------------------------------------------------------------------ #
    def _rand_free_cell(self, occupied: set[tuple[int, int]]) -> tuple[int, int]:
        for _ in range(200):
            c = (
                int(self.np_random.integers(0, self.GRID_SIZE)),
                int(self.np_random.integers(0, self.GRID_SIZE)),
            )
            if c not in occupied:
                return c
        for x in range(self.GRID_SIZE):
            for y in range(self.GRID_SIZE):
                if (x, y) not in occupied:
                    return (x, y)
        return (0, 0)

    def _build_layout(self) -> None:
        occupied: set[tuple[int, int]] = set()

        n_static = self.N_STATIC_OBSTACLES
        if self.regime == 2:
            n_static = self.N_STATIC_OBSTACLES + 4
        elif self.regime == 0:
            n_static = max(2, self.N_STATIC_OBSTACLES - 3)

        self.static_obs = []
        for _ in range(n_static):
            c = self._rand_free_cell(occupied)
            occupied.add(c)
            self.static_obs.append(c)

        rc = self._rand_free_cell(occupied)
        occupied.add(rc)
        self.robot = np.array(rc, dtype=np.int32)

        gc = self._rand_free_cell(occupied)
        for _ in range(20):
            cand = self._rand_free_cell(occupied)
            if abs(cand[0] - rc[0]) + abs(cand[1] - rc[1]) > abs(gc[0] - rc[0]) + abs(gc[1] - rc[1]):
                gc = cand
        occupied.add(gc)
        self.goal = np.array(gc, dtype=np.int32)

        n_dyn = self.N_DYNAMIC_OBSTACLES if self.regime in (1, 3, 4) else 0
        self.dynamic_obs = np.full((self.N_DYNAMIC_OBSTACLES, 2), -1, dtype=np.int32)
        self.dynamic_dir = np.zeros((self.N_DYNAMIC_OBSTACLES, 2), dtype=np.int32)
        for i in range(n_dyn):
            c = self._rand_free_cell(occupied)
            occupied.add(c)
            self.dynamic_obs[i] = np.array(c, dtype=np.int32)
            d = self._deltas[int(self.np_random.integers(0, 4))]
            self.dynamic_dir[i] = d

    def _static_set(self) -> set[tuple[int, int]]:
        return set(self.static_obs)

    def _dynamic_set(self) -> set[tuple[int, int]]:
        s: set[tuple[int, int]] = set()
        for i in range(self.N_DYNAMIC_OBSTACLES):
            if self.dynamic_obs[i, 0] >= 0:
                s.add((int(self.dynamic_obs[i, 0]), int(self.dynamic_obs[i, 1])))
        return s

    def _in_bounds(self, c: tuple[int, int]) -> bool:
        return 0 <= c[0] < self.GRID_SIZE and 0 <= c[1] < self.GRID_SIZE

    def _dist_to_goal(self) -> float:
        return float(np.linalg.norm(self.robot.astype(np.float32) - self.goal.astype(np.float32)))

    def _move_dynamic(self) -> None:
        statics = self._static_set()
        for i in range(self.N_DYNAMIC_OBSTACLES):
            if self.dynamic_obs[i, 0] < 0:
                continue
            if self.np_random.random() < (0.1 + 0.6 * self.rho):
                self.dynamic_dir[i] = self._deltas[int(self.np_random.integers(0, 4))]
            nxt = (
                int(self.dynamic_obs[i, 0] + self.dynamic_dir[i, 0]),
                int(self.dynamic_obs[i, 1] + self.dynamic_dir[i, 1]),
            )
            if (not self._in_bounds(nxt)) or (nxt in statics):
                self.dynamic_dir[i] = -self.dynamic_dir[i]
                continue
            self.dynamic_obs[i] = np.array(nxt, dtype=np.int32)

    def _maybe_change_event(self) -> bool:
        p_change = self.rho * 0.15
        if self.rho >= 0.999:
            p_change = 1.0
        if self.np_random.random() < p_change:
            self._apply_change()
            return True
        return False

    def _apply_change(self) -> None:
        occupied: set[tuple[int, int]] = {tuple(self.robot)}
        if self.regime == 0:
            if self.static_obs:
                idx = int(self.np_random.integers(0, len(self.static_obs)))
                self.static_obs[idx] = self._rand_free_cell(
                    occupied | self._static_set() | {tuple(self.goal)}
                )
        elif self.regime in (1, 4):
            for i in range(self.N_DYNAMIC_OBSTACLES):
                if self.dynamic_obs[i, 0] >= 0:
                    self.dynamic_dir[i] = self._deltas[int(self.np_random.integers(0, 4))]
        elif self.regime == 2:
            c = self._rand_free_cell(occupied | self._static_set() | {tuple(self.goal)})
            self.static_obs.append(c)
            if len(self.static_obs) > self.N_STATIC_OBSTACLES + 6:
                self.static_obs.pop(0)
        elif self.regime == 3:
            self.goal = np.array(
                self._rand_free_cell(occupied | self._static_set()), dtype=np.int32
            )

        self.steps_since_change = 0
        self.recover_penalty_applied = False
        self.success_since_change = False

    # ------------------------------------------------------------------ #
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self._init_episode_state()

        if options is not None and "regime" in options:
            self.regime = int(options["regime"]) % self.REGIME_COUNT
        else:
            self.regime = int(self.np_random.integers(0, self.REGIME_COUNT))

        if options is not None and "rho" in options:
            self.rho = float(np.clip(options["rho"], 0.0, 1.0))

        self._build_layout()

        self.prev_dist = self._dist_to_goal()
        self.heading = 0.0
        self.cum_shaping = 0.0

        obs = self._build_obs(surprise=0.0)
        info = {
            "regime": int(self.regime),
            "rho": float(self.rho),
            "reward_components": {
                "goal": 0.0,
                "progress": 0.0,
                "collision": 0.0,
                "repeated_fail": 0.0,
                "step": 0.0,
                "failure_to_recover": 0.0,
            },
        }
        return obs, info

    # ------------------------------------------------------------------ #
    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict]:
        try:
            a = int(action)
        except (TypeError, ValueError):
            a = 4
        a = int(np.clip(a, 0, self.N_ACTIONS - 1))

        self.step_count += 1
        self.steps_since_change += 1

        self._maybe_change_event()
        self._move_dynamic()

        statics = self._static_set()
        dynamics = self._dynamic_set()

        blocked = False
        collision = False

        delta = self._deltas[a]
        target = (int(self.robot[0] + delta[0]), int(self.robot[1] + delta[1]))

        if a == 4:
            if tuple(self.robot) in dynamics:
                collision = True
        else:
            if (not self._in_bounds(target)) or (target in statics):
                blocked = True
            elif target in dynamics:
                collision = True
            else:
                old = self.robot.copy()
                self.robot = np.array(target, dtype=np.int32)
                dx = float(self.robot[0] - old[0])
                dy = float(self.robot[1] - old[1])
                if dx != 0.0 or dy != 0.0:
                    self.heading = math.atan2(dy, dx)

        if blocked:
            self.consec_fail += 1
        else:
            if not collision:
                self.consec_fail = 0

        new_dist = self._dist_to_goal()
        raw_delta = self.prev_dist - new_dist
        self.prev_dist = new_dist

        progress_reward = 0.0
        if abs(raw_delta) > 1e-9:
            tentative = self.R_PROGRESS_SCALE * raw_delta
            room = self.SHAPING_CAP - self.cum_shaping
            if tentative > 0.0:
                tentative = min(tentative, max(0.0, room))
            progress_reward = tentative
            self.cum_shaping = float(
                np.clip(self.cum_shaping + progress_reward, -self.SHAPING_CAP, self.SHAPING_CAP)
            )

        rc = {
            "goal": 0.0,
            "progress": float(progress_reward),
            "collision": 0.0,
            "repeated_fail": 0.0,
            "step": self.R_STEP,
            "failure_to_recover": 0.0,
        }

        terminated = False
        truncated = False

        if collision:
            rc["collision"] = self.R_COLLISION
            self.collision_count += 1

        if blocked and self.consec_fail >= 2:
            rc["repeated_fail"] = self.R_REPEATED_FAIL

        reached = bool(np.array_equal(self.robot, self.goal))
        if reached:
            rc["goal"] = self.R_GOAL
            terminated = True
            self.success_since_change = True

        if (not self.recover_penalty_applied) and (self.steps_since_change >= self.RECOVERY_WINDOW):
            if not self.success_since_change:
                rc["failure_to_recover"] = self.R_FAIL_RECOVER
            self.recover_penalty_applied = True

        reward = float(sum(rc.values()))
        reward = float(np.clip(reward, -self.REWARD_CLIP, self.REWARD_CLIP))

        if self.collision_count >= self.COLLISION_BUDGET:
            terminated = True

        if self.step_count >= self.MAX_STEPS:
            truncated = True

        self.reward_count += 1
        self.running_reward_mean += (reward - self.running_reward_mean) / float(self.reward_count)
        surprise = abs(reward - self.running_reward_mean)

        self.h_actions.insert(0, a)
        self.h_actions.pop()
        self.h_rewards.insert(0, reward)
        self.h_rewards.pop()
        self.h_dist_deltas.insert(0, raw_delta)
        self.h_dist_deltas.pop()
        self.h_collision.insert(0, 1.0 if (collision or blocked) else 0.0)
        self.h_collision.pop()

        self.w_actions.append(a)
        self.w_rewards.append(reward)
        self.w_dist.append(new_dist)
        self.w_success.append(1.0 if reached else 0.0)
        self.w_collision.append(1.0 if collision else 0.0)
        self.w_cells.append((int(self.robot[0]), int(self.robot[1])))
        cell = (int(self.robot[0]), int(self.robot[1]))
        self._visitation_memory[cell] = self._visitation_memory.get(cell, 0) + 1

        for buf in (self.w_actions, self.w_rewards, self.w_dist, self.w_success, self.w_collision, self.w_cells):
            while len(buf) > self.WINDOW_W:
                buf.pop(0)

        obs = self._build_obs(surprise=surprise)

        info = {
            "regime": int(self.regime),
            "rho": float(self.rho),
            "collision_count": int(self.collision_count),
            "step_count": int(self.step_count),
            "distance_to_goal": float(new_dist),
            "reward_components": rc,
        }

        return obs, reward, bool(terminated), bool(truncated), info

    # ------------------------------------------------------------------ #
    def _build_obs(self, surprise: float) -> np.ndarray:
        obs = np.zeros(self.OBS_DIM, dtype=np.float32)
        gmax = float(self.GRID_SIZE - 1) if self.GRID_SIZE > 1 else 1.0

        # ---- L1 External State (15) ----
        l1 = np.zeros(self.L1_DIM, dtype=np.float32)
        l1[0] = (2.0 * self.robot[0] / gmax) - 1.0
        l1[1] = (2.0 * self.robot[1] / gmax) - 1.0
        l1[2] = (2.0 * self.goal[0] / gmax) - 1.0
        l1[3] = (2.0 * self.goal[1] / gmax) - 1.0
        l1[4] = float(np.clip(self._dist_to_goal() / (self._diag + 1e-9), 0.0, 1.0))

        statics = self._static_set()
        dynamics = self._dynamic_set()
        occ_cells = statics | dynamics
        offsets = [(-1, 1), (0, 1), (1, 1), (-1, 0), (1, 0), (-1, -1), (0, -1), (1, -1)]
        for k, (ox, oy) in enumerate(offsets):
            cx = int(self.robot[0] + ox)
            cy = int(self.robot[1] + oy)
            if not self._in_bounds((cx, cy)):
                l1[5 + k] = 1.0
            elif (cx, cy) in occ_cells:
                l1[5 + k] = 1.0
            else:
                l1[5 + k] = 0.0
        l1[13] = float(math.sin(self.heading))
        l1[14] = float(math.cos(self.heading))

        # ---- L2 Micro (21) ----
        l2 = np.zeros(self.L2_DIM, dtype=np.float32)
        for i in range(self.HISTORY_LEN):
            l2[i] = (2.0 * self.h_actions[i] / float(self.N_ACTIONS - 1)) - 1.0
        for i in range(self.HISTORY_LEN):
            l2[5 + i] = float(np.clip(self.h_rewards[i], -1.0, 1.0))
        for i in range(self.HISTORY_LEN):
            l2[10 + i] = float(np.clip(self.h_dist_deltas[i] / (self.GRID_SIZE - 1.0 + 1e-9), -1.0, 1.0))
        for i in range(self.HISTORY_LEN):
            l2[15 + i] = float(np.clip(self.h_collision[i], 0.0, 1.0))
        l2[20] = float(np.clip(surprise, 0.0, 1.0))

        # ---- L3 Mezo (11) ----
        l3 = np.zeros(self.L3_DIM, dtype=np.float32)
        wn = len(self.w_rewards)
        if wn >= 2:
            acts = np.array(self.w_actions, dtype=np.float32)
            rews = np.array(self.w_rewards, dtype=np.float32)
            dists = np.array(self.w_dist, dtype=np.float32)

            if np.std(acts) > 1e-6 and np.std(rews) > 1e-6:
                corr = float(np.corrcoef(acts, rews)[0, 1])
                if not np.isfinite(corr):
                    corr = 0.0
            else:
                corr = 0.0
            l3[0] = float(np.clip(corr, -1.0, 1.0))

            x = np.arange(wn, dtype=np.float32)
            try:
                slope = float(np.polyfit(x, -dists, 1)[0])
            except Exception:
                slope = 0.0
            if not np.isfinite(slope):
                slope = 0.0
            l3[1] = float(np.clip(slope / (self.GRID_SIZE + 1e-9), -1.0, 1.0))

            half = wn // 2
            recent = float(np.mean(rews[half:])) if wn - half > 0 else 0.0
            older = float(np.mean(rews[:half])) if half > 0 else recent
            l3[2] = float(np.clip(recent - older, -1.0, 1.0))

            counts = np.bincount(acts.astype(np.int32), minlength=self.N_ACTIONS).astype(np.float32)
            p = counts / max(1.0, counts.sum())
            ent = -float(np.sum([pi * math.log(pi + 1e-12) for pi in p]))
            max_ent = math.log(self.N_ACTIONS)
            l3[3] = float(np.clip(1.0 - (ent / (max_ent + 1e-9)), 0.0, 1.0))

            l3[4] = float(np.clip(np.mean(self.w_success), 0.0, 1.0))
            l3[5] = float(np.clip(np.mean(self.w_collision), 0.0, 1.0))

            min_d = float(np.min(dists))
            cur_d = float(dists[-1])
            l3[6] = float(np.clip((cur_d - min_d) / (self.GRID_SIZE + 1e-9), 0.0, 1.0))

            uniq = len(set(self.w_cells))
            l3[7] = float(np.clip(uniq / float(max(1, wn)), 0.0, 1.0))

            l3[8] = float(np.clip(self.consec_fail / float(self.HISTORY_LEN), 0.0, 1.0))

            reversals = 0
            pair = {0: 1, 1: 0, 2: 3, 3: 2}
            ai = acts.astype(np.int32)
            for j in range(1, len(ai)):
                if pair.get(int(ai[j - 1]), -99) == int(ai[j]):
                    reversals += 1
            l3[9] = float(np.clip(reversals / float(max(1, len(ai) - 1)), 0.0, 1.0))

            if self.steps_since_change < 9999:
                adapt = 1.0 if self.success_since_change else float(
                    np.clip(1.0 - self.steps_since_change / float(self.RECOVERY_WINDOW + 1e-9), 0.0, 1.0)
                )
            else:
                adapt = 0.0
            l3[10] = float(np.clip(adapt, 0.0, 1.0))

        # ---- L4 Macro (12) ZERO-MASKED ----
        l4 = np.zeros(self.L4_DIM, dtype=np.float32)

        obs[self.L1_START:self.L1_START + self.L1_DIM] = l1
        obs[self.L2_START:self.L2_START + self.L2_DIM] = l2
        obs[self.L3_START:self.L3_START + self.L3_DIM] = l3
        obs[self.L4_START:self.L4_START + self.L4_DIM] = l4

        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        return obs.astype(np.float32)

    # ------------------------------------------------------------------ #
    def render(self) -> Optional[str]:
        if self.render_mode != "ansi":
            return None
        statics = self._static_set()
        dynamics = self._dynamic_set()
        rows = []
        for y in range(self.GRID_SIZE - 1, -1, -1):
            row = ""
            for x in range(self.GRID_SIZE):
                c = (x, y)
                if c == (int(self.robot[0]), int(self.robot[1])):
                    row += "R"
                elif c == (int(self.goal[0]), int(self.goal[1])):
                    row += "G"
                elif c in dynamics:
                    row += "d"
                elif c in statics:
                    row += "#"
                else:
                    row += "."
            rows.append(row)
        return "\n".join(rows)

    def close(self) -> None:
        return None
