"""NS-Nav unified environment + layer-masking wrapper (audit-fixed).

Single-source protocol: ONE environment class computes all four observation
layers; the three experimental conditions (baseline / control / treatment)
are produced by LayerMaskWrapper zero-masking layers IN PLACE.

Audit fixes relative to the pilot treatment sibling:
  1. L3 `adapt`    -> keyed to steps since last CUSUM alarm (was: ground-truth
                      steps_since_change).
  2. L4 `recovery` -> keyed to steps since last CUSUM alarm (same fix).
  3. L4 `strat_match` -> skill memory read AND updated by the argmax of the
                      behavioral regime descriptor (was: true regime label).
  4. L4 `regime_descriptor` -> clutter/dynamics logits from egocentric
                      evidence: windowed mean occupancy and an EMA of
                      occupancy flips observed while the agent stayed
                      (was: global obstacle counts).

Ground-truth change timing (`_steps_since_change`) is used ONLY in the reward
(failure-to-recover penalty), which is byte-identical across conditions.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

REGIMES = ("static", "dynamic", "blocked", "goal_relocation", "unseen")
ACTION_DELTAS = ((0, -1), (0, 1), (-1, 0), (1, 0), (0, 0))
DESCRIPTOR_CLASSES = (0, 1, 2, 3)  # open / dynamic / blocked / changing


class NSNavEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    GRID_SIZE = 12
    MAX_STEPS = 200
    COLLISION_BUDGET = 5
    N_STATIC_OBSTACLES = 10
    N_DYNAMIC_OBSTACLES = 3
    HIST_LEN = 5
    WINDOW_W = 20
    RECOVERY_WINDOW = 15

    R_GOAL = 1.0
    R_PROGRESS_SCALE = 0.05
    R_COLLISION = -0.25
    R_REPEATED_BLOCK = -0.05
    R_STEP = -0.002
    R_FAIL_RECOVER = -0.1
    REWARD_CLIP = 10.0
    SHAPING_CAP = 0.9

    GAMMA = 0.99
    VALUE_LR = 0.05
    CUSUM_K = 0.05
    CUSUM_THRESH = 1.0
    DYN_EVIDENCE_LR = 0.05

    OBS_DIM = 59
    RHO_LEVELS = {"very_low": 0.005, "low": 0.02, "medium": 0.06,
                  "high": 0.15, "per_step": 1.0}

    def __init__(self, rho: str | float = "medium",
                 render_mode: Optional[str] = None,
                 initial_seed: Optional[int] = None) -> None:
        super().__init__()
        self.render_mode = render_mode
        if isinstance(rho, str):
            self._rho = float(self.RHO_LEVELS.get(rho, 0.06))
        else:
            self._rho = float(np.clip(rho, 0.0, 1.0))
        self._pending_seed = initial_seed

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(-1.0, 1.0, (self.OBS_DIM,), np.float32)
        self._diag = float(np.sqrt(2.0) * (self.GRID_SIZE - 1))
        self._reset_persistent()

    # ------------------------------------------------------------ persistent
    def _reset_persistent(self) -> None:
        self._visit: dict[tuple[int, int], int] = {}
        self._skill = {c: 0.0 for c in DESCRIPTOR_CLASSES}  # FIX 3: descriptor-keyed
        self._episode_counter = 0

    # ------------------------------------------------------------ buffers
    def _reset_buffers(self) -> None:
        H, W = self.HIST_LEN, self.WINDOW_W
        self._ab = deque([0] * H, maxlen=H)
        self._rb = deque([0.0] * H, maxlen=H)
        self._db = deque([0.0] * H, maxlen=H)
        self._fb = deque([0.0] * H, maxlen=H)
        self._wa: deque = deque(maxlen=W)
        self._wr: deque = deque(maxlen=W)
        self._wdist: deque = deque(maxlen=W)
        self._wf: deque = deque(maxlen=W)
        self._ws: deque = deque(maxlen=W)
        self._wc: deque = deque(maxlen=W)
        self._w_occmean: deque = deque(maxlen=W)  # FIX 4: egocentric clutter evidence

        self._t = 0
        self._coll = 0
        self._blocked_streak = 0
        self._cum_shaping = 0.0
        self._heading = 0.0
        self._surprise = 0.0
        self._value_est = 0.0

        # reward-side ground truth (NEVER read by feature builders)
        self._steps_since_change = 10 ** 6
        self._max_dist_since_change = 0.0
        self._recover_penalised = True

        # observable change detection
        self._cusum_pos = 0.0
        self._cusum_neg = 0.0
        self._reward_run_mean = 0.0
        self._cp = 0.0
        self._steps_since_alarm = 10 ** 6  # FIX 1/2: behavioral clock
        self._recovery_rate_est = 0.0

        self._dyn_evidence = 0.0          # FIX 4: egocentric motion evidence
        self._prev_occ: Optional[np.ndarray] = None
        self._desc_cache = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)

    # ------------------------------------------------------------ layout
    def _rand_free(self, blocked: set) -> np.ndarray:
        for _ in range(500):
            c = self.np_random.integers(0, self.GRID_SIZE, 2)
            if (int(c[0]), int(c[1])) not in blocked:
                return c.astype(np.int64)
        for x in range(self.GRID_SIZE):
            for y in range(self.GRID_SIZE):
                if (x, y) not in blocked:
                    return np.array([x, y], dtype=np.int64)
        return np.zeros(2, dtype=np.int64)

    def _generate_layout(self) -> None:
        self._static: set = set()
        n = self.N_STATIC_OBSTACLES
        if self._regime == "blocked":
            n += 5
        elif self._regime == "unseen":
            n += int(self.np_random.integers(-3, 6))
        n = int(np.clip(n, 0, self.GRID_SIZE ** 2 - 4))
        for _ in range(n):
            c = self._rand_free(self._static)
            self._static.add((int(c[0]), int(c[1])))
        self._robot = self._rand_free(self._static)
        blk = set(self._static)
        blk.add((int(self._robot[0]), int(self._robot[1])))
        self._goal = self._rand_free(blk)
        self._dyn: list[np.ndarray] = []
        nd = 0
        if self._regime in ("dynamic", "goal_relocation", "unseen"):
            nd = self.N_DYNAMIC_OBSTACLES
            if self._regime == "unseen":
                nd = int(np.clip(nd + self.np_random.integers(-1, 3), 0, 8))
        for _ in range(nd):
            blk.add((int(self._goal[0]), int(self._goal[1])))
            c = self._rand_free(blk)
            self._dyn.append(c.astype(np.int64))
            blk.add((int(c[0]), int(c[1])))

    # ------------------------------------------------------------ helpers
    def _eu(self, a, b) -> float:
        return float(np.hypot(float(a[0] - b[0]), float(a[1] - b[1])))

    def _isdyn(self, x: int, y: int) -> bool:
        return any(int(d[0]) == x and int(d[1]) == y for d in self._dyn)

    def _ego_occ(self) -> np.ndarray:
        occ = []
        rx, ry = int(self._robot[0]), int(self._robot[1])
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                if ox == 0 and oy == 0:
                    continue
                cx, cy = rx + ox, ry + oy
                if not (0 <= cx < self.GRID_SIZE and 0 <= cy < self.GRID_SIZE):
                    occ.append(1.0)
                elif (cx, cy) in self._static or self._isdyn(cx, cy):
                    occ.append(1.0)
                else:
                    occ.append(0.0)
        return np.asarray(occ, dtype=np.float32)

    def _move_dyn(self) -> None:
        if not self._dyn:
            return
        vol = 0.5
        if self._regime == "dynamic":
            vol = 0.9
        elif self._regime == "unseen":
            vol = float(np.clip(0.4 + self._rho, 0.0, 1.0))
        occ = set(self._static)
        occ.add((int(self._robot[0]), int(self._robot[1])))
        occ.add((int(self._goal[0]), int(self._goal[1])))
        for d in self._dyn:
            occ.add((int(d[0]), int(d[1])))
        for i, d in enumerate(self._dyn):
            if self.np_random.random() >= vol:
                continue
            dx, dy = ACTION_DELTAS[int(self.np_random.integers(0, 5))]
            nx, ny = int(d[0]) + dx, int(d[1]) + dy
            if not (0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE):
                continue
            if (nx, ny) in occ:
                continue
            occ.discard((int(d[0]), int(d[1])))
            self._dyn[i] = np.array([nx, ny], dtype=np.int64)
            occ.add((nx, ny))

    def _apply_change(self) -> None:
        blk = set(self._static)
        blk.add((int(self._robot[0]), int(self._robot[1])))
        if self._regime == "goal_relocation":
            self._goal = self._rand_free(blk)
        elif self._regime == "blocked":
            c = self._rand_free(blk)
            self._static.add((int(c[0]), int(c[1])))
        elif self._regime == "dynamic":
            if self._dyn:
                i = int(self.np_random.integers(0, len(self._dyn)))
                blk.add((int(self._goal[0]), int(self._goal[1])))
                self._dyn[i] = self._rand_free(blk)
        elif self._regime == "unseen":
            if self.np_random.random() < 0.5:
                self._goal = self._rand_free(blk)
            else:
                c = self._rand_free(blk)
                self._static.add((int(c[0]), int(c[1])))
        else:
            if self._static:
                lst = list(self._static)
                self._static.discard(lst[int(self.np_random.integers(0, len(lst)))])
                c = self._rand_free(blk)
                self._static.add((int(c[0]), int(c[1])))
        self._steps_since_change = 0
        self._max_dist_since_change = self._eu(self._robot, self._goal)
        self._recover_penalised = False

    def _update_value_surprise(self, r: float, nd: float) -> None:
        tgt = r + self.GAMMA * (-nd / max(self._diag, 1e-8))
        td = tgt - self._value_est
        self._value_est += self.VALUE_LR * td
        self._surprise = float(np.clip(abs(td), 0.0, 1.0))

    def _update_cusum(self, r: float) -> None:
        pm = self._reward_run_mean
        self._reward_run_mean = 0.95 * pm + 0.05 * r
        dev = r - pm
        self._cusum_pos = max(0.0, self._cusum_pos + dev - self.CUSUM_K)
        self._cusum_neg = max(0.0, self._cusum_neg - dev - self.CUSUM_K)
        cs = max(self._cusum_pos, self._cusum_neg) / max(self.CUSUM_THRESH, 1e-8)
        self._cp = float(np.clip(cs, 0.0, 1.0))
        if self._cp >= 1.0:
            self._cusum_pos = 0.0
            self._cusum_neg = 0.0
            self._steps_since_alarm = 0  # FIX 1/2: behavioral alarm clock

    def _compute_descriptor(self) -> np.ndarray:
        """FIX 4: logits from egocentric evidence only."""
        cl = float(np.mean(self._w_occmean)) if self._w_occmean else 0.0
        b = float(np.mean(self._wf)) if self._wf else 0.0
        dh = float(np.clip(self._dyn_evidence, 0.0, 1.0))
        logits = np.array([1.0 - cl - dh, dh, b + cl, self._cp], dtype=np.float32)
        logits = logits - np.max(logits)
        e = np.exp(logits)
        return (e / max(float(np.sum(e)), 1e-8)).astype(np.float32)

    # ------------------------------------------------------------ layers
    def _build_l1(self) -> np.ndarray:
        gs = float(self.GRID_SIZE - 1)
        rx = float(self._robot[0]) / gs
        ry = float(self._robot[1]) / gs
        gx = float(self._goal[0]) / gs
        gy = float(self._goal[1]) / gs
        dist = self._eu(self._robot, self._goal) / max(self._diag, 1e-8)
        occ = self._ego_occ()
        return np.concatenate([
            np.array([rx, ry, gx, gy, dist], dtype=np.float32),
            occ,
            np.array([np.sin(self._heading), np.cos(self._heading)], dtype=np.float32),
        ])

    def _build_l2(self) -> np.ndarray:
        acts = [float(a) / 4.0 for a in self._ab]
        return np.array(acts + list(self._rb) + list(self._db) + list(self._fb)
                        + [self._surprise], dtype=np.float32)

    def _build_l3(self) -> np.ndarray:
        wa = np.array(self._wa, dtype=np.float32)
        wr = np.array(self._wr, dtype=np.float32)
        wdist = np.array(self._wdist, dtype=np.float32)
        wf = np.array(self._wf, dtype=np.float32)
        ws = np.array(self._ws, dtype=np.float32)
        n = len(wr)
        if n >= 2 and np.std(wa) > 1e-6 and np.std(wr) > 1e-6:
            corr = float(np.clip(np.corrcoef(wa, wr)[0, 1], -1, 1))
        else:
            corr = 0.0
        if n >= 2:
            t = np.arange(n, dtype=np.float32)
            tm = t - t.mean()
            dm = wdist - wdist.mean()
            den = float(np.sum(tm * tm))
            slope = float(np.sum(tm * dm) / den) if den > 1e-8 else 0.0
            slope_trend = float(np.clip(-slope / max(self._diag, 1e-8) * 5.0, -1, 1))
        else:
            slope_trend = 0.0
        if n >= 4:
            half = n // 2
            mom = float(np.clip(2.0 * (np.mean(wr[half:]) - np.mean(wr[:half])), -1, 1))
        else:
            mom = 0.0
        if n >= 1:
            counts = np.bincount(wa.astype(np.int64), minlength=5).astype(np.float32)
            p = counts / max(float(counts.sum()), 1e-8)
            ent = -float(np.sum(p * np.log(p + 1e-8)))
            cons = float(np.clip(1.0 - ent / np.log(5.0), 0, 1))
        else:
            cons = 0.0
        succ = float(np.clip(np.mean(ws) if n else 0.0, 0, 1))
        coll = float(np.clip(np.mean(wf) if n else 0.0, 0, 1))
        reg = float(np.clip((float(wdist[-1]) - float(np.min(wdist)))
                            / max(self._diag, 1e-8), 0, 1)) if n else 0.0
        cov = float(np.clip(len(set(self._wc)) / max(self.WINDOW_W, 1), 0, 1))
        repf = float(np.clip(np.mean(wf[-5:]) if n else 0.0, 0, 1))
        osc = float(np.clip(np.mean((wa[1:] != wa[:-1]).astype(np.float32)), 0, 1)) if n >= 2 else 0.0
        # FIX 1: adaptation clock = CUSUM alarm clock, not ground truth
        tau = self._steps_since_alarm
        adapt = float(np.clip(tau / max(self.RECOVERY_WINDOW, 1), 0, 1)) if tau < self.RECOVERY_WINDOW else 1.0
        return np.array([corr, slope_trend, mom, cons, succ, coll, reg, cov,
                         repf, osc, adapt], dtype=np.float32)

    def _build_l4(self) -> np.ndarray:
        wr = np.array(self._wr, dtype=np.float32)
        n = len(wr)
        desc = self._desc_cache
        strat = float(np.clip(self._skill.get(int(np.argmax(desc)), 0.0), 0, 1))  # FIX 3
        cpd = float(np.clip(self._cp, 0, 1))
        tau = self._steps_since_alarm
        if tau < self.RECOVERY_WINDOW and n > 0:  # FIX 2
            rec = float(np.clip((np.mean(wr[-self.RECOVERY_WINDOW:]) + 1.0) / 2.0, 0, 1))
        else:
            rec = float(np.clip(self._recovery_rate_est, 0, 1))
        covr = float(np.clip(len(set(self._wc)) / max(self.WINDOW_W, 1), 0, 1))
        over, under = covr, float(np.clip(1.0 - covr, 0, 1))
        if n > 0:
            risk = float(np.clip(np.mean(wr) / max(float(np.std(wr)) + 1e-3, 1e-3), -1, 1))
        else:
            risk = 0.0
        cell = (int(self._robot[0]), int(self._robot[1]))
        fam = float(np.clip(self._visit.get(cell, 0) / 10.0, 0, 1))
        ret = float(np.clip(np.mean(list(self._skill.values())), 0, 1))
        return np.concatenate([desc, np.array([strat, cpd, rec, over, under,
                                               risk, fam, ret], dtype=np.float32)])

    def _obs(self) -> np.ndarray:
        o = np.concatenate([self._build_l1(), self._build_l2(),
                            self._build_l3(), self._build_l4()]).astype(np.float32)
        o = np.nan_to_num(o, nan=0.0, posinf=1.0, neginf=-1.0)
        return np.clip(o, -1.0, 1.0)

    # ------------------------------------------------------------ gym api
    def reset(self, *, seed: Optional[int] = None,
              options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        if seed is None and self._pending_seed is not None:
            seed = self._pending_seed
            self._pending_seed = None
        super().reset(seed=seed)
        if seed is not None:
            self._reset_persistent()
        self._regime = REGIMES[self._episode_counter % len(REGIMES)]
        self._episode_counter += 1
        self._reset_buffers()
        self._generate_layout()
        self._prev_dist = self._eu(self._robot, self._goal)
        self._value_est = -self._prev_dist / max(self._diag, 1e-8)
        self._prev_occ = self._ego_occ()
        obs = self._obs()
        return obs, {"regime": self._regime, "rho": self._rho, "is_success": False}

    def step(self, action: Any):
        try:
            a = int(action)
        except (TypeError, ValueError):
            a = 4
        a = int(np.clip(a, 0, 4))

        self._t += 1
        self._steps_since_change += 1
        if self._steps_since_alarm < 10 ** 6:
            self._steps_since_alarm += 1

        if self.np_random.random() < self._rho:
            self._apply_change()
        self._move_dyn()

        dx, dy = ACTION_DELTAS[a]
        nx, ny = int(self._robot[0]) + dx, int(self._robot[1]) + dy
        collided = blocked = False
        if a != 4:
            if not (0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE):
                collided = blocked = True
            elif (nx, ny) in self._static or self._isdyn(nx, ny):
                collided = blocked = True
            else:
                self._robot = np.array([nx, ny], dtype=np.int64)
                self._heading = float(np.arctan2(float(dy), float(dx)))
        self._blocked_streak = self._blocked_streak + 1 if blocked else 0

        # FIX 4: egocentric motion evidence, updated on 'stay' steps only
        occ_now = self._ego_occ()
        if a == 4 and self._prev_occ is not None:
            flip = 1.0 if bool(np.any(occ_now != self._prev_occ)) else 0.0
            self._dyn_evidence = ((1.0 - self.DYN_EVIDENCE_LR) * self._dyn_evidence
                                  + self.DYN_EVIDENCE_LR * flip)
        self._prev_occ = occ_now

        nd = self._eu(self._robot, self._goal)
        red = self._prev_dist - nd
        ndl = float(np.clip(red, -1.0, 1.0))
        reached = bool(np.array_equal(self._robot, self._goal))

        r = self.R_STEP
        if reached:
            r += self.R_GOAL
        sh = self.R_PROGRESS_SCALE * red
        if sh > 0.0:
            allowed = max(0.0, self.SHAPING_CAP - self._cum_shaping)
            sh = min(sh, allowed)
            self._cum_shaping += sh
        r += sh
        if collided:
            r += self.R_COLLISION
            self._coll += 1
        if blocked and self._blocked_streak >= 2:
            r += self.R_REPEATED_BLOCK
        # reward-side ground truth ONLY (identical across conditions)
        if 0 < self._steps_since_change <= self.RECOVERY_WINDOW:
            self._max_dist_since_change = max(self._max_dist_since_change, nd)
            if self._steps_since_change == self.RECOVERY_WINDOW and not self._recover_penalised:
                if nd >= self._max_dist_since_change - 1e-6:
                    r += self.R_FAIL_RECOVER
                self._recover_penalised = True
        r = float(np.clip(r, -self.REWARD_CLIP, self.REWARD_CLIP))

        flag = 1.0 if collided else 0.0
        self._update_value_surprise(r, nd)

        self._ab.append(a)
        self._rb.append(float(np.clip(r, -1, 1)))
        self._db.append(ndl)
        self._fb.append(flag)
        self._wa.append(a)
        self._wr.append(float(r))
        self._wdist.append(float(nd))
        self._wf.append(flag)
        self._ws.append(1.0 if reached else 0.0)
        cell = (int(self._robot[0]), int(self._robot[1]))
        self._wc.append(cell)
        self._visit[cell] = self._visit.get(cell, 0) + 1
        self._w_occmean.append(float(np.mean(occ_now)))

        self._update_cusum(r)
        self._desc_cache = self._compute_descriptor()

        # FIX 3: skill memory keyed by behavioral descriptor argmax
        prog = float(np.clip((ndl + 1.0) / 2.0, 0, 1))
        c = int(np.argmax(self._desc_cache))
        self._skill[c] = float(np.clip(0.99 * self._skill[c] + 0.01 * prog, 0, 1))
        self._recovery_rate_est = float(np.clip(0.95 * self._recovery_rate_est
                                                + 0.05 * prog, 0, 1))
        self._prev_dist = nd

        terminated = reached or self._coll >= self.COLLISION_BUDGET
        truncated = self._t >= self.MAX_STEPS
        obs = self._obs()
        info = {"regime": self._regime, "is_success": reached,
                "collisions": int(self._coll), "cp_score": float(self._cp)}
        return obs, r, bool(terminated), bool(truncated), info

    def render(self):
        if self.render_mode != "ansi":
            return None
        g = [["." for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        for (x, y) in self._static:
            g[y][x] = "#"
        for d in self._dyn:
            g[int(d[1])][int(d[0])] = "o"
        g[int(self._goal[1])][int(self._goal[0])] = "G"
        g[int(self._robot[1])][int(self._robot[0])] = "A"
        return "\n".join("".join(row) for row in g)


class LayerMaskWrapper(gym.ObservationWrapper):
    """Zero-masks observation layers IN PLACE (shape never changes)."""

    SLICES = {"L2": slice(15, 36), "L3": slice(36, 47), "L4": slice(47, 59)}

    def __init__(self, env: gym.Env, masked: tuple[str, ...] = ()) -> None:
        super().__init__(env)
        for k in masked:
            if k not in self.SLICES:
                raise ValueError(f"unknown layer {k}")
        self.masked = tuple(masked)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.copy()
        for k in self.masked:
            obs[self.SLICES[k]] = 0.0
        return obs


CONDITIONS = {
    "baseline": ("L2", "L3", "L4"),
    "control": ("L4",),
    "treatment": (),
}


def make_env(condition: str, rho: float = 0.06,
             seed: Optional[int] = None) -> gym.Env:
    return LayerMaskWrapper(NSNavEnv(rho=rho, initial_seed=seed),
                            masked=CONDITIONS[condition])
