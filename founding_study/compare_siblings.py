"""Measure how the three sibling environments differ besides their observation layers.

    python compare_siblings.py

The three conditions of the founding study were three separately written
environment classes. This script runs each under a uniform-random policy and
reports the realised change-event rate per step, the obstacle counts at reset and
the random-policy return, at each sibling's built-in default (the setting of
Table 2) and at the rho values of the sweep (Table 3).
"""
from __future__ import annotations

import numpy as np

from common import load_env_class

STEPS = 20_000
# (attribute that fires a change event, True when the method's return value signals it)
HOOK = {"full": ("_apply_change_event", False),
        "baseline": ("_maybe_trigger_change", True),
        "control": ("_apply_change", False)}
STATIC = {"full": "_static_obs", "baseline": "static_obstacles", "control": "static_obs"}


def n_dynamic(env, cond: str) -> int:
    if cond == "full":
        return len(env._dyn_obs)
    if cond == "baseline":
        return len(env.dynamic_obstacles)
    return int((env.dynamic_obs[:, 0] >= 0).sum())


def measure(cond: str, rho):
    cls = load_env_class(cond)
    env = cls() if rho is None else cls(rho=rho)
    name, by_return = HOOK[cond]
    count = {"n": 0}
    orig = getattr(env, name)

    def wrapped(*a, **k):
        out = orig(*a, **k)
        if (not by_return) or out:
            count["n"] += 1
        return out

    setattr(env, name, wrapped)
    rng = np.random.default_rng(0)
    env.reset(seed=0)
    statics, dyns, rets, ep_ret, regimes = [], [], [], 0.0, []
    statics.append(len(getattr(env, STATIC[cond]))); dyns.append(n_dynamic(env, cond))
    for _ in range(STEPS):
        _, r, term, trunc, info = env.step(int(rng.integers(0, 5)))
        ep_ret += float(r)
        if term or trunc:
            rets.append(ep_ret); ep_ret = 0.0
            _, info = env.reset()
            regimes.append(info.get("regime"))
            statics.append(len(getattr(env, STATIC[cond]))); dyns.append(n_dynamic(env, cond))
    env.close()
    return count["n"] / STEPS, float(np.mean(statics)), float(np.mean(dyns)), float(np.mean(rets)), regimes[:10]


def main() -> None:
    print(f"{'setting':<10} {'condition':<9} {'change events/step':>19} {'static obst.':>13} {'dynamic obst.':>14} {'random return':>14}")
    for rho in (None, 0.0, 0.25, 0.5, 0.75, 1.0):
        for cond in ("baseline", "control", "full"):
            rate, st, dy, ret, _ = measure(cond, rho)
            label = "default" if rho is None else f"rho={rho:g}"
            print(f"{label:<10} {cond:<9} {rate:>19.4f} {st:>13.2f} {dy:>14.2f} {ret:>14.3f}")
        print()
    print("first regimes after reset (default setting):")
    for cond in ("baseline", "control", "full"):
        print(f"  {cond:<9}", measure(cond, None)[4])


if __name__ == "__main__":
    main()
