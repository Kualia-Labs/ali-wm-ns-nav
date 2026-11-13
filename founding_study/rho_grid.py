"""Change-rate sweep behind Table 3: feedforward PPO, 50k steps, one seed per cell.

    python rho_grid.py                      # 3 conditions x 5 rho values (~20 min on one CPU core)
    GRID_TIMESTEPS=2000 python rho_grid.py  # smoke test

Same procedure as original/_rho_grid.py, which read the environment code from the
platform database; this version reads the same code from envs/. Each cell trains
PPO (Stable-Baselines3 defaults, seed 0) on env_class(rho=rho) and evaluates 10
deterministic episodes on that env with reset(seed=1000..1009).

Writes results_local/rho_grid.json and results_local/rho_grid.md. The recorded
output of the original run is results/rho_grid_result.json.

All three siblings receive the same rho value, but the control sibling maps it to a
different change-event probability (0.15 * rho below rho = 1); compare_siblings.py
measures the resulting rates.
"""
from __future__ import annotations

import json
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import torch

torch.set_num_threads(1)
from stable_baselines3 import PPO

from common import HERE, load_env_class

TIMESTEPS = int(os.getenv("GRID_TIMESTEPS", "50000"))
EVAL_EP = int(os.getenv("GRID_EVAL_EP", "10"))
SEED = 0
RHOS = [0.0, 0.25, 0.5, 0.75, 1.0]
CONDITIONS = ["baseline", "control", "full"]  # same order as the original run


def evaluate(model, env):
    rewards = []
    for ep in range(EVAL_EP):
        obs, _ = env.reset(seed=1000 + ep)
        done = trunc = False
        tot, steps = 0.0, 0
        while not (done or trunc) and steps < 3000:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, _ = env.step(int(action))
            tot += float(r)
            steps += 1
        rewards.append(tot)
    return float(np.mean(rewards)), float(np.std(rewards))


def main() -> None:
    out_dir = os.path.join(HERE, "results_local")
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    t_start = time.time()
    for cond in CONDITIONS:
        cls = load_env_class(cond)
        results[cond] = {}
        for rho in RHOS:
            t0 = time.time()
            env = cls(rho=rho)
            model = PPO("MlpPolicy", env, verbose=0, seed=SEED, device="cpu")
            model.learn(total_timesteps=TIMESTEPS)
            mean_r, std_r = evaluate(model, env)
            env.close()
            results[cond][str(rho)] = {"mean": round(mean_r, 4), "std": round(std_r, 4)}
            print(f"[{cond} rho={rho}] mean={mean_r:.3f} std={std_r:.3f} ({time.time() - t0:.0f}s)", flush=True)

    rows = []
    for rho in RHOS:
        b, c, f = (results[k][str(rho)]["mean"] for k in ("baseline", "control", "full"))
        rows.append({"rho": rho, "baseline": b, "control": c, "full": f,
                     "delta_base": round(f - b, 4), "delta_macro": round(f - c, 4)})
    table = ("| rho | baseline | control | full | full - baseline | full - control |\n|---|---|---|---|---|---|\n"
             + "\n".join(f"| {r['rho']:g} | {r['baseline']:.3f} | {r['control']:.3f} | {r['full']:.3f} | "
                         f"{r['delta_base']:.3f} | {r['delta_macro']:.3f} |" for r in rows))
    out = {"config": {"timesteps": TIMESTEPS, "n_seeds": 1, "seed": SEED, "eval_episodes": EVAL_EP,
                      "rhos": RHOS, "algorithm": "PPO", "torch": torch.__version__},
           "results": results, "rows": rows, "total_time_sec": round(time.time() - t_start, 1)}
    with open(os.path.join(out_dir, "rho_grid.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    with open(os.path.join(out_dir, "rho_grid.md"), "w") as fh:
        fh.write(f"### Return vs rho (PPO, {TIMESTEPS} steps, 1 seed)\n\n{table}\n")
    print(f"\ndone in {time.time() - t_start:.0f}s\n{table}")


if __name__ == "__main__":
    main()
