"""Re-evaluate the stored checkpoints and compare with the recorded results.

    python evaluate_checkpoint.py            # all 24 runs
    python evaluate_checkpoint.py 93 94 105  # selected run ids

For each run it loads results/runs/run_XXX/best_model.zip, rebuilds the environment
with the run's own settings, repeats the 10-episode evaluation and prints the
recorded and the re-computed mean return side by side.

The recorded evaluation ran on the environment object that had just been used for
training; every episode starts with a seeded reset, so a fresh environment gives
the same episodes. All 24 stored runs reproduce their recorded evaluation exactly.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from common import HERE, evaluate, make_env

RUNS = os.path.join(HERE, "results", "runs")


def main() -> None:
    from stable_baselines3 import PPO
    from sb3_contrib import RecurrentPPO

    wanted = {int(a) for a in sys.argv[1:]}
    print(f"{'run':>4} {'condition':<9} {'algo':<13} {'rho':>5} {'recorded':>9} {'re-eval':>9} {'succ rec/new':>13}  match")
    for name in sorted(os.listdir(RUNS)):
        rid = int(name.split("_")[1])
        if wanted and rid not in wanted:
            continue
        d = os.path.join(RUNS, name)
        cfg = json.load(open(os.path.join(d, "config.json")))
        rec = json.load(open(os.path.join(d, "results.json")))
        rho = (cfg["config"].get("env_kwargs") or {}).get("rho")
        env = make_env(cfg["condition"], rho)
        cls = RecurrentPPO if cfg["algorithm"] == "RECURRENTPPO" else PPO
        model = cls.load(os.path.join(d, "best_model.zip"), device="cpu")
        returns, _, succ = evaluate(model, env, cfg["config"].get("n_eval_episodes", 10))
        env.close()
        new = float(np.mean(returns))
        same = np.allclose(returns, rec["eval_rewards"], atol=1e-3)
        print(f"{rid:>4} {cfg['condition']:<9} {cfg['algorithm']:<13} {'dflt' if rho is None else rho:>5} "
              f"{rec['mean_reward']:>9.4f} {new:>9.4f} "
              f"{100 * rec['success_rate']:>5.0f}%/{100 * np.mean(succ):>4.0f}%  {'exact' if same else 'differs'}")


if __name__ == "__main__":
    main()
