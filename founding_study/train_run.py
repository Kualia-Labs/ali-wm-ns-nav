"""Train and evaluate one cell of the founding study.

Standalone equivalent of the training script the platform generated for each run
(see original/platform_build_training_script.py.txt): Stable-Baselines3 defaults,
one environment instance, 10 deterministic evaluation episodes on that same
instance with reset(seed=0..9).

Examples
    python train_run.py --condition full --algo RECURRENTPPO --seed 0
    python train_run.py --condition control --algo PPO --seed 1 --rho 0.25
    python train_run.py --condition baseline --algo PPO --seed 0 --steps 2000   # smoke test

Without --rho each sibling runs at its own built-in default, which is how runs
90-107 (Table 2) were produced. compare_siblings.py prints what those defaults are.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from common import evaluate, make_env


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=["full", "baseline", "control"])
    ap.add_argument("--algo", default="PPO", choices=["PPO", "RECURRENTPPO"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--rho", type=float, default=None)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--out", default=None, help="output directory (default: runs_local/<tag>)")
    args = ap.parse_args()

    # The platform capped each training process at one CPU thread.
    try:
        import torch
        torch.set_num_threads(max(1, int(os.getenv("OMP_NUM_THREADS", "1"))))
    except Exception:
        pass

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    tag = f"{args.condition}_{args.algo}_s{args.seed}" + ("" if args.rho is None else f"_rho{args.rho:g}")
    out = args.out or os.path.join("runs_local", tag)
    os.makedirs(out, exist_ok=True)

    class CurveCallback(BaseCallback):
        """Training curve: mean of the last 50 training episodes, 20 points per run."""

        def __init__(self, every: int):
            super().__init__()
            self.every = every
            self.curve, self.ep_r, self.ep_l = [], [], []
            self._r, self._l = 0.0, 0
            self.t0 = time.time()

        def _on_step(self) -> bool:
            self._r += float(self.locals["rewards"][0])
            self._l += 1
            if self.locals["dones"][0]:
                self.ep_r.append(self._r)
                self.ep_l.append(self._l)
                self._r, self._l = 0.0, 0
            if self.n_calls % self.every == 0 and self.ep_r:
                rr, ll = self.ep_r[-50:], self.ep_l[-50:]
                self.curve.append({
                    "step": self.num_timesteps,
                    "mean_reward": round(float(np.mean(rr)), 4),
                    "min_reward": round(float(np.min(rr)), 4),
                    "max_reward": round(float(np.max(rr)), 4),
                    "mean_ep_length": round(float(np.mean(ll)), 1),
                    "success_rate": round(sum(r > 0 for r in rr) / len(rr), 4),
                    "episodes": len(self.ep_r),
                    "elapsed_sec": round(time.time() - self.t0, 1),
                })
            return True

    env = make_env(args.condition, args.rho)
    if args.algo == "RECURRENTPPO":
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO("MlpLstmPolicy", env, verbose=0, seed=args.seed)
    else:
        model = PPO("MlpPolicy", env, verbose=0, seed=args.seed)

    cb = CurveCallback(every=max(args.steps // 20, 100))
    model.learn(total_timesteps=args.steps, callback=cb, progress_bar=False)
    model.save(os.path.join(out, "best_model"))

    returns, lengths, successes = evaluate(model, env, args.eval_episodes)

    hyper = {}
    for k in ("learning_rate", "gamma", "n_steps", "batch_size", "n_epochs", "ent_coef",
              "vf_coef", "max_grad_norm", "gae_lambda"):
        v = getattr(model, k, None)
        hyper[k] = float(v(1.0)) if callable(v) else v
    hyper["net_arch"] = str(getattr(model.policy, "net_arch", ""))

    import gymnasium
    import stable_baselines3
    results = {
        "status": "completed", "condition": args.condition, "algorithm": args.algo,
        "total_timesteps": args.steps,
        "mean_reward": round(float(np.mean(returns)), 4),
        "std_reward": round(float(np.std(returns)), 4),
        "mean_ep_length": round(float(np.mean(lengths)), 1),
        "success_rate": round(sum(successes) / len(successes), 4),
        "episodes_trained": len(cb.ep_r), "eval_episodes": args.eval_episodes,
        "eval_rewards": [round(r, 4) for r in returns], "eval_lengths": lengths,
        "training_time_sec": round(time.time() - cb.t0, 1),
        "hyperparameters": hyper, "sb3_version": stable_baselines3.__version__,
        "gymnasium_version": gymnasium.__version__, "random_seed": args.seed, "rho": args.rho,
    }
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(out, "curve.json"), "w") as f:
        json.dump(cb.curve, f, indent=2)
    env.close()
    print(f"{tag}: return {results['mean_reward']:.3f} +/- {results['std_reward']:.3f}, "
          f"success {100 * results['success_rate']:.0f}%  -> {out}")


if __name__ == "__main__":
    main()
