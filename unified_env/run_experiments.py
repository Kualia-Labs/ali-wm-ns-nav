"""One-command rerun of Experiment 1 under the unified protocol.

Usage:
    pip install gymnasium stable-baselines3 sb3-contrib matplotlib
    python run_experiments.py                    # full: 3 cond x 2 algo x 3 seeds, 100k steps (~4h CPU)
    python run_experiments.py --seeds 0,1,2,3,4  # more seeds
    python run_experiments.py --quick            # 3-minute smoke test of the whole pipeline
    python run_experiments.py --skip-recurrent   # feedforward cells only (~30 min)

Outputs (in --out, default ./results):
    results.json               all runs, machine-readable
    table.md                   paper-ready Table (mean +/- sd, n, success %)
    fig1_training_curves.pdf   paper-ready Figure 1 (and .png)
    curves/*.monitor.csv       raw per-episode training logs
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from nsnav_unified import make_env, CONDITIONS


def train_one(algo_name: str, condition: str, seed: int, steps: int,
              rho: float, out: Path):
    from stable_baselines3.common.monitor import Monitor
    curve_dir = out / "curves"
    curve_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{condition}_{algo_name}_s{seed}"
    env = Monitor(make_env(condition, rho=rho, seed=seed),
                  filename=str(curve_dir / tag),
                  info_keywords=("is_success",))

    common = dict(learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                  n_epochs=10, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
                  seed=seed, verbose=0,
                  policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])))
    if algo_name == "PPO":
        from stable_baselines3 import PPO
        model = PPO("MlpPolicy", env, n_steps=2048, batch_size=64, **common)
    elif algo_name == "RecurrentPPO":
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO("MlpLstmPolicy", env, n_steps=128, batch_size=128,
                             **common)
    else:
        raise ValueError(algo_name)

    t0 = time.time()
    model.learn(total_timesteps=steps, progress_bar=False)
    dur = time.time() - t0

    # ---- evaluation: 10 deterministic episodes on a fresh env ----
    eval_env = make_env(condition, rho=rho, seed=10_000 + seed)
    returns, successes = [], []
    for _ in range(10):
        obs, _ = eval_env.reset()
        state = None
        episode_start = np.ones((1,), dtype=bool)
        done = False
        ep_ret, succ = 0.0, False
        while not done:
            action, state = model.predict(obs, state=state,
                                          episode_start=episode_start,
                                          deterministic=True)
            obs, r, term, trunc, info = eval_env.step(int(action))
            episode_start = np.array([term or trunc])
            ep_ret += float(r)
            succ = succ or bool(info.get("is_success", False))
            done = term or trunc
        returns.append(ep_ret)
        successes.append(1.0 if succ else 0.0)
    eval_env.close()
    env.close()
    return {"condition": condition, "algo": algo_name, "seed": seed,
            "steps": steps, "mean_return": float(np.mean(returns)),
            "success_rate": float(np.mean(successes)),
            "train_seconds": round(dur, 1)}


def write_table(results: list[dict], out: Path) -> str:
    lines = ["| Condition | Architecture | n | Return (mean ± sd) | Success (%) |",
             "|---|---|---|---|---|"]
    order = [("baseline", "PPO"), ("control", "PPO"), ("treatment", "PPO"),
             ("baseline", "RecurrentPPO"), ("control", "RecurrentPPO"),
             ("treatment", "RecurrentPPO")]
    for cond, algo in order:
        cell = [r for r in results if r["condition"] == cond and r["algo"] == algo]
        if not cell:
            continue
        rets = np.array([r["mean_return"] for r in cell])
        succ = 100.0 * float(np.mean([r["success_rate"] for r in cell]))
        sd = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0
        lines.append(f"| {cond.capitalize()} | {algo} | {len(cell)} | "
                     f"{np.mean(rets):.3f} ± {sd:.3f} | {succ:.1f} |")
    text = "\n".join(lines)
    (out / "table.md").write_text(text)
    return text


def make_curve_figure(out: Path) -> None:
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def load(tag):
        p = out / "curves" / f"{tag}.monitor.csv"
        if not p.exists():
            return None
        steps_c, rews = [], []
        with open(p) as f:
            rows = list(csv.reader(f))[2:]
        cum = 0
        for row in rows:
            if len(row) < 2:
                continue
            r, l = float(row[0]), int(float(row[1]))
            cum += l
            steps_c.append(cum)
            rews.append(r)
        return np.array(steps_c), np.array(rews)

    colors = {"baseline": "0.45", "control": "#2ca02c", "treatment": "#d62728"}
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
    for ax, algo in zip(axes, ["PPO", "RecurrentPPO"]):
        for cond in CONDITIONS:
            grids, curves = [], []
            for f in sorted((out / "curves").glob(f"{cond}_{algo}_s*.monitor.csv")):
                data = load(f.stem.replace(".monitor", ""))
                if data is None or len(data[0]) < 5:
                    continue
                s, r = data
                w = max(1, min(20, len(r) // 5))
                r_s = np.convolve(r, np.ones(w) / w, mode="valid")
                s_s = s[w - 1:]
                grids.append(s_s)
                curves.append(r_s)
            if not curves:
                continue
            gmax = min(g[-1] for g in grids)
            grid = np.linspace(min(g[0] for g in grids), gmax, 200)
            interp = np.stack([np.interp(grid, g, c) for g, c in zip(grids, curves)])
            m = interp.mean(axis=0)
            ax.plot(grid, m, color=colors[cond], lw=1.8, label=cond)
            if interp.shape[0] > 1:
                ax.fill_between(grid, interp.min(axis=0), interp.max(axis=0),
                                color=colors[cond], alpha=0.15, lw=0)
        ax.set_title(algo)
        ax.set_xlabel("environment steps")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("episode reward (rolling mean)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig1_training_curves.pdf", bbox_inches="tight")
    fig.savefig(out / "fig1_training_curves.png", dpi=160, bbox_inches="tight")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--rho", type=float, default=0.06)
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument("--quick", action="store_true",
                    help="tiny smoke test of the full pipeline")
    ap.add_argument("--skip-recurrent", action="store_true")
    args = ap.parse_args()

    steps = 3_000 if args.quick else args.steps
    seeds = [0] if args.quick else [int(s) for s in args.seeds.split(",")]
    algos = ["PPO"] if args.skip_recurrent else ["PPO", "RecurrentPPO"]
    if args.quick:
        algos = ["PPO", "RecurrentPPO"]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    todo = [(c, a, s) for a in algos for c in CONDITIONS for s in seeds]
    print(f"[plan] {len(todo)} runs x {steps} steps  (rho={args.rho})")
    for i, (cond, algo, seed) in enumerate(todo, 1):
        rsteps = (1_500 if algo == "RecurrentPPO" else 3_000) if args.quick else steps
        print(f"[{i}/{len(todo)}] {cond} / {algo} / seed {seed} ...", flush=True)
        res = train_one(algo, cond, seed, rsteps, args.rho, out)
        print(f"    return {res['mean_return']:+.3f}  success {res['success_rate']:.0%}"
              f"  ({res['train_seconds']}s)", flush=True)
        results.append(res)
        (out / "results.json").write_text(json.dumps(results, indent=2))

    print("\n" + write_table(results, out))
    make_curve_figure(out)
    print(f"\n[done] outputs in {out}/ : results.json, table.md, "
          f"fig1_training_curves.pdf")


if __name__ == "__main__":
    main()
