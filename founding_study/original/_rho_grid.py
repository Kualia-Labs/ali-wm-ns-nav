"""Standalone uniform rho-grid: train PPO at a fixed budget across
treatment/baseline/control siblings x rho values, compute the
explicit-vs-implicit gap-vs-rho table + monotonicity verdict.
Writes /tmp/rho_grid.json and /tmp/rho_grid.md."""
import os, json, time, types
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import torch
torch.set_num_threads(1)
import gymnasium as gym
from stable_baselines3 import PPO
import sqlite3

TIMESTEPS = int(os.getenv("GRID_TIMESTEPS", "50000"))
EVAL_EP = int(os.getenv("GRID_EVAL_EP", "10"))
SEED = 0
RHOS = [0.0, 0.25, 0.5, 0.75, 1.0]
# role -> env_id (baseline/control first so siblings of equal interest are done early)
ENVS = {"baseline": 44, "control": 45, "treatment": 43}

db = sqlite3.connect("/app/data/bot.db")
db.row_factory = sqlite3.Row


def load_env_class(eid):
    code = db.execute("SELECT code FROM rl_environments WHERE id=?", (eid,)).fetchone()["code"]
    m = types.ModuleType(f"env_{eid}")
    exec(code, m.__dict__)
    for v in m.__dict__.values():
        if isinstance(v, type) and issubclass(v, gym.Env) and v is not gym.Env:
            return v
    raise RuntimeError(f"no gym.Env subclass in env {eid}")


def make_env(C, rho):
    try:
        return C(rho=rho)
    except TypeError:
        return C()


def evaluate(model, env):
    rewards = []
    for ep in range(EVAL_EP):
        obs, _ = env.reset(seed=1000 + ep)
        done = trunc = False
        tot = 0.0
        steps = 0
        while not (done or trunc) and steps < 3000:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, _ = env.step(int(action))
            tot += float(r)
            steps += 1
        rewards.append(tot)
    return float(np.mean(rewards)), float(np.std(rewards))


results = {}  # role -> {rho: {"mean": x, "std": y}}
classes = {role: load_env_class(eid) for role, eid in ENVS.items()}

t_start = time.time()
for role, C in classes.items():
    results[role] = {}
    for rho in RHOS:
        t0 = time.time()
        try:
            env = make_env(C, rho)
            model = PPO("MlpPolicy", env, verbose=0, seed=SEED, device="cpu")
            model.learn(total_timesteps=TIMESTEPS)
            mean_r, std_r = evaluate(model, env)
            env.close()
            results[role][rho] = {"mean": round(mean_r, 4), "std": round(std_r, 4)}
            print(f"[{role} rho={rho}] mean={mean_r:.3f} std={std_r:.3f} ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            results[role][rho] = {"mean": None, "std": None, "error": str(e)}
            print(f"[{role} rho={rho}] ERROR: {e}", flush=True)

# Build gap-vs-rho table
rows = []
gaps = []
for rho in RHOS:
    base = results["baseline"].get(rho, {}).get("mean")
    ctrl = results["control"].get(rho, {}).get("mean")
    treat = results["treatment"].get(rho, {}).get("mean")
    gap_b = (treat - base) if (treat is not None and base is not None) else None
    gap_c = (treat - ctrl) if (treat is not None and ctrl is not None) else None
    if gap_b is not None:
        gaps.append((rho, gap_b))
    f = lambda v: "N/A" if v is None else f"{v:.3f}"
    rows.append({"rho": rho, "baseline": base, "control": ctrl, "treatment": treat,
                 "gap_treat_baseline": gap_b, "gap_treat_control": gap_c,
                 "row_md": f"| {rho:g} | {f(base)} | {f(ctrl)} | {f(treat)} | {f(gap_b)} | {f(gap_c)} |"})

mono = "insufficient data to assess the trend"
verdict = "inconclusive"
if len(gaps) >= 2:
    diffs = [gaps[i + 1][1] - gaps[i][1] for i in range(len(gaps) - 1)]
    if all(d >= -1e-9 for d in diffs):
        verdict = "supported"
        mono = (f"monotonically NON-DECREASING with rho (consistent with H2b): "
                f"gap rises from {gaps[0][1]:.3f} at rho={gaps[0][0]:g} to {gaps[-1][1]:.3f} at rho={gaps[-1][0]:g}")
    elif all(d <= 1e-9 for d in diffs):
        verdict = "contradicted"
        mono = (f"monotonically non-increasing with rho (CONTRADICTS H2b): "
                f"gap falls from {gaps[0][1]:.3f} to {gaps[-1][1]:.3f}")
    else:
        verdict = "partial / non-monotonic"
        mono = (f"non-monotonic in rho (endpoints gap(rho={gaps[0][0]:g})={gaps[0][1]:.3f} -> "
                f"gap(rho={gaps[-1][0]:g})={gaps[-1][1]:.3f})")

table_md = (
    "| rho | baseline (ext-only) | control (micro+mezo) | treatment (full) | gap (treat-baseline) | gap (treat-control) |\n"
    "|---|---|---|---|---|---|\n" + "\n".join(r["row_md"] for r in rows)
)

out = {
    "config": {"timesteps": TIMESTEPS, "n_seeds": 1, "eval_episodes": EVAL_EP, "rhos": RHOS,
               "algorithm": "PPO", "env_ids": ENVS},
    "results": results,
    "rows": rows,
    "gaps_treat_baseline": gaps,
    "monotonicity": mono,
    "h2b_verdict": verdict,
    "table_md": table_md,
    "total_time_sec": round(time.time() - t_start, 1),
}
with open("/tmp/rho_grid.json", "w") as f:
    json.dump(out, f, indent=2)
with open("/tmp/rho_grid.md", "w") as f:
    f.write("### Gap vs rho (PPO, %d steps, 1 seed)\n\n" % TIMESTEPS + table_md +
            "\n\n_H2b trend: %s_\n\n_Verdict: %s_\n" % (mono, verdict))
print("\n=== DONE in %.0fs ===" % (time.time() - t_start), flush=True)
print(table_md, flush=True)
print("\nH2b:", verdict, "|", mono, flush=True)
