"""Figure 1 measurement: two timescales of input variation per observation layer.

    python measure_timescales.py            # prints the table, writes timescales.json
    python measure_timescales.py --plot     # also writes fig1_timescales.png

Protocol (Appendix A.4): uniform-random policy on the unified NS-Nav environment
with all layers visible, 20k steps per change rate, fixed seed.
  fast = mean per-step |delta f| within episodes
  slow = standard deviation of 200-step block means
Layer values average their dimensions; "cp" is the change-point score, one
component of the macro layer.

This is a re-implementation written for the release from the protocol above. The
numbers printed in the paper are kept in make_figure.py; this script prints both
side by side.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from nsnav_unified import NSNavEnv

RHOS = [0.0, 0.25, 0.5, 0.75, 1.0]
LAYERS = {"L1 external": slice(0, 15), "L2 micro": slice(15, 36), "L3 meso": slice(36, 47),
          "L4 macro": slice(47, 59), "cp (L4 component)": slice(52, 53)}
STEPS, BLOCK, SEED = 20_000, 200, 0

# values printed in the paper's Figure 1 (from make_figure.py)
PAPER_FAST = {"L1 external": [0.1403, 0.1452, 0.1539, 0.1598, 0.1634],
              "L2 micro": [0.2879, 0.3016, 0.3174, 0.3286, 0.3437],
              "L3 meso": [0.0317, 0.0340, 0.0366, 0.0396, 0.0423],
              "L4 macro": [0.0211, 0.0222, 0.0230, 0.0244, 0.0257],
              "cp (L4 component)": [0.0400, 0.0465, 0.0533, 0.0594, 0.0654]}
PAPER_SLOW = {"L1 external": [0.0883, 0.0840, 0.0838, 0.0867, 0.0890],
              "L2 micro": [0.0230, 0.0223, 0.0236, 0.0239, 0.0247],
              "L3 meso": [0.0276, 0.0282, 0.0296, 0.0317, 0.0313],
              "L4 macro": [0.0374, 0.0366, 0.0392, 0.0406, 0.0404],
              "cp (L4 component)": [0.0230, 0.0248, 0.0314, 0.0341, 0.0370]}


def rollout(rho: float):
    env = NSNavEnv(rho=rho)
    rng = np.random.default_rng(SEED)
    obs, _ = env.reset(seed=SEED)
    stream, diffs = [obs], []
    for _ in range(STEPS):
        new, _, term, trunc, _ = env.step(int(rng.integers(0, 5)))
        diffs.append(np.abs(new - obs))          # within-episode step
        stream.append(new)
        obs = new
        if term or trunc:
            obs, _ = env.reset()
            stream.append(obs)
    return np.array(stream[:STEPS]), np.array(diffs)


def measure():
    fast = {k: [] for k in LAYERS}
    slow = {k: [] for k in LAYERS}
    for rho in RHOS:
        stream, diffs = rollout(rho)
        blocks = stream.reshape(STEPS // BLOCK, BLOCK, -1).mean(axis=1)
        for k, sl in LAYERS.items():
            fast[k].append(round(float(diffs[:, sl].mean()), 4))
            slow[k].append(round(float(blocks[:, sl].std(axis=0).mean()), 4))
    return fast, slow


def show(title, got, paper):
    print(f"\n{title}\n{'layer':<20}" + "".join(f"{'rho=' + format(r, 'g'):>18}" for r in RHOS))
    for k in LAYERS:
        print(f"{k:<20}" + "".join(f"{g:>9.4f} ({p:.4f})" for g, p in zip(got[k], paper[k])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    fast, slow = measure()
    print("measured (paper value in brackets)")
    show("fast: mean per-step |delta f|", fast, PAPER_FAST)
    show("slow: sd of 200-step block means", slow, PAPER_SLOW)
    json.dump({"rho": RHOS, "fast": fast, "slow": slow, "steps": STEPS, "block": BLOCK, "seed": SEED},
              open("timescales.json", "w"), indent=2)
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        style = {"L1 external": dict(color="0.55", marker="o"), "L2 micro": dict(color="#1f77b4", marker="s"),
                 "L3 meso": dict(color="#2ca02c", marker="^"), "L4 macro": dict(color="#d62728", marker="D", lw=2.2),
                 "cp (L4 component)": dict(color="#d62728", marker="*", ls=":", ms=9)}
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.2))
        for k in LAYERS:
            a1.plot(RHOS, fast[k], label=k, **style[k])
            a2.plot(RHOS, slow[k], label=k, **style[k])
        a1.set_yscale("log"); a1.set_title("Fast timescale (per-step jitter)")
        a2.set_title("Slow timescale (block-mean drift)")
        for a in (a1, a2):
            a.set_xlabel("change rate rho"); a.grid(alpha=0.3, which="both")
        h, lab = a1.get_legend_handles_labels()
        fig.legend(h, lab, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.06))
        fig.tight_layout()
        fig.savefig("fig1_timescales.png", dpi=200, bbox_inches="tight")
        print("\nwrote fig1_timescales.png")


if __name__ == "__main__":
    main()
