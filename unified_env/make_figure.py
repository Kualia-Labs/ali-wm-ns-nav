"""Two-panel volatility figure for the paper (Section 7).

Values are the measured outputs of volatility.py (fast: mean per-step |df|)
and volatility2.py (slow: sd of 200-step block means), seed=0, 20k steps/rho,
uniform-random policy on the released treatment environment.

Release note: volatility.py and volatility2.py were not recovered. The numbers
below are the ones plotted in the paper; measure_timescales.py re-implements the
measurement and reproduces them to within a few percent.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "legend.fontsize": 8})

rho = [0.0, 0.25, 0.5, 0.75, 1.0]

fast = {
    "L1 external": [0.1403, 0.1452, 0.1539, 0.1598, 0.1634],
    "L2 micro":    [0.2879, 0.3016, 0.3174, 0.3286, 0.3437],
    "L3 meso":     [0.0317, 0.0340, 0.0366, 0.0396, 0.0423],
    "L4 macro":    [0.0211, 0.0222, 0.0230, 0.0244, 0.0257],
    "cp (L4 component)": [0.0400, 0.0465, 0.0533, 0.0594, 0.0654],
}
slow = {
    "L1 external": [0.0883, 0.0840, 0.0838, 0.0867, 0.0890],
    "L2 micro":    [0.0230, 0.0223, 0.0236, 0.0239, 0.0247],
    "L3 meso":     [0.0276, 0.0282, 0.0296, 0.0317, 0.0313],
    "L4 macro":    [0.0374, 0.0366, 0.0392, 0.0406, 0.0404],
    "cp (L4 component)": [0.0230, 0.0248, 0.0314, 0.0341, 0.0370],
}
style = {
    "L1 external":       dict(color="0.55",    marker="o", ls="-",  lw=1.4, ms=4.5),
    "L2 micro":          dict(color="#1f77b4", marker="s", ls="-",  lw=1.4, ms=4.5),
    "L3 meso":           dict(color="#2ca02c", marker="^", ls="-",  lw=1.4, ms=5.0),
    "L4 macro":          dict(color="#d62728", marker="D", ls="-",  lw=2.2, ms=5.0),
    "cp (L4 component)": dict(color="#d62728", marker="*", ls=":",  lw=1.8, ms=9.0),
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.2))

for k in fast:
    ax1.plot(rho, fast[k], label=k, **style[k])
ax1.set_yscale("log")
ax1.set_xlabel(r"change rate $\rho$")
ax1.set_ylabel(r"mean per-step $|\Delta f|$  (log)")
ax1.set_title("Fast timescale (per-step jitter)")
ax1.grid(alpha=0.3, which="both")

for k in slow:
    ax2.plot(rho, slow[k], label=k, **style[k])
ax2.set_xlabel(r"change rate $\rho$")
ax2.set_ylabel("sd of 200-step block means")
ax2.set_title("Slow timescale (block-mean drift)")
ax2.grid(alpha=0.3)

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
           bbox_to_anchor=(0.5, -0.06))
fig.tight_layout()
fig.savefig("fig_volatility.pdf", bbox_inches="tight")
print("figure written")
