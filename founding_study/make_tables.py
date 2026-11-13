"""Rebuild the paper's Table 2 and Table 3 from the recorded runs.

    python make_tables.py

Reads results/runs.csv and results/rho_grid_result.json, writes results/table2.md
and results/table3.md, and prints both. Standard deviations are sample standard
deviations (ddof = 1) across runs, as in the paper.
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
LABEL = {"baseline": "Baseline (L1)", "control": "Control (L1-L3)", "full": "Full (L1-L4)"}
ALGO = {"PPO": "PPO", "RECURRENTPPO": "RecurrentPPO"}


def cell_line(name, algo, runs):
    ret = np.array([float(r["eval_mean_return"]) for r in runs])
    suc = 100 * np.mean([float(r["eval_success_rate"]) for r in runs])
    sd = ret.std(ddof=1) if len(ret) > 1 else 0.0
    ids = ", ".join(r["run_id"] for r in runs)
    return f"| {name} | {ALGO[algo]} | {len(runs)} | {ret.mean():.3f} ± {sd:.3f} | {suc:.1f} | {ids} |"


def table2() -> str:
    rows = [r for r in csv.DictReader(open(os.path.join(RES, "runs.csv")))
            if r["used_in_paper_table2"] == "yes"]
    cells = defaultdict(list)
    for r in rows:
        cells[(r["algorithm"], r["condition"])].append(r)
    head = "| Condition | Architecture | n | Return (mean ± sd) | Success (%) | run ids |\n|---|---|---|---|---|---|"
    a = ["**(a) As printed in the paper: every completed 100k-step run of the cell**", "", head]
    b = ["**(b) Default-setting runs only (no rho argument; 3 seeds per cell)**", "", head]
    for algo in ("PPO", "RECURRENTPPO"):
        for cond in ("baseline", "control", "full"):
            runs = cells[(algo, cond)]
            a.append(cell_line(LABEL[cond], algo, runs))
            b.append(cell_line(LABEL[cond], algo, [r for r in runs if r["rho_kwarg"] == ""]))
    extra = ["**Full / PPO runs by rho argument** (these six sweep runs are pooled into (a))", "",
             "| rho argument | n | Return (mean) | Success (%) | run ids |", "|---|---|---|---|---|"]
    by_rho = defaultdict(list)
    for r in cells[("PPO", "full")]:
        by_rho[r["rho_kwarg"] or "default"].append(r)
    for k, runs in by_rho.items():
        ret = np.mean([float(r["eval_mean_return"]) for r in runs])
        suc = 100 * np.mean([float(r["eval_success_rate"]) for r in runs])
        extra.append(f"| {k} | {len(runs)} | {ret:.3f} | {suc:.1f} | {', '.join(r['run_id'] for r in runs)} |")
    return "\n".join(a + [""] + b + [""] + extra) + "\n"


def table3() -> str:
    d = json.load(open(os.path.join(RES, "rho_grid_result.json")))
    out = ["| rho | Baseline | Control | Full | full - baseline | full - control |", "|---|---|---|---|---|---|"]
    for r in d["rows"]:
        out.append(f"| {r['rho']:g} | {r['baseline']:.3f} | {r['control']:.3f} | {r['treatment']:.3f} | "
                   f"{r['gap_treat_baseline']:.3f} | {r['gap_treat_control']:.3f} |")
    out += ["", "Standard deviation over the 10 evaluation episodes of each cell:", "",
            "| rho | Baseline | Control | Full |", "|---|---|---|---|"]
    for rho in d["config"]["rhos"]:
        k = str(float(rho))
        out.append(f"| {rho:g} | " + " | ".join(f"{d['results'][c][k]['std']:.3f}" for c in ("baseline", "control", "treatment")) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    t2, t3 = table2(), table3()
    open(os.path.join(RES, "table2.md"), "w").write(t2)
    open(os.path.join(RES, "table3.md"), "w").write(t3)
    print("Table 2\n" + t2 + "\nTable 3\n" + t3)


if __name__ == "__main__":
    main()
