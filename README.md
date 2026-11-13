# Code and data for "Adaptive Learning Intelligence: A World Model Without Pretraining, or Post-Training as Online Timescale Separation"

This repository holds the environments, training and measurement scripts, raw run
records and trained checkpoints behind the paper's empirical section (Section 3,
Tables 2 and 3, Figure 1, Appendix A).


## Layout

| Folder | What it is |
|---|---|
| `founding_study/` | The three sibling environments, the training and sweep scripts, every recorded run (results, training curves, evaluation replays, checkpoints) and the scripts that rebuild Tables 2 and 3 from them. |
| `unified_env/` | `nsnav_unified.py`: one NS-Nav environment class with a layer-masking wrapper, as described in the paper. `run_experiments.py` reruns the Table 2 design on it. `measure_timescales.py` is the Figure 1 measurement. |

## Paper item to file

| Paper | Produced by | Recorded output |
|---|---|---|
| Table 2 (fixed regime, 100k steps) | `founding_study/train_run.py` | `founding_study/results/runs.csv`, `results/runs/run_090` to `run_113`, `results/table2.md` |
| Table 3 (change-rate sweep, 50k steps) | `founding_study/rho_grid.py` | `founding_study/results/rho_grid_result.json`, `rho_sweep_report.pdf`, `results/table3.md` |
| Figure 1 (two timescales) | `unified_env/measure_timescales.py` | values in `unified_env/make_figure.py`, `fig1_timescales_paper.png` |
| Appendix A (environment, layers, reward) | `unified_env/nsnav_unified.py`; `founding_study/envs/env_43_full.py` | |

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Everything runs on a CPU. The recorded runs used Stable-Baselines3 2.8.0 and
Gymnasium 1.2.3. The checks below were run with Python 3.13, Stable-Baselines3
2.9.2, sb3-contrib 2.9.0, Gymnasium 1.4.0 and PyTorch 2.14.

## Reproduce

```
cd founding_study

# 1. Rebuild Tables 2 and 3 from the recorded runs (seconds)
python make_tables.py

# 2. Re-evaluate the 24 stored checkpoints against their recorded results (seconds)
python evaluate_checkpoint.py

# 3. Train one cell from scratch (PPO about 3 min, RecurrentPPO about 25 min)
python train_run.py --condition full --algo RECURRENTPPO --seed 0

# 4. Rerun the change-rate sweep (about 20 min)
python rho_grid.py

# 5. Measure how the three sibling environments differ (about 1 min)
python compare_siblings.py
```

```
cd unified_env

# Figure 1 measurement (about 1 min)
python measure_timescales.py --plot

# Table 2 design on the single-class environment (about 4 h; --quick for a 3-minute smoke test)
python run_experiments.py
```

## What was checked for this release

- Every cell of Tables 2 and 3 was recomputed from the raw run records and matches
  the paper to three decimals (`results/table2.md`, `results/table3.md`).
- All 24 stored checkpoints, re-evaluated with the environment files in this
  repository, return their recorded evaluation episodes exactly.
- Retraining from scratch is not bit-identical across machines and library
  versions, and the sweep has one seed per cell, so a rerun of `rho_grid.py` will
  not return the recorded cells digit for digit.
- The Figure 1 measurement was re-implemented from Appendix A.4 and reproduces the
  plotted values to within a few percent.

## Files kept verbatim

`founding_study/original/` holds the sweep script and report generator as they were
run on the training server (they read environment code from the platform database
and so do not run standalone) and the function that generated the per-run training
script. `train_run.py` and `rho_grid.py` are standalone equivalents.

Checkpoints (`best_model.zip`, 76 MB in total) are ordinary Stable-Baselines3
archives. Delete `founding_study/results/runs/*/best_model.zip` if you want a small
repository; nothing but `evaluate_checkpoint.py` needs them.
