**(a) As printed in the paper: every completed 100k-step run of the cell**

| Condition | Architecture | n | Return (mean ± sd) | Success (%) | run ids |
|---|---|---|---|---|---|
| Baseline (L1) | PPO | 3 | 1.413 ± 0.012 | 100.0 | 96, 97, 98 |
| Control (L1-L3) | PPO | 3 | 1.422 ± 0.001 | 100.0 | 102, 103, 104 |
| Full (L1-L4) | PPO | 9 | 1.234 ± 0.124 | 97.8 | 90, 91, 92, 108, 109, 110, 111, 112, 113 |
| Baseline (L1) | RecurrentPPO | 3 | 0.316 ± 1.309 | 63.3 | 99, 100, 101 |
| Control (L1-L3) | RecurrentPPO | 3 | 1.404 ± 0.040 | 100.0 | 105, 106, 107 |
| Full (L1-L4) | RecurrentPPO | 3 | -1.076 ± 0.445 | 0.0 | 93, 94, 95 |

**(b) Default-setting runs only (no rho argument; 3 seeds per cell)**

| Condition | Architecture | n | Return (mean ± sd) | Success (%) | run ids |
|---|---|---|---|---|---|
| Baseline (L1) | PPO | 3 | 1.413 ± 0.012 | 100.0 | 96, 97, 98 |
| Control (L1-L3) | PPO | 3 | 1.422 ± 0.001 | 100.0 | 102, 103, 104 |
| Full (L1-L4) | PPO | 3 | 1.311 ± 0.001 | 100.0 | 90, 91, 92 |
| Baseline (L1) | RecurrentPPO | 3 | 0.316 ± 1.309 | 63.3 | 99, 100, 101 |
| Control (L1-L3) | RecurrentPPO | 3 | 1.404 ± 0.040 | 100.0 | 105, 106, 107 |
| Full (L1-L4) | RecurrentPPO | 3 | -1.076 ± 0.445 | 0.0 | 93, 94, 95 |

**Full / PPO runs by rho argument** (these six sweep runs are pooled into (a))

| rho argument | n | Return (mean) | Success (%) | run ids |
|---|---|---|---|---|
| default | 3 | 1.311 | 100.0 | 90, 91, 92 |
| 0.0 | 2 | 1.311 | 100.0 | 108, 109 |
| 0.25 | 2 | 1.059 | 90.0 | 110, 111 |
| 0.5 | 2 | 1.215 | 100.0 | 112, 113 |
