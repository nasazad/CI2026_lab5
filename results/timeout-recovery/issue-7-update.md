## Follow-up: why the LCS timed out, and how it learned to recover

The learned LCS cut timeouts from **20.50% to 0.98%** on a new held-out test of **2,000 caves and ten training seeds**. It earned **16.14 more points per run** than the same LCS without recovery, returned slightly more gold, and added **0.04 percentage points** to the death rate. The notebook with a planner still had zero timeouts and the highest mean score.

### What caused the timeouts?

I replayed the previous benchmark and audited all **1,793** historical timeouts. Every one repeated both the simulator state and the agent's full memory, so the frozen policy would continue making the same decisions even with a larger move limit. **59.6%** of the cycles moved among several squares; only **two** were repeated `climb` attempts away from the entrance. **1,770** happened before pickup, and in those runs the agent had not seen glitter. The main problem was finding gold without getting stuck, not bringing it home.

### How the LCS recovers

The original rules act unchanged until the agent detects an exact repeat in its **own observations and memory**. Then a separately trained rule table scores all six original commands. On further repeats, the agent samples more broadly from those learned scores, which gives it a chance to try a different move. The repeat trigger selects a rule table, not a command. There is no route search, action mask, hidden map, or expert trajectory. Extra death and timeout costs were used **only in training**; the test uses the original rewards and 120-command limit.

On cave 458, the original policy alternated `climb` and `fwd` against a wall. Both agents took the same first 28 commands. On command 29 the recovery agent turned left instead, eventually picked up the gold, and exited after **43 commands for +947 points**.

![Held-out cave 458: the frozen original repeats two ineffective commands; learned recovery turns away from the wall and exits with gold](https://raw.githubusercontent.com/tomg55555/CI2026_lab5-lcs-transfer/issue-7-timeout-recovery/results/timeout-recovery/held-out-recovery-example.png)

### Held-out comparison

These results use a different split from the previous follow-up. Every row below uses the same new layouts, seed panel, evaluation rewards, and command limit.

| Policy | Timeout | Gold returned | Death | Mean score |
|---|---:|---:|---:|---:|
| Previous LCS, retrained for this split | 20.50% | 30.88% | 3.44% | +237.65 |
| **LCS with cautious recovery** | **0.98%** | **31.07%** | **3.48%** | **+253.80** |
| LCS with a learned recovery selector | 1.88% | 32.25% | 3.72% | +262.08 |
| Notebook with breadth-first search | 0.00% | 31.30% | 2.00% | +282.16 |

![Held-out comparison of timeout rates and mean scores for the LCS variants and notebook planner](https://raw.githubusercontent.com/tomg55555/CI2026_lab5-lcs-transfer/issue-7-timeout-recovery/results/timeout-recovery/held-out-results.png)

The cautious policy kept all **15,901** already-terminating baseline runs unchanged. Of **4,099** baseline timeouts, **3,857** became empty exits, **38** became gold returns, **8** became deaths, and **196** remained. The paired 95% interval for the timeout difference is **[−22.74, −16.69] percentage points**; for the mean-score gain, it is **[+13.03, +19.62]**.

**The planner is not beaten.** The learned selector recovered more gold in this sample, but its gold-rate interval versus the notebook includes zero. It scored **20.08 points less** on average (paired 95% interval **[−37.87, −2.26]**). More deaths explain most of the difference, followed by extra moves and arrows. The repeat trigger cannot undo a death that happens before a loop. The next experiment I would prioritize is a calibrated, separately learned estimate of death risk **before** a loop, followed by a small recurrent, planner-free control at equal training-action budgets.

[Implementation and evaluation branch](https://github.com/tomg55555/CI2026_lab5-lcs-transfer/tree/issue-7-timeout-recovery) · [Report, confidence intervals and traces](https://github.com/tomg55555/CI2026_lab5-lcs-transfer/blob/issue-7-timeout-recovery/results/timeout-recovery/report.md) · [Experiment log and research references](https://github.com/tomg55555/CI2026_lab5-lcs-transfer/blob/issue-7-timeout-recovery/results/timeout-recovery/experiment-log.md)
