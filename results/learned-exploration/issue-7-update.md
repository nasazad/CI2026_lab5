## Follow-up: learning both navigation and strategy

The new learning classifier system (LCS) returned gold in **34.76%** of runs on unseen caves, close to the notebook with a planner (**35.45%**). It still earned fewer points and timed out more often. Unlike the notebook, this agent learns **which primitive command to execute at every step**; it is not given a route or a rule telling it when to explore or retreat.

### Why focus on exploration?

The earlier primitive LCS usually got home once it picked up the gold, but often got stuck before finding it. More time alone had not fixed loops in the original benchmark. I kept all six commands available and focused on how the agent explores during training and how it learns from its own successful trips. I selected the recipe on a separate 420-cave validation set; revisiting whole training caves and adding a novelty reward did not beat it.

### How the algorithm works

1. **Remember observations, not conclusions.** The agent combines what it senses now (for example, a breeze or glitter), what it is carrying, its last command, and raw clues it remembers relative to where it is facing. It can remember that a square behind it had an entrance signal, but it is not told that the next square is safe or how to get home.
2. **Generate reusable rules.** The learner creates rules from observations it actually encounters. Some conditions use only a few facts, so they apply across several caves. For each of the six commands—move forward, turn left, turn right, grab, shoot, and climb—it adds the strengths of the rules whose conditions match. If two matches contribute `+35` and `+12` to `fwd`, that command scores `+47`. It is selected if that is the highest score (illustrative numbers, not preset rules).
3. **Learn from consequences and replay real successes.** During training, a count bonus favors commands tried less often in similar situations, and random moves drop from 20% to 5%. Each update uses up to 12 observed rewards, plus an estimate of what comes later when the run has not reached a true ending. The agent also replays its own successful trips: if a detour ends at the exact same recorded state, it can be skipped during replay. Replay reinforces a command only when that cleaned sequence ended better than its current estimate predicts. Deaths and empty exits still inform ordinary learning. The training schedule starts with easier random caves, but the agent gets no route, hidden map, or expert demonstrations.
4. **Merge three independently learned tables.** Each learner sees the same training-cave sequence but explores with a different random seed. I average the weights for each condition and command. The result is **one frozen table**, not three agents voting or exploring during the test.

![Training and inference workflow: explore, assign credit, replay successes, merge three learned tables; then score all six primitive commands using matching conditions](https://raw.githubusercontent.com/tomg55555/CI2026_lab5-lcs-transfer/issue-7-learned-exploration/results/learned-exploration/figures/workflow.png)

### A concrete result

On test cave **350**, the agent moved east, backed up, followed the left and top edges to the gold at `(4,4)`, and returned to the entrance. It scored **+966 in 24 commands** (`1000 − 24 − 10`, including the extra cost of firing one arrow). The map below shows the actual trajectory, including the initial detour. It was drawn afterward; the agent never saw the hazard map.

![Two maps from the actual run on held-out cave 350: outbound path with a backtrack to the entrance, and a successful return from the gold at cell 4,4](https://raw.githubusercontent.com/tomg55555/CI2026_lab5-lcs-transfer/issue-7-learned-exploration/results/learned-exploration/figures/held-out-path.png)

### Comparison with the notebook

The notebook deduces safety using written rules and computes routes with breadth-first search. Its learning decides when to risk or retreat; it does not learn movement sequences. Both approaches used the **same 2,000 unseen layouts, five training seeds, original rewards, and 120-command cap**. No policy learned during testing.

| Policy | Gold returned | Mean score | Deaths | Timeouts |
|---|---:|---:|---:|---:|
| Previous primitive LCS | 16.15% | +85.45 | 1.70% | 42.59% |
| Single learned LCS, same total training actions as the averaged model | 32.29% | +239.42 | 4.23% | 23.02% |
| **Averaged learned LCS** | **34.76%** | **+284.49** | **2.97%** | **17.93%** |
| Notebook with planner | 35.45% | +327.90 | 1.55% | 0.00% |

![Bars comparing gold returns and mean original scores: the averaged LCS approaches the notebook in gold returns but trails it in score](https://raw.githubusercontent.com/tomg55555/CI2026_lab5-lcs-transfer/issue-7-learned-exploration/results/learned-exploration/figures/held-out-results.png)

The gold gap was **−0.69 percentage points** (paired 95% interval **−2.50 to +1.25**): close, but not proof of equivalence. The notebook scored **43.41 points more on average** and used about **one eighth as many training actions**. At the same training-action budget, averaging improved the score by **45.07 points** over the single LCS (paired 95% interval **+10.54 to +74.87**); the evidence for a gold-rate gain was less clear. The averaged LCS returned the gold in **99.3%** of runs where it picked it up. Finding it without getting stuck remains the main challenge.

[Implementation diff](https://github.com/tomg55555/CI2026_lab5-lcs-transfer/compare/issue-7-randomized-lcs...issue-7-learned-exploration) · [Full report, traces, and test data](https://github.com/tomg55555/CI2026_lab5-lcs-transfer/blob/issue-7-learned-exploration/results/learned-exploration/report.md) · [Experiment log](https://github.com/tomg55555/CI2026_lab5-lcs-transfer/blob/issue-7-learned-exploration/results/learned-exploration/experiment-log.md)
