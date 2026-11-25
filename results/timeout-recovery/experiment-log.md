# Wumpus LCS: why timeouts happen, and how to recover

This follow-up starts at `9f3ff4c`. The learned policy keeps the six original
primitive commands, the simulator's reward, and the 120-command evaluation
cap. No action filter, route search, inferred safety, hidden map or expert
trajectory is available to the policy. All training episodes start naturally.

## 1. Diagnose the published timeouts

I recreated each of the five published policies and checked the outcome,
score, steps, visits and pickup on every one of the **10,000** published
seed–cave runs. Then I instrumented all **1,793** timeouts offline. A cycle
requires the same simulator state (location, heading, gold, arrow, Wumpus,
transient percepts) **and** the same full policy state (tracked pose, remembered
clues and last command). Matching only a room, action, or observation is not
enough. On a fixed cave the frozen baseline is deterministic; after exact
repetition its future commands and state transitions repeat indefinitely.

| Certified cycle | Runs | Share of timeouts |
|---|---:|---:|
| Moves among multiple squares | 1,069 | 59.62% |
| Stays in one square, mixed commands | 630 | 35.14% |
| Rotates in place | 64 | 3.57% |
| Repeated wall bump | 20 | 1.12% |
| Repeated spent-arrow shot | 8 | 0.45% |
| Repeated `climb` away from the entrance | 2 | 0.11% |
| No exact repeat | 0 | 0.00% |

The median cycle starts at command **27** and repeats every **8** commands
(range 1–30). The latest cycle starts at command **71**, well before 120.
Exactly **1,770 (98.72%)** occur before pickup; none of those 1,770 had seen
glitter. Changing the cap cannot resolve these proven cycles. It would
prolong them. Fixing one ineffective command would not address the dominant
moving and mixed cycles. For instance, on historical cave 18 (seed 0),
the first repeated segment starts at command 4 and just turns right four
times. Other episodes repeat a much longer chain of movements and turns.

[Audit summary](diagnosis/summary.json) · [Per-timeout certificates](diagnosis/timeouts.csv)

## 2. Reduce timeout without trading it for death or lost gold

Freeze the three-table baseline. At the first repeated **observable** state,
activate extra learned rules for the rest of the episode. The repeat key
uses tracked pose, remembered clues, current percepts, last move, whether
the agent fired, and whether it heard the Wumpus die. Unlike the offline
audit, the policy never reads the simulator's actual state. Before the
trigger it executes exactly the original greedy policy; after it, it still
scores all six commands. There is no prescribed escape action. The replayed
outcomes and the fresh panels check that exact-repeat variants never
change a baseline episode that already terminates under the cap.

The recovery rule table estimates discounted returns from the actions it
actually executed after activation; it learns with Monte Carlo updates.
In the cautious profile, the target includes an extra 800-point cost
for a death and a 125-point cost at the cap, **only in training**. During
the activated portion of a frozen run, the adaptive readout samples
commands in proportion to the exponentiated learned scores. Its
temperature starts at 0.020 in 1,000-point reward units and rises by
0.018 for each earlier occurrence of the current state (up to four).
This allows a different action to be sampled on a later visit without
prescribing one or using randomness before the repeat.

The new split excludes all earlier holdouts and reserves **600 validation
and 2,000 final test caves** before training. Proposed validation guardrails:
aim for ≤2% timeouts while not losing over 1 percentage point of gold,
losing over 10 points of original mean score, or adding over 1 percentage
point of deaths versus the same frozen baseline. Report any residual
tradeoff; do not call near-zero "zero." Final comparison pairs by seed and
layout and resamples both axes.

- [Screen 1](screen-1): three seeds. With greedy commands, short action
  history and repeated-state counts beat raw recovery features
  (**5.06% vs. 9.28%** timeout). Sampling from learned recovery action scores
  cut the history variant to **2.17%**; cost-sensitive training cut it to
  **1.89%** at **31.39% gold / 2.28% deaths / +269.13** versus
  **31.33% / 2.11% / +258.62**. Of 317 baseline timeouts, this variant
  converted 279 to empty exits, one to gold, and three to deaths; 34
  remained. It did not change already terminating episodes.
- [Screen 2](screen-2): five seeds and 18,000 additional training caves per
  seed. With the cautious learner, fixed-temperature recovery cut timeouts
  to **1.20%** and increasing the temperature only on recurrent observations
  cut them to **1.23%** at **31.47% gold / 2.13% deaths / +272.41**. The
  baseline was **19.03% timeout / 31.33% gold / 2.03% deaths / +258.61**.
  Killing or forcing a retreat to eliminate a timeout would not be useful;
  death remains a separate metric. The extra death and cap costs only
  train the cautious rule table and never change evaluation rewards.

## 3. Can better recovery beat the planner?

The notebook on the same validation panel returned gold in **31.33%** with
**1.67% deaths, no timeouts, and +286.40**. Matching its aggregate gold rate
does not mean the agents succeed on the same caves: in Screen 1 the notebook
found gold on **117/317** baseline timeout runs but only **10/881** baseline
empty exits. Most potential lies in exploring better after stalls, not
reopening every exit. I did not use these counterfactual outcome labels
as training inputs.

A separate learned death critic plus richer action-history features and
successful self-replay recovered more gold (**34.07%** with the adaptive
readout), but also raised deaths to **4.30%**. Reopening the baseline's
empty-exit decisions allowed **35.60%** gold in the greedy variant and
**8.83%** deaths. A combined novelty/reopening variant reached **39.77%**
gold but **18.67%** deaths: higher gold alone is not progress. These variants
are not selected as safe timeout remedies.
The early-activation and empty-exit variants are not exact-repeat-only;
they may change already-terminating runs. For them, the CSV `recovered_*`
columns count only baseline timeout conversions, not all outcome changes.

- [Screen 3](screen-3): learned a mixture of two recovery tables from
  paired training episodes, each replayed **from the entrance**. The
  router estimates how much the exploratory table improves original score
  and how much it increases death risk, using only the observations the
  policy accumulated by the trigger. With a 30-point minimum predicted gain
  after a penalty for additional estimated death risk
  (`router_30`), it reached **32.77% gold / 2.40% deaths / 2.07% timeout /
  +281.70**. Relative to cautious recovery, gold added 13 points on average,
  deaths lost 2.67, commands lost 1.04. This was 4.70 points short of the
  notebook on validation. Stronger novelty on repeat-only recovery
  (`seek`) did not help the score.
- [Inventory screen](screen-inventory): letting the backbone distinguish
  an available arrow and a heard scream (raw observations) improved gold
  to **32.83%** with recovery, but also increased deaths to **3.43%**;
  **+271.57** was no gain over the published backbone with recovery.
- [Smoothed-target screen](screen-smooth): interpolate 90% of the
  greedy next value with 10% of the mean over all six action values during
  bootstrapping. This changes the training target, not action availability.
  It lowered raw timeouts from **19.03% to 14.80%**; with cautious recovery
  it reached **0.87% timeout / 31.27% gold / 2.07% deaths / +271.70**.
  Gold and score remain close to cautious recovery on the published backbone.

The [selection note](selection.md) locks the final evaluation before opening
the test panel. The paired score decomposition is in
[Screen 2](screen-2/paired-outcomes.json) and
[Screen 3](screen-3/router-outcomes.json).

### Research and independent review

I used Claude Opus 5.5 as a research reviewer and Sonnet 4.6 as the
documentation subagent. Opus suggested a paired score decomposition,
a separately learned death critic, and action-attempt history. I used
these with raw observations, rejecting computed safe frontiers and
hard-coded exit actions. The documentation review emphasized writing down
the before/after outcome counts and the validation guardrails. Its proposed
pose-only cycle test was too weak, so the audit includes both full states.

During the final code audit, Opus found that the router's exploratory
arm saw fewer action attempts than the same arm saw when it generated
the router's training labels. I added a regression test comparing
stand-alone and routed command selection and fixed the prefix accounting.
The audit also checked that the exact-state witness was complete and that
the frozen exact variants do not read hidden inputs or bypass commands.
I retained the initially locked run and executed a corrected sensitivity
rerun using the same settings. See [selection and audit notes](selection.md).

### Prioritized next tests, informed by the literature

1. **Calibrated two-critic LCS.** Keep the return and death predictors
   separate. Fit the death critic with a probabilistic loss and check
   calibration on independent validation caves. Tune the cost weight to
   a declared maximum death increase, rather than tuning only mean score.
   Compare against the tested fixed training surcharge and `router_30`
   at equal simulator actions. This uses the separation of reward and
   constraint advocated in [Achiam et al., ICML 2017](https://proceedings.mlr.press/v70/achiam17a.html);
   the experiments here are **not** an implementation of CPO and give no
   formal bound on risk. A useful auxiliary outcome is whether a probe
   finds a new square and is followed by a survival or gold exit.
2. **Coherent exploration, not more random moves.** Pick a bootstrap rule
   table for an entire recovery prefix during training and learn from its
   own completed trips. Contrast that against per-action noise and plain
   novelty at the same action budget. [Osband et al., NeurIPS 2016](https://proceedings.neurips.cc/paper_files/paper/2016/hash/8d8818c8e140c64c743113f563cf750f-Abstract.html)
   motivate consistent exploration over many actions, and
   [Badia et al., ICLR 2020](https://arxiv.org/abs/2002.06038) use learned
   episodic novelty. The failure of a larger fixed novelty bonus here
   argues for comparing this with calibrated risk, not just raising it.
   Preserve rare gold trips with positive-advantage
   [self-imitation](https://proceedings.mlr.press/v80/oh18b.html).
3. **Test whether additive rules are the bottleneck.** Train a small
   recurrent agent on the same raw percepts and command history, with no
   search, no oracle features, and the same action budget. Add observable
   next-percept or action-effect prediction only as an ablation. Compare
   its gold/death/timeout scores and successful-path length against the
   LCS. If it wins, try an interpretable rule interaction or distill from
   the agent's own simulated experience as a separate follow-up, not
   from the planner. [Ni et al., ICML 2022](https://proceedings.mlr.press/v162/ni22a.html)
   support a carefully tuned recurrent model-free baseline under partial
   observation. [DreamerV3, Hafner et al., Nature 2025](https://www.nature.com/articles/s41586-025-08744-2)
   is a higher-cost research comparator: it uses a learned model to
   imagine trajectories in training but samples an actor without
   lookahead. It was not run here and comes after the smaller recurrent
   baseline if training-time imagination is in scope.

The notebook's BFS finds short paths within its inferred map. Beating it
would require better *decisions under uncertainty*, more efficient primitive
routes, or both—not just moving more often. The notebook is a comparator,
not an optimal controller or a source of training labels.

### Held-out outcome

The primary cautious model cut timeouts **20.50% → 0.98%** and improved
the same backbone's score by **+16.14**. The smoothed variant's validation
advantage did not transfer: it reached 1.55% timeout but had lower score
and more deaths than the primary model. The corrected router reached
**32.25% gold / +262.08**, still **20.08 points short** of the notebook
despite a slightly higher gold point estimate. The [report](report.md)
contains outcome conversions, paired intervals, an illustrated trajectory,
and the distinction between the locked test and the code-audit rerun.
