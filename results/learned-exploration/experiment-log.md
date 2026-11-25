# Learned exploration: experiment log

Starting point: the primitive-action LCS with raw memory and self-replay
returned gold on 18.7% of an earlier test; the planner notebook returned
gold on 32.5%. The LCS timed out on 39.6%, but returned on 97.7% of
episodes where it picked up the gold. The aim is to close the exploration
gap without computing routes, masking actions, using hidden cave data as
input, or changing the original evaluation reward.

## Collaboration

I asked Claude Code to use **`claude-opus-5-5` at `xhigh`** in two
independent passes. Opus proposed correcting timeout bootstrapping,
passive loop memory, count exploration, and training-only potential
shaping. It also derived numerical tests and suggested checking that
training counts do not affect frozen evaluation. Its suggested safe
frontier signal was **not adopted**: computing that would provide a
navigation decision that the agent is meant to learn.

## Screen 1 — validation only

The progress split reserves a new set of 2,000 test caves, disjoint from
all earlier evaluation maps; these are not used for selection. The first
screen uses two seeds and 2,500, 5,000 and 10,000 training episodes on a
separate 420-map validation set. All profiles retain the original shared
raw rules and self-generated replay unless stated otherwise.

| Change | Expectation |
|---|---|
| `boot` vs. `replay_wide` | A training time limit is truncation, not a terminal environment event. The value of the actual last observation may reduce erroneous credit near the cap. |
| `loop_only`, `boot_loop` | Visit counts, time since motion, and repeated turns let a rule distinguish the first choice from a repeated failure. They do not forbid any action. |
| `boot_n1`, `boot_n4`, `boot`; `count_n1`, `count_n4`, `count_n12` | Compare 1-, 4- and 12-step credit with and without a training-only state-action count bonus. More exploration should produce new physical visits, not just more turns. |
| `boot_potential` | An annealed progress potential may encourage useful visits in training without giving extra points in evaluation. |
| `boot_balance` | Balanced replay of successful episodes by observed pickup distance may keep rarer long tours from being dominated by adjacent-gold examples. |

The executable records the per-seed validation numbers and exact profile
settings in [`screen-1`](screen-1). The frozen test uses the original
simulator scores and all six actions; no test-time count bonus is allowed.

Results: at 10,000 episodes the 12-step count variant reached **21.4%**
gold and **+144.1** on the two-seed validation screen. Balanced experience
replay reached **21.9%** gold but only **+44.7**. Combining passive loop
features with truncation alone did not improve gold. The corrected
transition and potential tests were implemented from Opus's arithmetic
examples. The count term is used only to choose training actions, not to
change rewards or the frozen policy.

## Screen 2 — what to retain when training gets longer?

The [four-seed screen](screen-2) tests count exploration with passive loop
memory, balanced replay, self-imitation, exact observable loop erasure,
annealed random actions and annealed novelty shaping. At 10,000, replay
of loop-erased successful experiences reached **24.1%** gold versus
**17.9%** for the count control. Reducing random actions from 20% to 5%
over training gave **28.3%** gold at 16,000, but mean score fell from
**+148.9** at 10,000 to **+103.7** at 16,000. Novelty shaping did not
beat the count control. Self-imitation only reinforces positive advantage
from the original return; estimates still train on real deaths and retreats.

## Screen 3 — level replay vs. rule-table averaging

I implemented Opus's suggested training-level replay, with uniform
revisiting as a control. The scheduler sees a layout identifier and
positive TD error, never its hidden attributes. Both replay samplers
perform worse than or near the non-revisit control. The
[three-seed screen](screen-3) also combines low-noise training with clean
self-replay, and tests averaging three independently trained rule tables.
The averaged clean profile reached **34.2%** gold and **+285.8** points at
16,000 episodes per member. Averaging identical condition keys retains a
single greedy LCS at inference. Extra training experience is counted.

## Final confirmation — selection locked before opening the test maps

Compare `clean_ensemble3` and `low_ensemble3` at 16,000 episodes per
member against the notebook at 16,000, the prior `replay_wide` at its
earlier selected 5,000, and the strongest short single control `low_clean`
at 10,000. Use five seeds and the same 2,000 fresh layouts for every arm.
An additional `low_clean` single model receives **exactly the same
simulator actions per seed** as `clean_ensemble3`; the episode limit is
shortened only on its final training episode as needed.

Opus also supplied a paired seed×cave bootstrap implementation to audit
the final deltas without treating the same cave across seeds as separate
independent maps.

**Observed held-out:** the clean averaged policy returned gold in
**34.76%** of runs (+284.49 mean score), compared with **16.15%** (+85.45)
for the previous primitive LCS and **35.45%** (+327.90) for the notebook.
The action-matched single model achieved **32.29%** and **+239.42**.
Exact per-seed budgets were verified. Opus independently reviewed the
interpretation: the gold estimates are close, not a formal equivalence
result; the remaining score gap and the extra training cost must be
reported. See the [report](report.md) and
[paired intervals](final/paired-bootstrap.json). The primary test was
rerun and every per-cave outcome reproduced byte-for-byte.
