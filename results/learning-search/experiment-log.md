# Learning navigation and strategy: experiment log

The learner chooses all six primitive commands. No planner, action mask,
expert trajectory, inferred safety, or hidden map is supplied. Training
changes do not alter the original evaluation score. A new test split is
reserved before training and kept closed during model screening.

## Pass 1 — sharing, credit, and incentives

The raw-percept inputs remain those of the primitive-navigation experiment.
The new learner creates condition–action rules on experience but shares
weights across partial views of the same kind of situation. For example, a
subset can match the same local evidence regardless of the last command or
what was remembered on the far side. No subset maps a percept to a command
until trained.

| Profile | Hypothesis |
|---|---|
| `shared_td` | Partial matching alone may reuse experience that exact covering split across 20K+ rules. |
| `shared_n12` | A longer discount horizon and 12-step backup may carry the gold reward back through turns and moves. |
| `early25`, `early100` | An empty exit without ever leaving the entrance should no longer dominate as easily. The larger penalty could merely relocate the retreat or induce stalling. Waiting at the entrance does not avoid either penalty. |
| `novel20` | A small reward for first reaching a new square may teach exploratory progress rather than standing still. |
| `early25_novel20` | Incentivizing movement and discouraging immediate surrender may be complementary; the earlier profiles isolate the two changes. |
| `shared_curriculum`, `self_replay` | Replaying the agent's own successful episodes may preserve useful experience after the easy-to-mixed curriculum switch. The control uses the same training layouts but does not replay. |

The profile definitions, seeds and per-checkpoint numbers are recorded in
[`screen-1`](screen-1). Training and validation are separate; only the
validation comparison informs follow-up experiments.

**Observed at 5,000 episodes (two seeds, validation):** shared one-step and
12-step updates on their own ended at 0% gold and 100% first-move retreat.
The −25 penalty and +20 novelty bonus, alone or together, also produced 0%
gold. The −100 penalty reached **6.7%** gold, **2.3%** on non-adjacent maps
and **+35.1** mean score; immediate retreat persisted on 50% of maps. The
curriculum without replay produced 0% gold. With one self-replay per
episode, gold reached **10.0%** overall and **4.5%** on non-adjacent maps,
with **+36.2** mean score. This is a screening result, not a held-out result.

## Pass 2 — confirm vs. vary the source and dosage of self-replay

Run five seeds for the two promising profiles and check five additional
variants on the same validation maps before making any test-set selection:

| Profile | Hypothesis |
|---|---|
| `replay_uniform` | Reusing successes might be enough without a curriculum. |
| `replay4` | More reuse may help propagate rare useful experience, or might overweight short successful routes. |
| `replay_early100` | Combining the two strongest pilot changes may improve both exploration and retention. |
| `replay_wide` | Keeping farther raw records may help disambiguate a return trip; it still gives no route or suggested command. |
| `replay_long` | A 32-step target and higher discount may carry credit through longer trajectories. |

**Observed at 5,000 episodes (five seeds, validation):**

| Profile | Gold | Non-adjacent gold | Mean original score |
|---|---:|---:|---:|
| `early100` | 4.8% | 0.9% | +31.9 |
| `self_replay` | 13.5% | 6.5% | +71.1 |
| `replay_uniform` | 11.6% | 4.7% | −34.1 |
| `replay4` | 15.1% | 6.6% | −165.8 |
| `replay_early100` | 14.3% | 7.0% | −45.4 |
| `replay_wide` | **17.7%** | **9.2%** | **+108.4** |
| `replay_long` | 9.6% | 2.5% | +2.4 |

The full [five-seed validation curves](screen-2) include 1,600 and 10,000
episodes. More replay and a stronger incentive to stay in the cave were
not automatically better: some variants returned gold more often at the
cost of a worse original score. The wider-memory result also declined
between 5,000 and 10,000 episodes, so the confirmation uses 5,000 for
every profile.

## Confirmation — selection made before opening the final maps

Test the strongest candidate (`replay_wide`) on the reserved 600
maps, alongside its local-memory counterpart (`self_replay`), the two
early-penalty strengths, the no-replay curriculum control, the shared
unshaped control, the original exact-rule LCS, and the planner notebook as a
reference. Primary measures are gold return rate and original mean score;
non-adjacent returns and representative traces check that successes involve
more than picking up gold one cell from the entrance.

**Observed on the reserved test:** `replay_wide` returned gold on **18.7%**
of the 600 caves per seed (range: 93–132 successes); on the 519 caves with
non-adjacent gold, the success rate was **10.7%**. The mean original score
was **+108.1**. `self_replay` reached **14.6%** and **+72.3**; −100 alone
reached **4.8%** and **+31.7**. The original exact LCS, shared unshaped
control, −25 penalty, and curriculum without replay all had **0%** gold.
Results and traces are in [`final`](final). The planner reference remains
ahead at **32.5%** gold and **+283.5** mean score.

## Follow-up — can a penalty as large as death help?

The additional doses are **−250, −500 and −1,000**, with 0, −25, and −100
as controls. On a retreat without having left the entrance, −1,000 plus
the normal step cost is **−1,001**, exactly the reward for a fatal move.
All six commands remain available at training and test. The training
penalty never applies in test scoring. Waiting is not a loophole; advancing
just one square and returning is possible and will be counted.

A separate pair applies −1,000 to the previously confirmed candidate
(`replay_wide_early1000` vs. `replay_wide`). Every variant uses five seeds and
5,000 episodes. The new validation and test maps exclude both of the
previous splits, and the dose sweep is fixed before evaluation.

**Observed on the new 600-map test:** going from 0 to −1,000 reduced
immediate retreat from **100% to 0%** and increased gold from **0% to
8.7%**, but timeouts rose to **65.4%** and deaths to **13.1%**. The
largest positive mean score among the tested doses was **+12.4** at
−250, not at −1,000. Empty retreats after visiting just one new square
were **18.6%** at −250 and **23.6%** at −500.

On the stronger `replay_wide` policy, adding −1,000 raised overall gold
slightly (**14.3% → 15.0%**) but reduced non-adjacent success
(**9.7% → 8.4%**) and lowered the original mean score
(**+46.3 → −104.3**). The same trade-off was visible on validation.
The penalty does not survive this check. See [`penalty-dose`](penalty-dose)
for the per-seed outcomes and the exact evaluation maps.
