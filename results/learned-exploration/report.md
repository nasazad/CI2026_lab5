# The primitive-action LCS closes most of the gap

**On 2,000 new caves, the LCS returned gold in 34.76% of runs versus 35.45% for the notebook planner.** Its mean score remained lower (**284.49 vs. 327.90**): it still got stuck or died more often. The original primitive-action control returned gold in 16.15% on this same test.

The selected LCS averages the strengths of matching condition–action rules from three independently trained models. At inference this is **one frozen rule table**, choosing one of the original six commands. It does not invoke a path planner, receive computed safety or home directions, mask actions, or read the hidden cave. The individual models learn from raw observations and their own actions; no expert path was supplied.

## Held-out comparison

Every row uses the same 2,000 distinct held-out layouts, with five paired training seeds. Outcomes use the original reward and a fixed 120-action cap. Non-adjacent means at least two Manhattan steps from the entrance (1,724 layouts).

| Policy | Gold | Non-adjacent gold | Mean score | Deaths | Timeouts |
|---|---:|---:|---:|---:|---:|
| Previous primitive LCS, wider raw memory | 16.15% | 9.28% | +85.45 | 1.70% | 42.59% |
| Single LCS with clean self-imitation, 10K episodes | 32.23% | 25.12% | +225.22 | 5.81% | 21.89% |
| Single LCS with clean self-imitation, **matched actions** | 32.29% | 25.55% | +239.42 | 4.23% | 23.02% |
| Three averaged LCSs, ordinary success replay | 32.66% | 25.58% | +254.72 | 3.51% | 21.60% |
| **Three averaged LCSs, clean self-imitation** | **34.76%** | **27.81%** | **+284.49** | **2.97%** | **17.93%** |
| Notebook with planner | 35.45% | 29.41% | +327.90 | 1.55% | 0.00% |

The paired 95% bootstrap interval for the LCS-minus-notebook gold difference (**−0.69 percentage points**) is **[−2.50, +1.25] pp**. The estimate is close; this is not a formal equivalence result. The score gap is clear: **−43.41**, 95% interval **[−61.83, −25.23]**. The bootstrap pairs policies on every seed and cave and resamples those two axes independently.

### Is averaging simply buying more training?

Each averaged model uses 16,000 training episodes per member. The action-matched control trains **one** LCS with the same recipe until it has consumed exactly the ensemble's simulator steps for its seed, about **1.35M actions** on average. Averaging still improves the original score by **+45.07** (95% interval **[+10.54, +74.87]**); the **+2.47 pp** gain in gold is less certain (interval **[−0.41, +5.07] pp**). The notebook itself used about **0.17M** training actions per seed, so closeness in gold rate is **not** equal sample efficiency.

## Which changes mattered?

- **Training exploration and reduced late-stage noise.** Count bonuses and uniform random actions are annealed in training; neither changes the frozen action selection or the score.
- **Replay cleaned successes.** Self-imitation reinforces a command only when the actual discounted return exceeds its current estimate. Exact repeated states are removed from successful training records before replay; it does not invent an action or a path. Deaths and retreats still train the main action values.
- **Average learned strengths.** Three learners see the same sequence of training layouts within each experiment seed, but use different action-exploration streams. The arithmetic mean of their rule weights is frozen. Missing rules contribute zero for that member.
- **Changes not selected.** Replaying whole levels, novelty-potential shaping, and adding passive loop counters by themselves did not match the selected recipe on validation. The [experiment log](experiment-log.md) gives the rationale and numbers for each trial.

Once it picked up the gold, the selected policy returned it on **3,476 of 3,499 runs (99.3%)**. The remaining gap is mostly in finding gold reliably without stalling, not in bringing it back once found.

For one concrete held-out example, seed 0 reached the gold at `(4,4)` on
cave 350 and returned in **24 actions for +966**. Each of the five seeds
has a recorded successful held-out trace with gold at `(4,4)`.

[Per-cave outcomes](final/test-episodes.csv) · [Paired confidence intervals](final/paired-bootstrap.json) · [Held-out trajectories](final/traces.json) · [Exact-action control](matched-actions/validation.csv)
