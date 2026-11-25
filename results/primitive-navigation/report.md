# Can an LCS learn navigation and strategy from primitive commands?

**Not with the tested memory and learner.** A curriculum increased the number of successful training episodes, but at the final checkpoint both raw-memory LCS variants climbed out immediately on every held-out cave. The experiment does not rule out other ways of learning navigation without a planner.

## Setup

The new LCS observes the current percepts and remembers the last command and previously observed clues in four relative directions. Dead reckoning attaches observations to locations internally and tracks gold possession. No classifier sees coordinates, inferred safety, a route home, or a recommended action. The six commands, scoring, rule creation, and strength updates were unchanged; no action was masked.

The curriculum used adjacent gold for 400 episodes, then gold at most two safe moves away through episode 1,600. After that, the curriculum and uniform agents saw **identical layouts within each seed**. Training was capped at 100 actions and evaluation at 120. Validation (300 caves) did not justify extending past 10,000 episodes. Final results use **500 fresh caves over five seeds**, with no exploration or updates.

## Held-out results at 10,000 episodes

| Policy | Gold exits / 500 | Immediate retreats / 500 | Timeouts / 500 | Mean cells reached | Mean score |
|---|---:|---:|---:|---:|---:|
| LCS with safety and route hints (PR #8 control) | 11.4 | 127.4 | 360 | 1.59 | −66.0 |
| Raw-memory LCS, uniform training | 0 | 500 | 0 | 1.00 | −1.0 |
| Raw-memory LCS, curriculum | 0 | 500 | 0 | 1.00 | −1.0 |
| Notebook baseline with planner | 175 | 232 | 0 | 3.43 | +317.1 |

Counts are means per seed. Neither raw-memory variant picked up gold, died, or stopped for lack of a rule. Its higher mean score than the hinted LCS is **not evidence of better navigation**: it never left the entrance square.

## What the curriculum changed

By episode 1,600, the curriculum averaged **221.2 training gold exits**, compared with **34.0** for uniform training. It also returned gold on **1.5%** of the mixed validation caves at that checkpoint, versus **0%** without it. But by episode 5,000 both raw-memory variants were back at **0%** validation gold, and neither recovered by 10,000.

The raw-memory variants ended with about **20,554** and **23,830** rules per seed, versus **348** for the LCS with hints. Covering creates rules for the full set of observed messages, so differing memory facts often split visits across different rules. This is a plausible bottleneck, not a proven cause of the retreat policy.

In the [diagnostic trace](traces.json), the notebook returned gold in 15 moves on cave 5. At the entrance, the raw-memory LCS rated `climb` at **−1** and `fwd` at **−76.154**; the curriculum LCS also rated `climb` at **−1**, but `fwd` at **−629.944**. These are learned rule strengths, not the actual rewards on this cave. This failure occurred **before** any navigation attempt: viable commands were available, but the policy preferred not to leave the entrance.

**Next experiment:** keep primitive actions, rewards, and raw memory fixed; compare current covering with a discovery mechanism that can generalize a rule across different memory records by dropping irrelevant conditions. A separate control can test longer-horizon credit assignment. Neither requires adding a planner.

[Configuration](config.json) · [Validation checkpoints](validation.csv) · [Per-seed test summary](test-summary.csv) · [Per-cave outcomes](test-episodes.csv) · [Exact layouts](layouts.csv)
