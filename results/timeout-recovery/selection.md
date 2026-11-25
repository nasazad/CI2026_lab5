# Locked final comparison

Selection uses only 600 validation layouts (seeds 0–4). The final panel has
2,000 different layouts and seeds 0–9; seeds 5–9 were not used in the screens.
The new final layouts were reserved for all retraining, including the base
ensembles and the notebook. The historical published split is diagnostic only.

| Role | Backbone | Recipe | Validation timeout | Gold | Death | Mean |
|---|---|---|---:|---:|---:|---:|
| Primary timeout remedy | Published | `cautious_adaptive` | 1.23% | 31.47% | 2.13% | +272.41 |
| Strategic challenger | Published | `router_30` with `risk` arm | 2.07% | 32.77% | 2.40% | +281.70 |
| Lower-timeout ablation | Smoothed target | `cautious_adaptive` | 0.87% | 31.27% | 2.07% | +271.70 |

The primary model makes no architectural change to the existing LCS before
the exact-repeat trigger. The challenger is the highest-scoring validation
router; it is a score/gold trial, not the primary low-timeout claim. The
smoothed variant is the lowest-timeout validation configuration selected
for confirmation; report its own backbone control as well. Inventory and
broad empty-exit reopening are not promoted due to validation deaths.

The output is not used to change thresholds, profiles, rewards or the cap.
Primary comparison: paired timeouts versus the published backbone; the
gold/death/score guardrails are −1 percentage point, +1 percentage point,
and −10 points respectively. Treat ≤2% as the near-zero target and report
the achieved rate, not an absolute-zero guarantee. For a planner win, require
positive paired mean-score difference with its 95% interval above zero, and
inspect gold, deaths, and timeout separately. No claim of equal training budgets.

**Post-run code audit.** A routed exploratory arm did not retain attempt
counts from the frozen prefix, while it did in the independent trials used
to fit the router. The retest fixes only that prefix accounting, with the
same split, seeds, candidates, and fixed settings; no hyperparameters or
selection decisions change. The first locked run is retained in `final/`;
the corrected sensitivity rerun is in `final-corrected/`. The cautious
policy does not route and is unaffected by the bug.
