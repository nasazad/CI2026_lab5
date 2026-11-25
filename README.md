Lab 5: The Wumpus World with classifier systems

Video presentation

https://drive.google.com/file/d/1UH04lIiehiL2IuBxS5nEd-DPmVLJu4z1/view?usp=sharing

## Issue 5: fixed-cave transfer benchmark

From this fork's `issue-5-fixed-cave-transfer` branch, with Python 3.10 or later:

```sh
python -B -m unittest -v test_transfer
python -B benchmark_transfer.py --output results/issue-5
```

The benchmark reads the randomized-cave baseline from a pinned notebook
revision, verifies its SHA-256, and loads only its definition cells. To run
offline, pass `--notebook /path/to/Wumpus_World.ipynb` with that exact revision.
Training uses each implementation's original setup. Both policies are then
frozen and evaluated with the same simulator and 120-action limit. The fixed
LCS is evaluated before optional GA compaction.

See [`results/issue-5`](results/issue-5) for the configuration, exact cave
layouts, per-policy and per-cave outcomes, representative traces, and chart.

## [Issue #7](https://github.com/nasazad/CI2026_lab5/issues/7): randomized training

```sh
python -B -m unittest -v test_transfer test_randomized
python -B benchmark_randomized.py
```

Both LCS variants use the original covering, strength updates, and action set.
The coordinate control uses `DeadReckoning`. The relative variant replaces its
coordinates and heading in classifier messages with the inferred safety ahead,
the relative direction of a known route home, and whether it carries gold.
Both still receive the same raw percepts. The notebook baseline is retrained on
each shared training stream.

The experiment reserves validation, fresh test, and Issue #5 layouts before
sampling training streams. Training is capped at 100 actions; evaluation at
120. By default it checks five seeds at 1,600, 5,000 and 10,000 episodes.
Validation gains determine whether to extend to 25,000 or 50,000 episodes.
If validation timeouts reach 10%, a separate sensitivity check compares 120
and 240 actions on validation only.

The [`results/issue-2`](results/issue-2) data include per-seed learning
curves, average-per-cave outcomes, the exact layouts, and representative
traces. The [report](results/issue-2/report.md) summarizes both experiments.
Run `python -B render_learning_curves.py` to regenerate the main chart.
For offline use, pass `--notebook` to the benchmark and set
`WUMPUS_BASELINE_NOTEBOOK` for the tests to your copy of the pinned notebook.

To reproduce the exploratory follow-up, which tests whether blocking only
`climb` away from the entrance helps the relative-input variant:

```sh
python -B benchmark_randomized.py --climb-ablation
```

This run retrains both guarded and unguarded agents on the same streams and
uses a new validation/test split, separate from the main experiment. It
records results in [`results/issue-2-climb`](results/issue-2-climb).

## Follow-up: learn primitive navigation without route hints

```sh
python -B -m unittest -v test_transfer test_randomized test_navigation
python -B benchmark_navigation.py
```

The raw-memory LCS keeps the existing rule discovery, strength updates, and
six commands. It remembers observed breeze, stench, and entrance bits in
four egocentric slots and the last command. Dead reckoning associates
observations with squares and tracks gold possession; classifiers do not
receive coordinates, inferred safety, a home direction, or a route. The
relative-hints LCS from PR #8 is a separate control, not part of the raw
policy. The notebook baseline is also included for comparison.

The curriculum uses randomized maps with adjacent gold for 400 episodes,
then gold reachable in at most two safe moves until episode 1,600. From
episode 1,601 the two raw-memory agents use exactly the same uniform
training layouts. This changes the order and difficulty of training maps,
not rewards, and exposes no map attributes to the agent. Validation chooses
whether to extend the planned 10,000 episodes. A fresh 500-map test is run
once with a 120-action cap. The run records gold pickups separately from
gold exits, cells reached, immediate retreats, and timeouts without a new
square in the last 30 actions.

See [`results/primitive-navigation`](results/primitive-navigation) for the
report, seeds, learning curves, per-cave results, and representative traces.
The offline notebook options are the same as for the earlier benchmarks.
