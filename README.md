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
