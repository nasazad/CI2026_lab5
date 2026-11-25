"""Paired seed×cave bootstrap; the same cave is reused across training seeds."""

import argparse
import csv
import json
import math
from pathlib import Path
from random import Random
from statistics import quantiles


def outcome_grid(rows, policy):
    cells = {}
    for row in rows:
        if row["policy"] != policy:
            continue
        key = row["seed"], row["cave_id"]
        score = float(row["score"])
        if key in cells or not math.isfinite(score):
            raise ValueError(
                f"duplicate seed/cave or non-finite score for {policy}: {key}"
            )
        cells[key] = (int(row["outcome"] == "gold"), score)
    return cells


def paired_intervals(rows, candidate, reference, n=2000, seed=123):
    """Estimate candidate-minus-reference with percentile product-bootstrap CIs."""
    if candidate == reference or n < 2:
        raise ValueError("specify distinct policies and at least two draws")
    rows = list(rows)
    cand, ref = outcome_grid(rows, candidate), outcome_grid(rows, reference)
    if not cand or cand.keys() != ref.keys():
        raise ValueError("policies need matching, nonempty seed×cave grids")
    seeds = sorted({s for s, _ in cand})
    caves = sorted({c for _, c in cand})
    ns, nc = len(seeds), len(caves)
    if len(cand) != ns * nc:
        raise ValueError("incomplete seed×cave grid")
    # Opus independently derived the two-axis resampling. Each replicate
    # draws cave indices once and reuses those indices across its seed draws.
    deltas = [
        [[cand[s, c][m] - ref[s, c][m] for c in caves] for s in seeds] for m in (0, 1)
    ]
    rng = Random(seed)
    samples = [[], []]
    for _ in range(n):
        si = rng.choices(range(ns), k=ns)
        ci = rng.choices(range(nc), k=nc)
        for matrix, series in zip(deltas, samples):
            series.append(sum(matrix[i][j] for i in si for j in ci) / (ns * nc))
    result = {"n_seeds": ns, "n_caves": nc, "replicates": n}
    for metric, matrix, series in zip(("gold_rate", "mean_score"), deltas, samples):
        bounds = quantiles(series, n=40, method="inclusive")
        result[metric] = {
            "delta": sum(map(sum, matrix)) / (ns * nc),
            "lo": bounds[0],
            "hi": bounds[-1],
        }
    return result


def load_rows(path, *, rename_from=None, rename_to=None):
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            if row["policy"] == rename_from:
                row["policy"] = rename_to
            yield row


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("csv", type=Path)
    cli.add_argument("candidate")
    cli.add_argument("reference")
    cli.add_argument("--draws", type=int, default=2000)
    cli.add_argument("--extra", type=Path)
    cli.add_argument("--extra-policy")
    args = cli.parse_args()
    rows = list(load_rows(args.csv))
    if args.extra:
        extra = args.extra_policy or args.reference
        rows.extend(load_rows(args.extra, rename_from=extra, rename_to=args.reference))
    print(
        json.dumps(
            paired_intervals(rows, args.candidate, args.reference, n=args.draws),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
