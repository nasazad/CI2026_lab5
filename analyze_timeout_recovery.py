"""Decompose paired outcomes and original reward for the recovery benchmark."""

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from paired_wumpus import load_rows, paired_intervals


def score_parts(row):
    gold = 1000 * (row["outcome"] == "gold")
    deaths = -1000 * (row["outcome"] == "death")
    steps = -int(row["actions"])
    arrow = int(row["score"]) - gold - deaths - steps
    if arrow not in (-10, 0):
        raise ValueError(f"score does not decompose: {row}")
    return gold, deaths, steps, arrow


def summarize(rows, candidate, reference):
    def grid(policy):
        subset = [r for r in rows if r["policy"] == policy]
        cells = {(r["seed"], r["cave_id"]): r for r in subset}
        if len(cells) != len(subset):
            raise ValueError("duplicate seed×layout row")
        return cells

    a, b = grid(candidate), grid(reference)
    if a.keys() != b.keys() or not a:
        raise ValueError("nonempty paired panels required")
    transitions = Counter((b[k]["outcome"], a[k]["outcome"]) for k in a)
    differences = [
        [x - y for x, y in zip(score_parts(a[k]), score_parts(b[k]), strict=True)]
        for k in a
    ]
    return {
        "reference_to_candidate_outcomes": {
            f"{before} → {after}": count
            for (before, after), count in sorted(transitions.items())
        },
        "score_delta_components": dict(
            zip(
                ("gold", "deaths", "steps", "arrow"),
                (mean(values) for values in zip(*differences, strict=True)),
                strict=True,
            )
        ),
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("episodes", type=Path)
    cli.add_argument("candidate")
    cli.add_argument("reference", nargs="+")
    cli.add_argument("--draws", type=int, default=2000)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    rows = list(load_rows(args.episodes))
    output = {}
    for name in args.reference:
        comparison = summarize(rows, args.candidate, name)
        if args.draws:
            comparison["paired_intervals"] = paired_intervals(
                rows,
                args.candidate,
                name,
                n=args.draws,
                outcomes=("gold", "death", "timeout"),
            )
        output[name] = comparison
    text = json.dumps(output, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
