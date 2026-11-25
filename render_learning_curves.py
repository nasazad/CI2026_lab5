"""Render the primary experiment's CSV data as a portable SVG."""

import csv
import sys
from pathlib import Path
from statistics import mean


COLORS = {
    "coordinate_lcs": ("Coordinate LCS", "#aa452c"),
    "relative_lcs": ("Relative LCS", "#126351"),
    "notebook_baseline": ("Notebook baseline", "#335b91"),
}


def render(root):
    with (root / "validation-curve.csv").open() as file:
        rows = list(csv.DictReader(file))
    budgets = sorted({int(row["episodes"]) for row in rows})
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 920 377"'
        ' role="img" aria-labelledby="title desc">',
        '<title id="title">Learning across randomized caves</title>',
        '<desc id="desc">Validation gold rate and mean score at each training checkpoint. '
        "The notebook baseline is far ahead; neither LCS improves from 5,000 to 10,000 episodes.</desc>",
        '<rect width="920" height="377" rx="14" fill="#f8fafc"/>',
        '<g font-family="system-ui, sans-serif" fill="#172838">',
        '<text x="28" y="36" font-size="24" font-weight="700">Learning across randomized caves</text>',
        '<text x="28" y="60" font-size="14">Validation means over 5 seeds on the same 300 caves · '
        "greedy evaluation at 120 actions</text>",
    ]
    for i, (label, color) in enumerate(COLORS.values()):
        x = (29, 203, 336)[i]
        parts.append(f'<circle cx="{x + 5}" cy="85" r="5" fill="{color}"/>')
        parts.append(f'<text x="{x + 17}" y="90" font-size="13">{label}</text>')
    panels = [
        (
            31,
            "Gold returned",
            (0, 10, 20, 30, 40),
            lambda row: 100 * int(row["gold"]) / int(row["caves"]),
        ),
        (
            480,
            "Mean score",
            (-100, 0, 100, 200, 300),
            lambda row: float(row["mean_score"]),
        ),
    ]
    for left, title, ticks, metric in panels:
        x0, width, top, height = left + 46, 340, 145, 164
        low, high = min(ticks), max(ticks)
        parts.append(
            f'<text x="{left}" y="125" font-size="17" font-weight="700">{title}</text>'
        )
        for tick in ticks:
            yy = top + height * (high - tick) / (high - low)
            parts.append(
                f'<path d="M{x0},{yy:g} h{width}" stroke="#ccd6df" stroke-width="1"/>'
            )
            suffix = "%" if title == "Gold returned" else ""
            parts.append(
                f'<text x="{x0 - 10}" y="{yy + 4:g}" text-anchor="end" font-size="12">{tick}{suffix}</text>'
            )
        xs = [
            x0 + width * (n - budgets[0]) / max(1, budgets[-1] - budgets[0])
            for n in budgets
        ]
        for xx, n in zip(xs, budgets):
            parts.append(
                f'<text x="{xx:g}" y="333" text-anchor="middle" font-size="12">{n:,}</text>'
            )
        parts.append(
            f'<text x="{x0 + width / 2:g}" y="355" text-anchor="middle" font-size="13">Training episodes</text>'
        )
        for policy, (_, color) in COLORS.items():
            ys = []
            for n in budgets:
                value = mean(
                    metric(row)
                    for row in rows
                    if row["policy"] == policy and int(row["episodes"]) == n
                )
                ys.append(top + height * (high - value) / (high - low))
            coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
            parts.append(
                f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5"/>'
            )
            for x, y in zip(xs, ys):
                parts.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.8" fill="{color}" stroke="white"/>'
                )
    parts.append("</g></svg>")
    (root / "learning-curves.svg").write_text("\n".join(parts) + "\n", encoding="utf-8")


if __name__ == "__main__":
    render(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/issue-2"))
