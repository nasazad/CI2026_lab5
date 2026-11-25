"""Create the retreat-penalty SVG from the held-out summary."""

import csv
import sys
from pathlib import Path
from statistics import mean


DOSES = (0, 25, 100, 250, 500, 1000)
POLICIES = ("shared_n12", *(f"early{dose}" for dose in DOSES[1:]))
SERIES = (
    ("immediate_retreat", "Immediate retreat", "#436486"),
    ("timeout", "Timeout", "#99601b"),
    ("death", "Death", "#a13b35"),
    ("gold", "Gold exit", "#097360"),
)


def render(root):
    with (root / "penalty-dose" / "test-summary.csv").open() as file:
        rows = list(csv.DictReader(file))
    groups = [[r for r in rows if r["policy"] == policy] for policy in POLICIES]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 930 395" role="img" aria-labelledby="title desc">',
        '<title id="title">The cost of penalizing immediate retreat</title>',
        '<desc id="desc">On 600 unseen maps, a 1000-point penalty removed immediate retreats but increased timeouts and deaths. '
        "The best original mean score among the six doses was obtained at 250.</desc>",
        '<rect width="930" height="395" rx="14" fill="#f8fafc"/>',
        '<g font-family="system-ui, sans-serif" fill="#172838">',
        '<text x="26" y="37" font-size="24" font-weight="700">The cost of penalizing immediate retreat</text>',
        '<text x="26" y="61" font-size="14">600 fresh caves per seed · five seeds · original scoring at test time</text>',
        '<text x="27" y="108" font-size="17" font-weight="700">Outcome rates</text>',
        '<text x="572" y="108" font-size="17" font-weight="700">Mean score</text>',
    ]
    top, h = 134, 172
    positions = ((65, 446), (610, 270))
    ticks = ((0, 25, 50, 75, 100), (-150, -100, -50, 0, 25))
    for panel, ((left, width), levels) in enumerate(zip(positions, ticks)):
        lo, hi = levels[0], levels[-1]
        xs = [left + i * width / 5 for i in range(6)]
        ys = lambda v: top + h * (hi - v) / (hi - lo)
        for value in levels:
            yy = ys(value)
            label = f"{value}%" if panel == 0 else str(value)
            parts.append(f'<path d="M{left},{yy:.1f} h{width}" stroke="#cdd7df"/>')
            parts.append(
                f'<text x="{left - 8}" y="{yy + 4:.1f}" text-anchor="end" font-size="11">{label}</text>'
            )
        for x, dose in zip(xs, DOSES):
            parts.append(
                f'<text x="{x:.1f}" y="327" text-anchor="middle" font-size="11">{dose:,}</text>'
            )
        parts.append(
            f'<text x="{left + width / 2:.1f}" y="350" text-anchor="middle" font-size="13">Tested penalties (not to scale)</text>'
        )
        series = SERIES if panel == 0 else (("mean_score", "Mean score", "#283f54"),)
        for metric, _, color in series:
            values = [
                mean(
                    float(r[metric]) * (100 / int(r["caves"]) if panel == 0 else 1)
                    for r in group
                )
                for group in groups
            ]
            points = " ".join(
                f"{x:.1f},{ys(value):.1f}" for x, value in zip(xs, values)
            )
            parts.append(
                f'<polyline points="{points}" stroke="{color}" stroke-width="2.5" fill="none"/>'
            )
            for x, value in zip(xs, values):
                parts.append(
                    f'<circle cx="{x:.1f}" cy="{ys(value):.1f}" r="4" fill="{color}" stroke="white"/>'
                )
    for (_, label, color), x in zip(SERIES, (28, 184, 296, 392)):
        parts.append(f'<circle cx="{x + 5}" cy="373" r="4" fill="{color}"/>')
        parts.append(f'<text x="{x + 14}" y="377" font-size="12">{label}</text>')
    parts.append("</g></svg>")
    (root / "penalty-dose.svg").write_text("\n".join(parts) + "\n", encoding="utf-8")


if __name__ == "__main__":
    render(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/learning-search"))
