"""Render documentation figures from the frozen, already published evaluation."""

import argparse
import csv
import json
from itertools import groupby
from math import hypot
from pathlib import Path
from statistics import mean
from xml.sax.saxutils import escape


INK, MUTED = "#17333d", "#496570"
TEAL, TEAL_BG = "#086754", "#e1f2ec"
BLUE, BLUE_BG = "#345b85", "#e8eff8"
GOLD, GOLD_BG = "#a26816", "#faeed4"
RED, RED_BG = "#913f3a", "#f8e8e4"
PAPER, RULE, WHITE = "#f5f8f8", "#cddadd", "#ffffff"


class Figure:
    def __init__(self, height, title, subtitle):
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="980" height="{height}" '
            f'viewBox="0 0 980 {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title>',
            f'<desc id="desc">{escape(subtitle)}</desc>',
            '<g font-family="Arial, sans-serif">',
        ]
        self.rect(0, 0, 980, height, PAPER, r=12)
        self.text(30, 45, title, size=26, weight=700)
        self.text(30, 71, subtitle, size=15, ink=MUTED)

    def text(self, x, y, text, *, size=15, weight=400, ink=INK, anchor="start"):
        self.parts.append(
            f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
            f'fill="{ink}" text-anchor="{anchor}">{escape(text)}</text>'
        )

    def rect(self, x, y, w, h, fill, *, r=0, stroke="none"):
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" '
            f'fill="{fill}" stroke="{stroke}"/>'
        )

    def circle(self, x, y, rad, fill, *, stroke="none"):
        self.parts.append(
            f'<circle cx="{x}" cy="{y}" r="{rad}" fill="{fill}" stroke="{stroke}"/>'
        )

    def line(self, x1, y1, x2, y2, ink=RULE, sw=1.4, dash=""):
        self.parts.append(
            f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="{ink}" '
            f'stroke-width="{sw}" stroke-linecap="round" stroke-dasharray="{dash or "none"}"/>'
        )

    def arrow(self, a, b, ink=TEAL, sw=2.1, dash=""):
        x1, y1 = a
        x2, y2 = b
        length = hypot(x2 - x1, y2 - y1)
        ux, uy = (x2 - x1) / length, (y2 - y1) / length
        self.line(x1, y1, x2, y2, ink, sw, dash)
        for sign in (-1, 1):
            self.line(
                x2,
                y2,
                round(x2 - 7 * ux + sign * 4 * uy, 2),
                round(y2 - 7 * uy - sign * 4 * ux, 2),
                ink,
                sw,
                dash="",
            )

    def card(self, x, y, width, height, heading, lines, fill=WHITE, accent=INK):
        self.rect(x, y, width, height, fill, r=12)
        self.text(x + 14, y + 28, heading, size=16, weight=700, ink=accent)
        for i, line in enumerate(lines):
            self.text(x + 14, y + 53 + i * 20, line, size=14)

    def save(self, path):
        path.write_text("\n".join([*self.parts, "</g></svg>"]) + "\n", encoding="utf-8")


def workflow(out):
    s = Figure(
        420,
        "How a learned classifier chooses one move",
        "The LCS learns which command to use; it never calculates a route.",
    )
    s.text(30, 108, "TRAINING", size=13, weight=700, ink=BLUE)
    training = (
        ("Try actions", ["Explore with all six", "primitive commands."]),
        ("Assign credit", ["Update earlier actions", "from observed rewards."]),
        ("Replay successes", ["Skip repeated loops", "from actual good runs."]),
        ("Merge three tables", ["Average the learned", "strengths once."]),
    )
    for i, (title, lines) in enumerate(training):
        x = 30 + i * 233
        s.card(x, 121, 212, 87, title, lines, BLUE_BG, BLUE)
        if i < 3:
            s.arrow((x + 215, 165), (x + 229, 165), BLUE)
    s.text(30, 239, "FROZEN POLICY  /  EVERY MOVE", size=13, weight=700, ink=TEAL)
    inference = (
        (
            "Read and remember",
            [
                "Current percepts, last action,",
                "and raw observations relative",
                "to the agent.",
            ],
        ),
        (
            "Score the commands",
            [
                "Find matching conditions.",
                "Add their learned strengths",
                "separately for each command.",
            ],
        ),
        (
            "Execute one command",
            [
                "Take the highest score:",
                "forward, turn left / right,",
                "grab, shoot, or climb.",
            ],
        ),
    )
    for i, (title, lines) in enumerate(inference):
        x = 30 + i * 313
        s.card(x, 252, 286, 108, title, lines, TEAL_BG, TEAL)
        if i < 2:
            s.arrow((x + 290, 300), (x + 309, 300))
    s.rect(30, 376, 912, 27, WHITE, r=6)
    s.text(
        41,
        394,
        "Illustration: two matching conditions contribute +35 and +12 to forward, for a combined score of +47.",
        size=14,
    )
    s.save(out / "workflow.svg")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def comparison(root, out):
    main = read_csv(root / "final" / "test-summary.csv")
    matched = read_csv(root / "matched-actions" / "test-summary.csv")
    rows = (
        ("Previous primitive LCS", "replay_wide", main, "#8498a0"),
        ("Single LCS, matched training actions", "low_clean", matched, "#60788b"),
        ("Averaged learned LCS", "clean_ensemble3", main, TEAL),
        ("Notebook with planner", "notebook", main, BLUE),
    )
    s = Figure(
        365,
        "Close in gold returns; a gap in total score",
        "2,000 unseen layouts × five seeds · original rewards · 120-action cap",
    )
    columns = ((323, 250, 40, "Gold returned"), (647, 244, 350, "Mean score"))
    for x, width, maximum, name in columns:
        s.text(x, 110, name, size=17, weight=700)
        for tick in range(5 if maximum == 40 else 8):
            value = tick * (10 if maximum == 40 else 50)
            wx = x + width * value / maximum
            s.line(wx, 126, wx, 290, RULE, 0.9)
            s.text(
                round(wx, 2),
                302,
                f"{value}%" if maximum == 40 else str(value),
                size=11,
                ink=MUTED,
                anchor="middle",
            )
    for i, (label, policy, source, color) in enumerate(rows):
        group = [r for r in source if r["policy"] == policy]
        if len(group) != 5 or any(int(r["caves"]) != 2000 for r in group):
            raise ValueError(f"unexpected test panel for {policy}")
        gold = mean(100 * int(r["gold"]) / int(r["caves"]) for r in group)
        score = mean(float(r["mean_score"]) for r in group)
        yy = 133 + i * 42
        s.text(33, yy + 17, label, size=15, weight=700 if i >= 2 else 400)
        for (x, w, maximum, _), value, annotation in zip(
            columns, (gold, score), (f"{gold:.2f}%", f"+{score:.2f}"), strict=True
        ):
            bw = round(w * value / maximum, 2)
            s.rect(x, yy, bw, 23, color, r=4)
            s.text(
                x + bw + 8,
                yy + 16,
                annotation,
                size=14,
                weight=700,
                ink=INK if i < 2 else color,
            )
    s.line(30, 308, 949, 308)
    s.text(
        30,
        332,
        "The averaged LCS still timed out in 17.93% of runs; the planner never timed out on this test.",
        size=14,
        ink=MUTED,
    )
    s.save(out / "held-out-results.svg")


def round_trip(root, out):
    trace = json.loads((root / "final" / "traces.json").read_text())[
        "clean_ensemble3:0"
    ]
    assert (trace["cave_id"], trace["actions"], trace["score"]) == (350, 24, 966)
    gold, wumpus, pits = trace["layout"]
    steps = trace["trace"]
    grabbed = next(i for i, step in enumerate(steps) if step["action"] == "grab")
    legs = [
        [pos for pos, _ in groupby(tuple(step["square"]) for step in segment)]
        for segment in (steps[: grabbed + 1], steps[grabbed:])
    ]
    s = Figure(
        500,
        "A learned round trip on an unseen cave",
        "Test cave 350, seed 0 · gold at (4,4) · 24 commands · +966 points",
    )
    for i, (x0, title, leg) in enumerate(
        zip(
            (115, 592),
            ("1  Explore and grab", "2  Return and climb"),
            legs,
            strict=True,
        )
    ):
        s.card(x0 - 85, 91, 440, 335, title, [], WHITE)
        y0, unit = 142, 57

        def pos(square):
            return x0 + (square[0] - 0.5) * unit, y0 + (4.5 - square[1]) * unit

        for y in range(1, 5):
            for x in range(1, 5):
                cx, cy = pos((x, y))
                cell = [x, y]
                fill = (
                    RED_BG
                    if cell in pits or cell == wumpus
                    else GOLD_BG
                    if cell == gold
                    else PAPER
                )
                s.rect(cx - 27, cy - 27, 54, 54, fill, r=4, stroke=RULE)
                if cell in pits or cell == wumpus:
                    s.text(
                        cx,
                        cy + 6,
                        "W" if cell == wumpus else "P",
                        size=16,
                        weight=700,
                        ink=RED,
                        anchor="middle",
                    )
            s.text(
                x0 - 12, pos((1, y))[1] + 5, str(y), size=12, ink=MUTED, anchor="middle"
            )
        for x in range(1, 5):
            s.text(
                pos((x, 1))[0], y0 + 246, str(x), size=12, ink=MUTED, anchor="middle"
            )
        pairs = list(zip(leg, leg[1:]))
        for a, b in pairs:
            u, v = pos(a), pos(b)
            dx, dy = (v[0] - u[0]) / unit, (v[1] - u[1]) / unit
            backtrack = (b, a) in pairs
            shift = 8 if backtrack else 0
            s.arrow(
                (u[0] + 13 * dx - dy * shift, u[1] + 13 * dy + dx * shift),
                (v[0] - 13 * dx - dy * shift, v[1] - 13 * dy + dx * shift),
                GOLD if backtrack else TEAL if i == 0 else BLUE,
                2.1,
                "4 3" if backtrack else "",
            )
        for cell in sorted(set(leg)):
            s.circle(*pos(cell), 4, TEAL if i == 0 else BLUE, stroke=WHITE)
        start = pos((1, 1))
        treasure = pos(gold)
        s.text(start[0], start[1] + 20, "Start", size=12, anchor="middle", weight=700)
        s.text(
            treasure[0],
            treasure[1] - 7,
            "Gold",
            size=12,
            anchor="middle",
            weight=700,
            ink=GOLD,
        )
        note = (
            "The dashed arrows show an initial backtrack."
            if i == 0
            else "The return follows the top and left edges."
        )
        s.text(x0 - 69, 408, note, size=13, ink=MUTED)
    s.text(
        30,
        454,
        "P = pit · W = Wumpus. Hazards are shown for the reader only; the LCS did not see the map.",
        size=14,
        ink=MUTED,
    )
    s.text(
        30,
        477,
        "Arrows show square changes only; the score also includes turns, grabbing, firing one arrow, and climbing.",
        size=13,
        ink=MUTED,
    )
    s.save(out / "held-out-path.svg")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "--results", type=Path, default=Path("results/learned-exploration")
    )
    args = cli.parse_args()
    figures = args.results / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    workflow(figures)
    comparison(args.results, figures)
    round_trip(args.results, figures)


if __name__ == "__main__":
    main()
