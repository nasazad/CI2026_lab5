"""Render a directly labeled chart from the frozen timeout study output."""

import argparse
import json
from itertools import groupby
from pathlib import Path
from statistics import mean

from render_exploration_figures import (
    BLUE,
    GOLD,
    GOLD_BG,
    INK,
    MUTED,
    PAPER,
    RED,
    RED_BG,
    RULE,
    TEAL,
    TEAL_BG,
    WHITE,
    Figure,
    read_csv,
)


def chart(root, name):
    primary = read_csv(root / "final-corrected" / "summary.csv")
    smooth = read_csv(root / "final-smooth" / "summary.csv")
    sources = [
        ("Published LCS, retrained", primary, "original", "#60788b"),
        ("Learned cautious recovery", primary, "cautious_adaptive", TEAL),
        ("Smoothed LCS + recovery", smooth, "cautious_adaptive", TEAL),
        ("Learned recovery router", primary, "router_30", "#527d9b"),
        ("Notebook with BFS", primary, "notebook", BLUE),
    ]
    fig = Figure(
        374,
        "Learning to leave deterministic loops",
        "2,000 fresh caves × ten training seeds · fixed 120-command cap · original rewards",
    )
    cols = ((321, 232, 25, "Timeout"), (626, 247, 400, "Mean score"))
    palette = ((0, 5, 10, 15, 20, 25), (0, 100, 200, 300, 400))
    for (x, width, ceiling, heading), ticks in zip(cols, palette, strict=True):
        fig.text(x, 108, heading, size=17, weight=700)
        for tick in ticks:
            mark = x + width * tick / ceiling
            fig.line(mark, 117, mark, 309, ink=RULE, sw=0.8)
            fig.text(
                mark,
                327,
                f"{tick}%" if ceiling == 25 else str(tick),
                anchor="middle",
                ink=MUTED,
                size=12,
            )
    for i, (label, table, key, color) in enumerate(sources):
        rows = [r for r in table if r["policy"] == key]
        if len(rows) != 10 or any(int(r["caves"]) != 2000 for r in rows):
            raise ValueError(f"not the locked final panel for {label}")
        timeout = 100 * mean(int(r["timeout"]) / int(r["caves"]) for r in rows)
        score = mean(float(r["mean_score"]) for r in rows)
        y = 125 + 37 * i
        fig.text(30, y + 17, label, size=15, weight=700 if i in (1, 4) else 400)
        for (x, width, ceiling, _), value, text in zip(
            cols, (timeout, score), (f"{timeout:.2f}%", f"+{score:.2f}"), strict=True
        ):
            bar = width * value / ceiling
            fig.rect(x, y, bar, 23, color, r=4)
            fig.text(x + bar + 8, y + 16, text, size=14, weight=700, ink=INK)
    fig.text(
        30,
        354,
        "A timeout avoided is not necessarily a gold return. See the report for all outcomes.",
        ink=MUTED,
        size=14,
    )
    fig.save(root / name)


def example(root):
    row = json.loads((root / "final" / "traces.json").read_text())[
        "cautious_adaptive:0"
    ]
    gate = row["activated_at"] - 1
    assert (row["cave_id"], gate, row["score"]) == (458, 28, 947)
    assert row["baseline_commands"][gate : gate + 4] == ["climb", "fwd"] * 2
    route = [
        cell
        for cell, _ in groupby(tuple(r["square"]) for r in row["recovery_trace"][gate:])
    ]
    gold, wumpus, pits = row["layout_for_reader_only"]
    s = Figure(
        443,
        "One loop, two different outcomes",
        "Held-out cave 458, seed 0 · both agents take exactly the same first 28 commands",
    )
    s.card(30, 92, 432, 303, "Frozen original  /  timeout", [], WHITE)
    s.card(485, 92, 465, 303, "Learned recovery  /  +947 points", [], WHITE)
    s.text(56, 155, "At (4,2), facing the east wall:", size=15)
    s.card(
        58, 176, 310, 65, "climb", ["No effect away from the entrance."], RED_BG, RED
    )
    s.card(58, 258, 310, 65, "fwd", ["Bumps into the wall. No movement."], RED_BG, RED)
    s.arrow((200, 243), (200, 255), RED)
    s.line(369, 290, 412, 290, RED, 2)
    s.line(412, 290, 412, 207, RED, 2)
    s.arrow((412, 207), (370, 207), RED)
    s.text(56, 361, "The same two commands repeat until 120.", size=15, weight=700)
    s.text(511, 154, "Command 29: turn left instead of climbing.", size=15)

    def point(cell):
        x, y = cell
        return 579 + 57 * (x - 1), 184 + 52 * (4 - y)

    for x in range(1, 5):
        for y in range(1, 5):
            cell = [x, y]
            cx, cy = point(cell)
            fill = (
                GOLD_BG
                if cell == gold
                else RED_BG
                if cell in pits or cell == wumpus
                else TEAL_BG
                if tuple(cell) in route
                else PAPER
            )
            s.rect(cx - 25, cy - 23, 50, 46, fill, r=4, stroke=RULE)
            if cell in pits or cell == wumpus:
                s.text(
                    cx,
                    cy + 5,
                    "W" if cell == wumpus else "P",
                    size=14,
                    anchor="middle",
                    ink=RED,
                    weight=700,
                )
    for a, b in zip(route, route[1:]):
        x1, y1 = point(a)
        x2, y2 = point(b)
        start = ((2 * x1 + x2) / 3, (2 * y1 + y2) / 3)
        end = ((x1 + 2 * x2) / 3, (y1 + 2 * y2) / 3)
        s.arrow(start, end, TEAL)
    for cell in route:
        s.circle(*point(cell), 4, TEAL, stroke=WHITE)
    for cell, text, ink in (
        (gold, "Gold", GOLD),
        ((1, 1), "Exit", TEAL),
        (route[0], "Loop", RED),
    ):
        x, y = point(cell)
        s.text(x, y + 16, text, size=11, anchor="middle", ink=ink, weight=700)
    s.text(511, 382, "It picks up gold and exits at command 43.", size=14, weight=700)
    s.text(
        30,
        420,
        "Arrows show cell changes after the repeat. Hazards (W = Wumpus, P = pit) are shown only for the reader.",
        size=13,
        ink=MUTED,
    )
    s.save(root / "held-out-recovery-example.svg")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--results", type=Path, default=Path("results/timeout-recovery"))
    args = cli.parse_args()
    chart(args.results, "held-out-results.svg")
    example(args.results)


if __name__ == "__main__":
    main()
