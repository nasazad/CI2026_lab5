"""Reproduce the fixed-cave transfer experiment from issue #5."""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from random import Random
from statistics import mean
from types import FunctionType, ModuleType
from urllib.request import urlopen

from cave import Cave
from classifiers import HAND_WRITTEN, run_episode
from learning import DeadReckoning, LearningCS


UPSTREAM_REVISION = "b0117d6eaa1b24580b930c29a6027a78d697ad99"
NOTEBOOK_REVISION = "6b19ce0e2d4279f2157217386e4730947a332ef9"
NOTEBOOK_URL = (
    "https://raw.githubusercontent.com/tomg55555/CI2026_lab5/"
    f"{NOTEBOOK_REVISION}/Wumpus_World.ipynb"
)
NOTEBOOK_SHA256 = "8deed33226b39153cb0fd7dffe0fdcd4a6dc7ec233f26db58d54a61ace82a794"
FIXED_LAYOUT = ((2, 3), (1, 3), ((3, 1), (3, 3), (4, 4)))
FIXED_SEEDS = tuple(range(10))
FIXED_EPISODES = 1500
LAYOUT_SEED, ACTION_SEED, TEST_SEED = 5, 19, 50
BASELINE_EPISODES, TEST_CAVES, TEST_ACTION_LIMIT = 1600, 500, 120
OUTCOMES = ("gold", "death", "retreat", "timeout", "no_matching_action")
LABELS = {
    "fixed_handwritten": "Handwritten (fixed cave)",
    "fixed_lcs": "Learned LCS (fixed cave)",
    "randomized_baseline": "Randomized-cave baseline",
}


def load_notebook(path=None):
    """Load only the definitions from the exact audited notebook revision."""
    if path is None:
        with urlopen(NOTEBOOK_URL, timeout=30) as response:
            data = response.read()
    else:
        data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != NOTEBOOK_SHA256:
        raise ValueError("Notebook checksum mismatch; use the pinned revision.")
    code = [
        "".join(c["source"])
        for c in json.loads(data)["cells"]
        if c["cell_type"] == "code"
    ]
    notebook = ModuleType("_transfer_baseline")
    sys.modules[notebook.__name__] = notebook
    # These five cells define the environment and policy. Do not run the demo or experiments.
    for source in code[:5]:
        exec(compile(source, NOTEBOOK_URL, "exec"), vars(notebook))
    return notebook


def make_layouts(random_layout, train_count=BASELINE_EPISODES, test_count=TEST_CAVES):
    train_rng, test_rng = Random(LAYOUT_SEED), Random(TEST_SEED)
    train = [random_layout(train_rng) for _ in range(train_count)]
    reserved = set(train) | {FIXED_LAYOUT}
    test = []
    while len(test) < test_count:
        layout = random_layout(test_rng)
        if layout not in reserved:
            reserved.add(layout)
            test.append(layout)
    return train, test


def make_cave(layout):
    gold, wumpus, pits = layout
    return Cave(gold=gold, wumpus=wumpus, pits=pits)


def terminal_outcome(cave):
    if cave.dead:
        return "death"
    if cave.escaped:
        return "gold" if cave.has_gold else "retreat"
    return None


class NotebookCaveAdapter:
    """Expose the notebook's Cave API over the shared simulator."""

    def __init__(self, gold, wumpus, pits):
        self.cave = make_cave((gold, wumpus, pits))
        self.rewards = []

    def perceive(self):
        return set(self.cave.sensors())

    def step(self, action):
        sensors, reward, _ = self.cave.step(action)
        self.rewards.append(reward)
        return set(sensors), reward, terminal_outcome(self.cave)


@dataclass
class Rollout:
    outcome: str
    score: int
    trace: list
    unmatched_messages: tuple = ()


def run_fixed_policy(rules, layout, *, dead_reckoning=False, limit=TEST_ACTION_LIMIT):
    cave = make_cave(layout)
    memory = DeadReckoning() if dead_reckoning else None
    score, trace = run_episode(rules, cave, memory=memory, max_steps=limit)
    outcome = terminal_outcome(cave)
    unmatched = ()
    if outcome is None:
        outcome = "timeout" if len(trace) == limit else "no_matching_action"
        if outcome == "no_matching_action":
            posted = trace[-1].posted if trace else frozenset()
            known = memory.messages() if memory else frozenset()
            unmatched = tuple(sorted(cave.sensors() | posted | known))
    steps = [
        {
            "square": s.square,
            "heading": s.heading,
            "messages": sorted(s.msgs),
            "action": s.winner.effector,
            "reward": s.reward,
        }
        for s in trace
    ]
    return Rollout(outcome, score, steps, unmatched)


def run_notebook_policy(notebook, values, layout, *, limit=TEST_ACTION_LIMIT):
    adapter = NotebookCaveAdapter(*layout)

    # Preserve the notebook's policy and interaction loop. Replace only its
    # Cave constructor for this evaluation, without mutating the notebook.
    play = FunctionType(
        notebook.play.__code__,
        {**vars(notebook), "Cave": lambda *_: adapter},
        argdefs=notebook.play.__defaults__,
    )
    outcome, score, trace = play(
        layout, values, epsilon=0, learn=False, max_steps=limit
    )
    steps = [
        {"square": square, "percepts": percepts, "action": action, "reward": reward}
        for (square, percepts, action), reward in zip(
            trace, adapter.rewards, strict=True
        )
    ]
    return Rollout(outcome, score, steps)


def train_notebook(notebook, layouts):
    values, rng = notebook.RuleValues(), Random(ACTION_SEED)
    for layout in layouts:
        notebook.play(layout, values, rng, epsilon=0.4, learn=True)
    return values


def train_fixed(seed):
    cave = make_cave(FIXED_LAYOUT)
    lcs = LearningCS(memory=DeadReckoning(), seed=seed)
    for _ in range(FIXED_EPISODES):
        lcs.episode(cave)
    return lcs


def evaluate_policy(name, seed, evaluate, layouts):
    rows, cases = [], {}
    for cave_id, layout in enumerate(layouts, 1):
        result = evaluate(layout)
        rows.append(
            {
                "policy": name,
                "seed": seed,
                "cave_id": cave_id,
                "outcome": result.outcome,
                "score": result.score,
                "actions": len(result.trace),
            }
        )
        if seed == 0 or name == "randomized_baseline":
            cases[cave_id] = result
    counts = Counter(row["outcome"] for row in rows)
    summary = {
        "policy": name,
        "seed": seed,
        "caves": len(rows),
        "mean_score": mean(row["score"] for row in rows),
        **{outcome: counts[outcome] for outcome in OUTCOMES},
    }
    return rows, summary, cases


def select_examples(layouts, fixed, baseline):
    examples = {}
    for outcome in ("death", "no_matching_action"):
        for cave_id, a in fixed.items():
            b = baseline[cave_id]
            if a.outcome == outcome and b.outcome == "gold":
                gold, wumpus, pits = layouts[cave_id - 1]
                examples[outcome] = {
                    "cave_id": cave_id,
                    "gold": gold,
                    "wumpus": wumpus,
                    "pits": pits,
                    "fixed_lcs": vars(a),
                    "randomized_baseline": vars(b),
                }
                break
    return examples


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def write_chart(path, summary):
    """Render an accessible, dependency-free outcome distribution chart."""
    colors = dict(
        zip(OUTCOMES, ("#207461", "#ae3d3a", "#466e9e", "#916100", "#586471"))
    )
    start, width = 236, 592
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 920 335"'
        ' role="img" aria-labelledby="title desc">',
        '<title id="title">Outcomes on the same 500 unseen caves</title>',
        '<desc id="desc">Stacked bars show gold exits, deaths, retreats, timeouts, and stops '
        "without a matching rule. The table in the report gives exact counts.</desc>",
        '<rect width="920" height="335" rx="14" fill="#f8fafc"/>',
        '<g font-family="system-ui, sans-serif" fill="#182737">',
        '<text x="24" y="36" font-size="24" font-weight="700">'
        "Fixed-map transfer to unseen caves</text>",
        '<text x="24" y="59" font-size="14">Same 500 caves per policy · shared simulator, '
        "120-action cap · fixed LCS averaged over 10 seeds</text>",
    ]
    names = ("Gold", "Death", "Retreat", "Timeout", "No matching rule")
    for i, (outcome, name) in enumerate(zip(OUTCOMES, names)):
        x = (24, 135, 254, 369, 496)[i]
        parts += [
            f'<rect x="{x}" y="77" width="14" height="14" rx="2" fill="{colors[outcome]}"/>',
            f'<text x="{x + 21}" y="89" font-size="13">{name}</text>',
        ]
    parts.append(
        '<text x="864" y="100" text-anchor="middle" font-size="12">Mean score</text>'
    )
    for i, policy in enumerate(
        ("fixed_handwritten", "fixed_lcs", "randomized_baseline")
    ):
        group = [row for row in summary if row["policy"] == policy]
        y, x = 116 + i * 61, start
        parts.append(
            f'<text x="24" y="{y + 24}" font-size="15">{LABELS[policy]}</text>'
        )
        for outcome in OUTCOMES:
            share = mean(row[outcome] / row["caves"] for row in group)
            if share == 0:
                continue
            w = width * share
            parts.append(
                f'<rect x="{x:.2f}" y="{y}" width="{w:.2f}" height="36" fill="{colors[outcome]}"/>'
            )
            if share >= 0.085:
                parts.append(
                    f'<text x="{x + w / 2:.2f}" y="{y + 23}" fill="white" font-size="13"'
                    f' font-weight="650" text-anchor="middle">{share:.1%}</text>'
                )
            x += w
        score = mean(row["mean_score"] for row in group)
        parts.append(
            f'<text x="864" y="{y + 24}" text-anchor="middle" font-size="15">{score:.1f}</text>'
        )
    for i in range(5):
        x = start + i * width / 4
        parts.append(
            f'<text x="{x:.0f}" y="306" text-anchor="middle" font-size="12">{25 * i}%</text>'
        )
    parts.append("</g></svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebook", type=Path, help="use a local copy of the pinned notebook"
    )
    parser.add_argument("--output", type=Path, default=Path("results/issue-5"))
    args = parser.parse_args()

    notebook = load_notebook(args.notebook)
    train, test = make_layouts(notebook.random_layout)
    values = train_notebook(notebook, train)
    reference = run_fixed_policy(HAND_WRITTEN, FIXED_LAYOUT)
    assert (reference.outcome, reference.score, len(reference.trace)) == (
        "gold",
        988,
        12,
    )
    rows, stats, checks = [], [], {}

    def record(name, seed, evaluate):
        data, summary, cases = evaluate_policy(name, seed, evaluate, test)
        rows.extend(data)
        stats.append(summary)
        print(summary, flush=True)
        return cases

    before = values.mean.copy(), values.count.copy()
    baseline = record(
        "randomized_baseline",
        ACTION_SEED,
        lambda layout: run_notebook_policy(notebook, values, layout),
    )
    assert before == (values.mean, values.count)
    record(
        "fixed_handwritten", "", lambda layout: run_fixed_policy(HAND_WRITTEN, layout)
    )

    for seed in FIXED_SEEDS:
        lcs = train_fixed(seed)
        original = run_fixed_policy(lcs.rules, FIXED_LAYOUT, dead_reckoning=True)
        assert (original.outcome, original.score, len(original.trace)) == (
            "gold",
            988,
            12,
        ), seed
        checks[seed] = {
            "score": original.score,
            "actions": len(original.trace),
            "rules": len(lcs.rules),
        }
        snapshot = [
            (rule.condition, rule.action, rule.strength, rule.stamp)
            for rule in lcs.rules
        ]
        fixed = record(
            "fixed_lcs",
            seed,
            lambda layout: run_fixed_policy(lcs.rules, layout, dead_reckoning=True),
        )
        assert snapshot == [
            (rule.condition, rule.action, rule.strength, rule.stamp)
            for rule in lcs.rules
        ]
        if seed == FIXED_SEEDS[0]:
            examples = select_examples(test, fixed, baseline)

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output / "layouts.csv",
        [
            {
                "cave_id": i,
                "gold": json.dumps(gold),
                "wumpus": json.dumps(wumpus),
                "pits": json.dumps(pits),
            }
            for i, (gold, wumpus, pits) in enumerate(test, 1)
        ],
    )
    write_csv(args.output / "episodes.csv", rows)
    write_csv(args.output / "summary.csv", stats)
    write_json(args.output / "traces.json", examples)
    write_json(
        args.output / "config.json",
        {
            "fixed_lcs_source_revision": UPSTREAM_REVISION,
            "baseline_notebook_revision": NOTEBOOK_REVISION,
            "baseline_notebook_sha256": NOTEBOOK_SHA256,
            "python": sys.version.split()[0],
            "fixed_training": {
                "seeds": FIXED_SEEDS,
                "episodes_per_seed": FIXED_EPISODES,
                "compaction": False,
                "lab_map_checks": checks,
            },
            "baseline_training": {
                "layout_seed": LAYOUT_SEED,
                "action_seed": ACTION_SEED,
                "episodes": len(train),
                "distinct_layouts": len(set(train)),
                "epsilon": 0.4,
            },
            "evaluation": {
                "layout_seed": TEST_SEED,
                "layouts": len(test),
                "action_limit": TEST_ACTION_LIMIT,
                "exclude": ["lab map", "baseline training layouts"],
                "exploration": False,
                "learning": False,
            },
        },
    )
    write_chart(args.output / "outcomes.svg", stats)


if __name__ == "__main__":
    main()
