"""Screen learning changes on validation maps before evaluating on fresh caves."""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from random import Random
from statistics import mean
from time import perf_counter

from benchmark_navigation import (
    MeasuredAdapter,
    curriculum_layouts,
    gold_distance,
    make_maps,
    measured_cave,
    notebook_picked_gold,
    summary,
)
from benchmark_randomized import TRAIN_SEED_BASE, distinct_layouts, training_layouts
from benchmark_transfer import load_notebook, run_notebook_policy, write_csv, write_json
from classifiers import MAX_STEPS
from primitive_memory import RawPerceptMemory
from randomized_lcs import IndexedLCS
from shared_rule_lcs import Settings, SharedRuleLCS


VALIDATION_SIZE = 300
TEST_SIZE = 600
VALIDATION_SEED, TEST_SEED = 74207, 180173
PENALTY_VALIDATION_SEED, PENALTY_TEST_SEED = 29947, 247393
TRAIN_CAP, EVAL_CAP = MAX_STEPS, 120


@dataclass(frozen=True)
class Profile:
    settings: Settings
    reason: str
    curriculum: bool = False
    kind: str = "shared"


PROFILES = {
    "exact": Profile(
        Settings(),
        "Exact covering control: little sharing between memory records.",
        kind="exact",
    ),
    "shared_td": Profile(
        Settings(n=1, gamma=0.9), "Allow partial matches without lengthening credit."
    ),
    "shared_n12": Profile(
        Settings(),
        "Propagate a successful exit to earlier actions over a longer horizon.",
    ),
    "early25": Profile(
        Settings(penalty=25),
        "Make retreat without leaving the entrance less attractive in training.",
    ),
    "early100": Profile(
        Settings(penalty=100),
        "Measure whether a stronger early-retreat penalty only causes stalling.",
    ),
    "early250": Profile(
        Settings(penalty=250),
        "Check the middle of the dose-response for premature retreat.",
    ),
    "early500": Profile(
        Settings(penalty=500),
        "Check whether a half-death penalty promotes useful exploration.",
    ),
    "early1000": Profile(
        Settings(penalty=1000),
        "Train with premature retreat and death both worth -1001 after the action cost.",
    ),
    "novel20": Profile(
        Settings(novel_reward=20),
        "Reward reaching a new square, not waiting or repeated motion.",
    ),
    "early25_novel20": Profile(
        Settings(penalty=25, novel_reward=20),
        "Test whether both exploration signals help beyond the individual changes.",
    ),
    "self_replay": Profile(
        Settings(replay=1),
        "Revisit actual successful episodes to preserve rare useful experience.",
        curriculum=True,
    ),
    "shared_curriculum": Profile(
        Settings(),
        "Control for self-replay: same easy start, no extra replay.",
        curriculum=True,
    ),
    "replay_uniform": Profile(
        Settings(replay=1),
        "Check whether self-replay can discover useful policies without an easy start.",
    ),
    "replay4": Profile(
        Settings(replay=4),
        "Give rare actual successes more repeated credit than with one replay.",
        curriculum=True,
    ),
    "replay_early100": Profile(
        Settings(replay=1, penalty=100),
        "Test whether self-replay and an early penalty are complementary.",
        curriculum=True,
    ),
    "replay_wide": Profile(
        Settings(replay=1, wider_memory=True),
        "Expose farther raw observations to distinguish return situations.",
        curriculum=True,
    ),
    "replay_wide_early1000": Profile(
        Settings(replay=1, wider_memory=True, penalty=1000),
        "Check the death-sized penalty on the strongest confirmed learner as a separate paired ablation.",
        curriculum=True,
    ),
    "replay_long": Profile(
        Settings(replay=1, n=32, gamma=0.995),
        "Check whether much longer credit helps the longer successful trips.",
        curriculum=True,
    ),
    "notebook": Profile(
        Settings(),
        "Planner baseline; reference only, not a learned primitive solution.",
        kind="notebook",
    ),
}


def evaluation_maps(notebook, split="main"):
    old = make_maps(notebook.random_layout, 300, 500)
    reserved = set(old.reserved)
    validation = distinct_layouts(
        notebook.random_layout, VALIDATION_SIZE, Random(VALIDATION_SEED), reserved
    )
    test = distinct_layouts(
        notebook.random_layout, TEST_SIZE, Random(TEST_SEED), reserved
    )
    if split == "penalty":
        validation = distinct_layouts(
            notebook.random_layout,
            VALIDATION_SIZE,
            Random(PENALTY_VALIDATION_SEED),
            reserved,
        )
        test = distinct_layouts(
            notebook.random_layout, TEST_SIZE, Random(PENALTY_TEST_SEED), reserved
        )
    return validation, test, frozenset(reserved)


def worker_init(path, validation, test, reserved):
    global NB, VAL, TEST, RESERVED
    NB, VAL, TEST, RESERVED = load_notebook(path), validation, test, reserved


@lru_cache(maxsize=16)
def training_stream(seed, limit, curriculum):
    uniform = training_layouts(NB.random_layout, seed, limit, RESERVED)
    if curriculum:
        return curriculum_layouts(NB.random_layout, seed, uniform, RESERVED)
    return uniform


def rollout(agent, profile, layout, *, trace=False):
    cave = measured_cave(layout)
    if profile.kind == "notebook":
        result = run_notebook_policy(NB, agent, layout, adapter=MeasuredAdapter(cave))
        outcome, score, records = result.outcome, result.score, result.trace
    elif profile.kind == "exact":
        outcome, score, _, records, _ = agent.greedy_episode(
            cave, EVAL_CAP, trace=trace
        )
    else:
        outcome, score, _, records = agent.evaluate(cave, EVAL_CAP, trace=trace)
    row = {
        "outcome": outcome,
        "score": score,
        "actions": cave.steps,
        "picked_gold": int(cave.has_gold),
        "distinct_cells": len(cave.reached),
        "immediate_retreat": int(outcome == "retreat" and cave.steps == 1),
        "stalled_timeout": int(
            outcome == "timeout" and cave.steps - cave.last_new >= 30
        ),
        "gold_distance": gold_distance(layout),
    }
    return row, records if trace else None


def summarize(rows):
    metrics = summary(rows)
    far = [row for row in rows if row["gold_distance"] >= 2]
    metrics["far_caves"] = len(far)
    metrics["far_gold"] = sum(row["outcome"] == "gold" for row in far)
    metrics["end_pos_only_start"] = sum(row["distinct_cells"] == 1 for row in rows)
    return metrics


def select_trace(rows, agent, profile):
    match = max(
        (r for r in rows if r["outcome"] == "gold" and r["gold_distance"] >= 2),
        key=lambda r: (r["gold_distance"], -r["cave_id"]),
        default=None,
    )
    if match is None:
        return None
    layout = TEST[match["cave_id"] - 1]
    row, steps = rollout(agent, profile, layout, trace=True)
    return {"cave_id": match["cave_id"], "layout": layout, **row, "trace": steps}


def execute_job(job):
    name, seed, checkpoints, use_test = job
    profile = PROFILES[name]
    if profile.kind == "exact":
        agent = IndexedLCS(RawPerceptMemory(), seed, RawPerceptMemory.ARITY)
    elif profile.kind == "notebook":
        agent = NB.RuleValues()
    else:
        agent = SharedRuleLCS(seed, profile.settings)
    rng = Random(seed)
    actions = picked = gold = 0
    seconds, last = 0.0, 0
    validation = []
    for budget in checkpoints:
        t0 = perf_counter()
        for layout in training_stream(seed, checkpoints[-1], profile.curriculum)[
            last:budget
        ]:
            if profile.kind == "notebook":
                outcome, _, trace = NB.play(
                    layout, agent, rng, epsilon=0.4, learn=True, max_steps=TRAIN_CAP
                )
                picked += notebook_picked_gold(trace)
                gold += outcome == "gold"
                actions += len(trace)
            else:
                cave = measured_cave(layout)
                if profile.kind == "exact":
                    agent.episode(cave)
                else:
                    agent.train_episode(cave, TRAIN_CAP)
                picked += cave.has_gold
                gold += cave.has_gold and cave.escaped
                actions += cave.steps
        seconds += perf_counter() - t0
        last = budget
        rules = (
            len(agent.weights) * 6
            if isinstance(agent, SharedRuleLCS)
            else (len(agent.rules) if profile.kind == "exact" else len(agent.mean))
        )
        metrics = summarize([rollout(agent, profile, layout)[0] for layout in VAL])
        validation.append(
            {
                "policy": name,
                "seed": seed,
                "episodes": budget,
                "training_actions": actions,
                "training_pickups": picked,
                "training_gold": gold,
                "rules": rules,
                "seconds": round(seconds, 3),
                **metrics,
            }
        )
    rows = []
    example = None
    if use_test:
        rows = [
            {
                "policy": name,
                "seed": seed,
                "cave_id": i,
                **rollout(agent, profile, layout)[0],
            }
            for i, layout in enumerate(TEST, 1)
        ]
        example = select_trace(rows, agent, profile)
    return validation, rows, example


def show_validation(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["policy"], row["episodes"]), []).append(row)
    for (policy, episodes), data in groups.items():
        print(
            f"{policy:>23} {episodes:>6} episodes | score {mean(r['mean_score'] for r in data):7.1f} "
            f"| gold {mean(r['gold'] / r['caves'] for r in data):6.1%} "
            f"| far gold {mean(r['far_gold'] / r['far_caves'] for r in data):6.1%} "
            f"| first retreat {mean(r['immediate_retreat'] / r['caves'] for r in data):6.1%}",
            flush=True,
        )


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--profiles", nargs="+", choices=PROFILES, required=True)
    cli.add_argument("--notebook", type=Path)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    cli.add_argument("--checkpoints", type=int, nargs="+", default=[1600, 5000, 10000])
    cli.add_argument("--workers", type=int, default=8)
    cli.add_argument("--split", choices=("main", "penalty"), default="main")
    cli.add_argument(
        "--test",
        action="store_true",
        help="evaluate the reserved test set after training",
    )
    args = cli.parse_args()
    if min(args.checkpoints) < 1 or args.checkpoints != sorted(set(args.checkpoints)):
        cli.error("checkpoints must be positive and strictly increasing")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.profiles)) != len(
        args.profiles
    ):
        cli.error("seeds and profiles must not repeat")
    if args.workers < 1:
        cli.error("at least one worker is required")
    nb = load_notebook(args.notebook)
    validation, test, reserved = evaluation_maps(nb, args.split)
    jobs = [
        (name, seed, args.checkpoints, args.test)
        for name in args.profiles
        for seed in args.seeds
    ]
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=worker_init,
        initargs=(args.notebook, validation, test, reserved),
    ) as pool:
        results = list(pool.map(execute_job, jobs))
    curve = [r for rows, _, _ in results for r in rows]
    params = {
        "python": sys.version.split()[0],
        "split": args.split,
        "validation_seed": PENALTY_VALIDATION_SEED
        if args.split == "penalty"
        else VALIDATION_SEED,
        "test_seed": PENALTY_TEST_SEED if args.split == "penalty" else TEST_SEED,
        "training_layout_seed_base": TRAIN_SEED_BASE,
        "seeds": args.seeds,
        "checkpoints": args.checkpoints,
        "training_cap": TRAIN_CAP,
        "evaluation_cap": EVAL_CAP,
        "phase": "test" if args.test else "validation",
        "profiles": {name: asdict(PROFILES[name]) for name in args.profiles},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "validation.csv", curve)
    write_json(args.output / "config.json", params)
    write_csv(
        args.output / "layouts.csv",
        [
            {"split": split, "cave_id": i, "layout": json.dumps(layout)}
            for split, layouts in (("validation", validation), ("test", test))
            if split == "validation" or args.test
            for i, layout in enumerate(layouts, 1)
        ],
    )
    show_validation(curve)
    if args.test:
        all_rows = [r for _, rows, _ in results for r in rows]
        write_csv(args.output / "test-episodes.csv", all_rows)
        write_csv(
            args.output / "test-summary.csv",
            [
                {"policy": name, "seed": seed, **summarize(rows)}
                for (name, seed, _, _), (_, rows, _) in zip(jobs, results, strict=True)
            ],
        )
        write_json(
            args.output / "traces.json",
            {
                f"{name}:{seed}": example
                for (name, seed, _, _), (_, _, example) in zip(
                    jobs, results, strict=True
                )
                if example is not None
            },
        )


if __name__ == "__main__":
    main()
