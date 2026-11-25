"""Screen learning changes on validation maps before evaluating on fresh caves."""

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
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
from level_replay import LevelReplay
from primitive_memory import RawPerceptMemory
from randomized_lcs import IndexedLCS
from shared_rule_lcs import Settings, SharedRuleLCS, average_rule_tables


VALIDATION_SIZE = 300
TEST_SIZE = 600
VALIDATION_SEED, TEST_SEED = 74207, 180173
PENALTY_VALIDATION_SEED, PENALTY_TEST_SEED = 29947, 247393
PROGRESS_VALIDATION_SEED, PROGRESS_TEST_SEED = 447173, 338959
PROGRESS_VALIDATION_SIZE, PROGRESS_TEST_SIZE = 420, 2000
TIMEOUT_VALIDATION_SEED, TIMEOUT_TEST_SEED = 617383, 804589
TIMEOUT_VALIDATION_SIZE, TIMEOUT_TEST_SIZE = 600, 2000
TRAIN_CAP, EVAL_CAP = MAX_STEPS, 120
WIDE = Settings(replay=1, wider_memory=True)
BOOT = replace(WIDE, bootstrap_truncation=True)
LOOP = replace(BOOT, loop_memory=True)
COUNT = replace(BOOT, count_bonus=0.25)
LOW = replace(COUNT, epsilon_final=0.05)
LOW_CLEAN = replace(LOW, imitation=True, clean_replay=True)


@dataclass(frozen=True)
class Profile:
    settings: Settings
    reason: str
    curriculum: bool = False
    kind: str = "shared"
    members: int = 1
    level_rate: float = 0.0
    prioritize_levels: bool = True


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
        WIDE,
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
    "boot": Profile(
        BOOT, "Distinguish a time limit from a genuine terminal state.", curriculum=True
    ),
    "loop_only": Profile(
        replace(WIDE, loop_memory=True),
        "Learn a different action after observable stalling; no action filter.",
        curriculum=True,
    ),
    "boot_loop": Profile(
        LOOP,
        "Combine passive progress memory and correct treatment of the time limit.",
        curriculum=True,
    ),
    "boot_n1": Profile(
        replace(BOOT, n=1),
        "One-step credit control without count-based exploration.",
        curriculum=True,
    ),
    "boot_n4": Profile(
        replace(BOOT, n=4),
        "Four-step credit control without count-based exploration.",
        curriculum=True,
    ),
    "count_n1": Profile(
        replace(BOOT, n=1, count_bonus=0.25),
        "Compare count exploration with one-step credit.",
        curriculum=True,
    ),
    "count_n4": Profile(
        replace(BOOT, n=4, count_bonus=0.25),
        "Compare count exploration with four-step credit.",
        curriculum=True,
    ),
    "count_n12": Profile(
        replace(BOOT, count_bonus=0.25),
        "Compare count exploration with twelve-step credit.",
        curriculum=True,
    ),
    "boot_potential": Profile(
        replace(BOOT, potential_points=45),
        "Add annealed training-only potential for observed progress.",
        curriculum=True,
    ),
    "boot_balance": Profile(
        replace(BOOT, balanced_replay=True),
        "Retain rare longer successes by stratifying self-replay with the distance actually traveled to the pickup location.",
        curriculum=True,
    ),
    "count_loop": Profile(
        replace(COUNT, loop_memory=True),
        "Combine count exploration during training with a learned response to stalling.",
        curriculum=True,
    ),
    "count_balance": Profile(
        replace(COUNT, balanced_replay=True),
        "Use count exploration to find successes and balanced replay to retain longer trips.",
        curriculum=True,
    ),
    "count_sil": Profile(
        replace(COUNT, imitation=True),
        "Rehearse the original returns of actual successful trips, not only their bootstrapped estimates.",
        curriculum=True,
    ),
    "count_clean": Profile(
        replace(COUNT, imitation=True, clean_replay=True),
        "Remove exact observed cycles from successes before self-imitation.",
        curriculum=True,
    ),
    "count_low_noise": Profile(
        LOW,
        "Anneal uniform exploration so credit on successful long trajectories is not dominated by random moves.",
        curriculum=True,
    ),
    "count_potential": Profile(
        replace(COUNT, potential_points=80),
        "Test whether annealed novelty potential complements count-based action selection.",
        curriculum=True,
    ),
    "count_level25": Profile(
        COUNT,
        "Revisit a quarter of training layouts using TD learning potential.",
        curriculum=True,
        level_rate=0.25,
    ),
    "count_uniform25": Profile(
        COUNT,
        "Uniform-revisit control for the training-level selection study.",
        curriculum=True,
        level_rate=0.25,
        prioritize_levels=False,
    ),
    "low_level25": Profile(
        LOW,
        "Reuse promising training layouts after reducing action noise.",
        curriculum=True,
        level_rate=0.25,
    ),
    "low_clean": Profile(
        LOW_CLEAN,
        "Retain genuine successful paths while annealing exploration.",
        curriculum=True,
    ),
    "low_ensemble3": Profile(
        LOW,
        "Average identically defined rule weights from three independent action-exploration streams.",
        curriculum=True,
        members=3,
    ),
    "clean_ensemble3": Profile(
        LOW_CLEAN,
        "Test whether averaging reduces errors from self-imitation of lucky outcomes.",
        curriculum=True,
        members=3,
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
    if split in ("penalty", "progress", "timeout"):
        validation = distinct_layouts(
            notebook.random_layout,
            VALIDATION_SIZE,
            Random(PENALTY_VALIDATION_SEED),
            reserved,
        )
        test = distinct_layouts(
            notebook.random_layout, TEST_SIZE, Random(PENALTY_TEST_SEED), reserved
        )
    if split in ("progress", "timeout"):
        validation = distinct_layouts(
            notebook.random_layout,
            PROGRESS_VALIDATION_SIZE,
            Random(PROGRESS_VALIDATION_SEED),
            reserved,
        )
        test = distinct_layouts(
            notebook.random_layout,
            PROGRESS_TEST_SIZE,
            Random(PROGRESS_TEST_SEED),
            reserved,
        )
    if split == "timeout":
        validation = distinct_layouts(
            notebook.random_layout,
            TIMEOUT_VALIDATION_SIZE,
            Random(TIMEOUT_VALIDATION_SEED),
            reserved,
        )
        test = distinct_layouts(
            notebook.random_layout,
            TIMEOUT_TEST_SIZE,
            Random(TIMEOUT_TEST_SEED),
            reserved,
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
    name, seed, checkpoints, use_test, action_limit = job
    profile = PROFILES[name]
    if profile.kind == "exact":
        agent = IndexedLCS(RawPerceptMemory(), seed, RawPerceptMemory.ARITY)
    elif profile.kind == "notebook":
        agent = NB.RuleValues()
    else:
        agents = [
            SharedRuleLCS(seed + 10003 * m, profile.settings)
            for m in range(profile.members)
        ]
        agent = agents[0]
    levels = (
        LevelReplay(
            seed + 42001, profile.level_rate, prioritized=profile.prioritize_levels
        )
        if profile.level_rate
        else None
    )
    rng = Random(seed)
    actions = picked = gold = 0
    seconds, last = 0.0, 0
    validation = []
    for budget in checkpoints:
        t0 = perf_counter()
        for i, fresh in enumerate(
            training_stream(seed, checkpoints[-1], profile.curriculum)[last:budget],
            start=last,
        ):
            layout = levels.choose(fresh, i) if levels else fresh
            if profile.kind == "notebook":
                outcome, _, trace = NB.play(
                    layout, agent, rng, epsilon=0.4, learn=True, max_steps=TRAIN_CAP
                )
                picked += notebook_picked_gold(trace)
                gold += outcome == "gold"
                actions += len(trace)
            elif profile.kind == "exact":
                cave = measured_cave(layout)
                agent.episode(cave)
                picked += cave.has_gold
                gold += cave.has_gold and cave.escaped
                actions += cave.steps
            else:
                for member in agents:
                    cave = measured_cave(layout)
                    cap = (
                        min(TRAIN_CAP, action_limit - actions)
                        if action_limit
                        else TRAIN_CAP
                    )
                    member.train_episode(cave, cap)
                    picked += cave.has_gold
                    gold += cave.has_gold and cave.escaped
                    actions += cave.steps
                if levels:
                    signal = mean(member.last_learning_signal for member in agents)
                    levels.record(layout, i, signal)
                if action_limit and actions == action_limit:
                    break
        seconds += perf_counter() - t0
        if action_limit and actions != action_limit:
            raise ValueError(
                "layout budget exhausted before reaching the matched action count"
            )
        last = i + 1
        evaluated = average_rule_tables(agents) if profile.members > 1 else agent
        rules = (
            len(evaluated.weights) * 6
            if isinstance(evaluated, SharedRuleLCS)
            else (
                len(evaluated.rules) if profile.kind == "exact" else len(evaluated.mean)
            )
        )
        metrics = summarize([rollout(evaluated, profile, layout)[0] for layout in VAL])
        validation.append(
            {
                "policy": name,
                "seed": seed,
                "episodes": last,
                "total_training_episodes": last * profile.members,
                "level_replays": levels.replays if levels else 0,
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
                **rollout(evaluated, profile, layout)[0],
            }
            for i, layout in enumerate(TEST, 1)
        ]
        example = select_trace(rows, evaluated, profile)
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
    cli.add_argument(
        "--budget", action="append", default=[], metavar="PROFILE=EPISODES"
    )
    cli.add_argument(
        "--match-training-actions",
        type=Path,
        help="CSV with a reference profile's per-seed training action counts",
    )
    cli.add_argument("--reference-policy", default="clean_ensemble3")
    cli.add_argument("--split", choices=("main", "penalty", "progress"), default="main")
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
    budgets = {}
    try:
        for item in args.budget:
            name, count = item.split("=", 1)
            if name not in args.profiles or name in budgets or int(count) < 1:
                raise ValueError(item)
            budgets[name] = [int(count)]
    except ValueError:
        cli.error("each --budget must be a unique selected PROFILE=POSITIVE_EPISODES")
    limits = {}
    if args.match_training_actions:
        if (
            len(args.profiles) != 1
            or PROFILES[args.profiles[0]].kind != "shared"
            or PROFILES[args.profiles[0]].members != 1
        ):
            cli.error("action matching requires one single-member shared LCS")
        if len(args.checkpoints) != 1 or budgets:
            cli.error(
                "use one --checkpoints upper bound and no --budget for action matching"
            )
        with args.match_training_actions.open() as file:
            for row in csv.DictReader(file):
                if row["policy"] == args.reference_policy:
                    limits[int(row["seed"])] = int(row["training_actions"])
        if not set(args.seeds) <= limits.keys() or any(
            limits[s] < 1 for s in args.seeds
        ):
            cli.error("missing or invalid reference action count")
    nb = load_notebook(args.notebook)
    validation, test, reserved = evaluation_maps(nb, args.split)
    jobs = [
        (name, seed, budgets.get(name, args.checkpoints), args.test, limits.get(seed))
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
        "validation_seed": {
            "main": VALIDATION_SEED,
            "penalty": PENALTY_VALIDATION_SEED,
            "progress": PROGRESS_VALIDATION_SEED,
        }[args.split],
        "test_seed": {
            "main": TEST_SEED,
            "penalty": PENALTY_TEST_SEED,
            "progress": PROGRESS_TEST_SEED,
        }[args.split],
        "training_layout_seed_base": TRAIN_SEED_BASE,
        "seeds": args.seeds,
        "checkpoints": args.checkpoints,
        "profile_budgets": budgets,
        "matched_training_actions": {seed: limits[seed] for seed in args.seeds}
        if limits
        else {},
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
                for (name, seed, *_), (_, rows, _) in zip(jobs, results, strict=True)
            ],
        )
        write_json(
            args.output / "traces.json",
            {
                f"{name}:{seed}": example
                for (name, seed, *_), (_, _, example) in zip(jobs, results, strict=True)
                if example is not None
            },
        )


if __name__ == "__main__":
    main()
