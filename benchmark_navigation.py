"""Test whether an LCS learns primitive navigation and strategy without route hints."""

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from random import Random
from statistics import mean
from time import perf_counter

from benchmark_randomized import (
    TRAIN_SEED_BASE,
    distinct_layouts,
    make_split,
    qualifying_extensions,
    training_layouts,
)
from benchmark_transfer import (
    NOTEBOOK_REVISION,
    OUTCOMES,
    TEST_ACTION_LIMIT,
    UPSTREAM_REVISION,
    NotebookCaveAdapter,
    load_notebook,
    run_notebook_policy,
    write_csv,
    write_json,
)
from cave import Cave, START
from classifiers import MAX_STEPS
from primitive_memory import RawPerceptMemory
from randomized_lcs import IndexedLCS, RelativeInputs


RELATIVE = "relative_hints_lcs"
RAW = "raw_memory_lcs"
CURRICULUM = "raw_curriculum_lcs"
NOTEBOOK = "notebook_baseline"
POLICIES = (RELATIVE, RAW, CURRICULUM, NOTEBOOK)
RAW_POLICIES = (RAW, CURRICULUM)
VALIDATION_SEED, TEST_SEED = 1481, 36001
CURRICULUM_SEED = 37400
EASY_END, NEAR_END = 400, 1600
STALL_LENGTH = 30
CHECKPOINTS = (400, 1600, 5000, 10000)


@dataclass
class Maps:
    validation: tuple
    test: tuple
    reserved: frozenset


def make_maps(random_layout, validation_size, test_size):
    # Also reserve every earlier split that informed the experiment design.
    previous = make_split(random_layout, 300, 500, climb_ablation=True)
    reserved = set(previous.reserved)
    validation = distinct_layouts(
        random_layout, validation_size, Random(VALIDATION_SEED), reserved
    )
    test = distinct_layouts(random_layout, test_size, Random(TEST_SEED), reserved)
    return Maps(validation, test, frozenset(reserved))


def gold_distance(layout):
    x, y = layout[0]
    return abs(x - START[0]) + abs(y - START[1])


def easy_layout(layout, stage):
    """Training-layout selection, never a policy input or action constraint."""
    distance = gold_distance(layout)
    if distance == 1:
        return True
    if stage == 1 or distance != 2:
        return False
    gold, wumpus, pits = layout
    return any(
        abs(x - gold[0]) + abs(y - gold[1]) == 1
        and (x, y) != wumpus
        and (x, y) not in pits
        for x, y in ((2, 1), (1, 2))
    )


def curriculum_layouts(random_layout, seed, uniform, reserved):
    rng = Random(CURRICULUM_SEED + seed)
    result = list(uniform)
    for i in range(min(len(uniform), NEAR_END)):
        while True:
            layout = random_layout(rng)
            if layout not in reserved and easy_layout(layout, 1 if i < EASY_END else 2):
                result[i] = layout
                break
    # All other episodes use precisely the same layouts as the uniform control.
    return tuple(result)


class MeasuredCave(Cave):
    """Collect metrics from simulator truth only after commands are selected."""

    def reset(self):
        sensors = super().reset()
        self.reached = {START}
        self.steps = self.last_new = 0
        return sensors

    def step(self, action):
        result = super().step(action)
        self.steps += 1
        if self.pos not in self.reached:
            self.reached.add(self.pos)
            self.last_new = self.steps
        return result


class MeasuredAdapter(NotebookCaveAdapter):
    def __init__(self, cave):
        self.cave = cave
        self.rewards = []


def measured_cave(layout):
    gold, wumpus, pits = layout
    return MeasuredCave(wumpus=wumpus, gold=gold, pits=pits)


@dataclass
class Snapshot:
    rules: list
    random_state: object
    actions: int
    pickups: int
    gold_exits: int
    train_seconds: float


def worker_init(path, maps):
    global NB, MAPS
    NB, MAPS = load_notebook(path), maps


@lru_cache(maxsize=12)
def stream(seed, count, curriculum):
    uniform = training_layouts(NB.random_layout, seed, count, MAPS.reserved)
    if curriculum:
        return curriculum_layouts(NB.random_layout, seed, uniform, MAPS.reserved)
    return uniform


def notebook_rules():
    return (NB.RETREAT, NB.RISK_EARLY, NB.RISK_LATE)


def restore(policy, seed, snapshot):
    if policy == NOTEBOOK:
        model, rng = NB.RuleValues(), Random(seed)
        if snapshot is not None:
            for rule, (value, count) in zip(
                notebook_rules(), snapshot.rules, strict=True
            ):
                model.mean[rule], model.count[rule] = value, count
    else:
        memory = RelativeInputs(NB) if policy == RELATIVE else RawPerceptMemory()
        arity = 3 if policy == RELATIVE else RawPerceptMemory.ARITY
        model = IndexedLCS(
            memory, seed, arity, rules=[] if snapshot is None else snapshot.rules
        )
        if snapshot is not None:
            model.t = snapshot.actions
        rng = model.rng
    if snapshot is not None:
        rng.setstate(snapshot.random_state)
    return model, rng


def notebook_picked_gold(trace):
    return any(
        action == "grab" and "glitter" in percepts for _, percepts, action in trace
    )


def train_chunk(task):
    policy, seed, previous, start, end, total = task
    model, rng = restore(policy, seed, previous)
    actions = previous.actions if previous else 0
    pickups = previous.pickups if previous else 0
    gold_exits = previous.gold_exits if previous else 0
    seconds = previous.train_seconds if previous else 0.0
    layouts = stream(seed, total, policy == CURRICULUM)[start:end]
    t0 = perf_counter()
    for layout in layouts:
        if policy == NOTEBOOK:
            outcome, _, trace = NB.play(
                layout, model, rng, epsilon=0.4, learn=True, max_steps=MAX_STEPS
            )
            actions += len(trace)
            pickups += notebook_picked_gold(trace)
            gold_exits += outcome == "gold"
        else:
            # Use the shared simulator. Metrics do not feed back into the agent.
            gold, wumpus, pits = layout
            cave = Cave(gold=gold, wumpus=wumpus, pits=pits)
            model.episode(cave)
            pickups += cave.has_gold
            gold_exits += cave.escaped and cave.has_gold
    if policy != NOTEBOOK:
        actions = model.t
    rules = (
        [(model.mean[r], model.count[r]) for r in notebook_rules()]
        if policy == NOTEBOOK
        else model.rules
    )
    snapshot = Snapshot(
        rules,
        rng.getstate(),
        actions,
        pickups,
        gold_exits,
        seconds + perf_counter() - t0,
    )
    rows = [evaluate(policy, model, layout)[0] for layout in MAPS.validation]
    return snapshot, {
        "policy": policy,
        "seed": seed,
        "episodes": end,
        "training_actions": actions,
        "training_pickups": pickups,
        "training_gold": gold_exits,
        "rules": len(rules),
        "train_seconds": round(snapshot.train_seconds, 3),
        **summary(rows),
    }


def evaluate(policy, model, layout, *, trace=False):
    cave = measured_cave(layout)
    if policy == NOTEBOOK:
        result = run_notebook_policy(NB, model, layout, adapter=MeasuredAdapter(cave))
        outcome, score, steps, unmatched = (
            result.outcome,
            result.score,
            result.trace,
            [],
        )
    else:
        outcome, score, _, steps, unmatched = model.greedy_episode(
            cave, TEST_ACTION_LIMIT, trace=trace
        )
    row = {
        "outcome": outcome,
        "score": score,
        "actions": cave.steps,
        "distinct_cells": len(cave.reached),
        "picked_gold": int(cave.has_gold),
        "immediate_retreat": int(outcome == "retreat" and cave.steps == 1),
        "stalled_timeout": int(
            outcome == "timeout" and cave.steps - cave.last_new >= STALL_LENGTH
        ),
    }
    report = {"steps": steps, "unmatched_messages": unmatched} if trace else None
    if report is not None and policy != NOTEBOOK:
        # Inspect the exact auction at the entrance after collecting metrics.
        sensors = cave.reset()
        model.reset_inputs(sensors)
        messages = sensors | model.memory.messages()
        bids = {}
        for rule in model.applicable(messages):
            bids[rule.effector] = max(
                rule.strength, bids.get(rule.effector, -float("inf"))
            )
        report["entrance_messages"] = sorted(messages)
        report["entrance_best_bid_by_action"] = {
            action: round(bid, 3) for action, bid in bids.items()
        }
    return row, report


def summary(rows):
    outcomes = Counter(row["outcome"] for row in rows)
    return {
        "caves": len(rows),
        "mean_score": mean(row["score"] for row in rows),
        "mean_distinct_cells": mean(row["distinct_cells"] for row in rows),
        **{name: outcomes[name] for name in OUTCOMES},
        **{
            name: sum(row[name] for row in rows)
            for name in ("picked_gold", "immediate_retreat", "stalled_timeout")
        },
    }


def final_test(task):
    policy, seed, snapshot, budget = task
    model, _ = restore(policy, seed, snapshot)
    meta = {"policy": policy, "seed": seed, "episodes": budget}
    rows = [
        {
            **meta,
            "cave_id": i,
            "gold_distance": gold_distance(layout),
            **evaluate(policy, model, layout)[0],
        }
        for i, layout in enumerate(MAPS.test, 1)
    ]
    return rows, {**meta, **summary(rows)}


def diagnostic_traces(rows, seed, states, maps):
    choices = {}
    fresh = {(r["policy"], r["cave_id"]): r for r in rows if r["seed"] == seed}
    for i, layout in enumerate(maps.test, 1):
        raw, curriculum, notebook = (fresh[p, i] for p in (RAW, CURRICULUM, NOTEBOOK))
        for label, condition in {
            "curriculum_returns_gold": curriculum["outcome"] == "gold",
            "raw_picks_gold_without_return": raw["picked_gold"]
            and raw["outcome"] != "gold",
            "raw_no_matching_rule": raw["outcome"] == "no_matching_action",
            "raw_stalls_notebook_succeeds": raw["stalled_timeout"]
            and notebook["outcome"] == "gold",
            "raw_retreats_notebook_succeeds": raw["immediate_retreat"]
            and curriculum["immediate_retreat"]
            and notebook["outcome"] == "gold",
        }.items():
            if condition:
                choices.setdefault(label, (i, layout))
    models = {
        policy: restore(policy, seed, states[policy, seed])[0] for policy in POLICIES
    }
    results = {}
    for label, (i, layout) in choices.items():
        policies = {}
        for policy, model in models.items():
            result, trace = evaluate(policy, model, layout, trace=True)
            policies[policy] = {**result, **trace}
        results[label] = {
            "cave_id": i,
            "seed": seed,
            "layout": layout,
            "policies": policies,
        }
    return results


def run(args):
    global NB
    NB = load_notebook(args.notebook)
    maps = make_maps(NB.random_layout, args.validation_size, args.test_size)
    budgets = args.checkpoints + ([args.extension] if args.extension else [])
    jobs = [(p, s) for p in POLICIES for s in args.seeds]
    states, curve, decisions = {}, [], []
    previous, current, last = [], [], 0
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=worker_init,
        initargs=(args.notebook, maps),
    ) as pool:
        for end in budgets:
            if end == args.extension:
                allowed = (
                    qualifying_extensions(previous, current, RAW_POLICIES)
                    if previous
                    else []
                )
                decisions.append(
                    {
                        "from": last,
                        "to": end,
                        "qualified": allowed,
                        "ran": bool(allowed),
                    }
                )
                if not allowed:
                    print(
                        f"Stopped at {last}: no raw-input LCS met the validation threshold.",
                        flush=True,
                    )
                    break
            tasks = [
                (p, s, states.get((p, s)), last, end, budgets[-1]) for p, s in jobs
            ]
            results = list(pool.map(train_chunk, tasks))
            states = {key: state for key, (state, _) in zip(jobs, results, strict=True)}
            previous, current = current, [row for _, row in results]
            curve.extend(current)
            last = end
            for policy in POLICIES:
                data = [row for row in current if row["policy"] == policy]
                print(
                    f"{end:>6} {policy:>21}  validation score={mean(r['mean_score'] for r in data):7.1f}  "
                    f"picked={mean(r['picked_gold'] / r['caves'] for r in data):5.1%}  "
                    f"gold={mean(r['gold'] / r['caves'] for r in data):5.1%}  "
                    f"rules={mean(r['rules'] for r in data):.0f}",
                    flush=True,
                )
        print(
            f"Training depth fixed at {last}; starting the held-out test.", flush=True
        )
        tested = list(
            pool.map(final_test, [(p, s, states[p, s], last) for p, s in jobs])
        )

    rows = [row for episode_rows, _ in tested for row in episode_rows]
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "validation.csv", curve)
    write_csv(out / "test-summary.csv", [summary_row for _, summary_row in tested])
    write_csv(out / "test-episodes.csv", rows)
    write_csv(
        out / "layouts.csv",
        [
            {"split": split, "cave_id": i, "layout": json.dumps(layout)}
            for split in ("validation", "test")
            for i, layout in enumerate(getattr(maps, split), 1)
        ],
    )
    write_json(
        out / "traces.json", diagnostic_traces(rows, args.seeds[0], states, maps)
    )
    write_json(
        out / "config.json",
        {
            "lcs_source_revision": UPSTREAM_REVISION,
            "notebook_revision": NOTEBOOK_REVISION,
            "python": sys.version.split()[0],
            "seeds": args.seeds,
            "training_layout_seed_base": TRAIN_SEED_BASE,
            "validation_seed": VALIDATION_SEED,
            "test_seed": TEST_SEED,
            "validation_caves": args.validation_size,
            "test_caves": args.test_size,
            "training_cap": MAX_STEPS,
            "test_cap": TEST_ACTION_LIMIT,
            "checkpoints": args.checkpoints,
            "extension": decisions,
            "final_budget": last,
            "curriculum": {
                "seed_base": CURRICULUM_SEED,
                "adjacent_gold_episodes": EASY_END,
                "short_path_end": NEAR_END,
                "then": "same uniform layouts as controls",
            },
            "extension_rule": "Gold rate +2 pp and positive score change in aggregate, both gains in a majority of seeds.",
            "stalled_timeout": f"No new square reached in the last {STALL_LENGTH} actions of a timeout.",
            "ga": False,
            "lcs_hyperparameters": {"beta": 0.5, "gamma": 0.9, "epsilon": 0.2},
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("results/primitive-navigation")
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=list(CHECKPOINTS))
    parser.add_argument(
        "--extension",
        type=int,
        default=25000,
        help="optional validation-gated checkpoint; 0 disables",
    )
    parser.add_argument("--validation-size", type=int, default=300)
    parser.add_argument("--test-size", type=int, default=500)
    args = parser.parse_args()
    budgets = args.checkpoints + ([args.extension] if args.extension else [])
    if budgets[0] < 1 or sorted(set(budgets)) != budgets:
        parser.error("budgets must be positive and strictly increasing")
    if (
        len(set(args.seeds)) != len(args.seeds)
        or min(args.validation_size, args.test_size, args.workers) < 1
    ):
        parser.error(
            "seeds must be distinct and split sizes and workers must be positive"
        )
    run(args)


if __name__ == "__main__":
    main()
