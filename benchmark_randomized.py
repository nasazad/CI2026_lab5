"""Paired learning curves and held-out evaluation for randomized-cave LCS."""

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

from benchmark_transfer import (
    FIXED_EPISODES,
    FIXED_LAYOUT,
    NOTEBOOK_REVISION,
    OUTCOMES,
    TEST_ACTION_LIMIT,
    UPSTREAM_REVISION,
    load_notebook,
    make_cave,
    make_layouts,
    run_notebook_policy,
    write_csv,
    write_json,
)
from classifiers import MAX_STEPS
from learning import DeadReckoning
from randomized_lcs import IndexedLCS, RelativeInputs


VALIDATION_SEED, FINAL_TEST_SEED, TRAIN_SEED_BASE = 101, 2026, 7000
POLICIES = ("coordinate_lcs", "relative_lcs", "notebook_baseline")
CHECKPOINTS, EXTENSIONS = (1600, 5000, 10000), (25000, 50000)
MIN_GOLD_GAIN, TIMEOUT_TRIGGER = 0.02, 0.10


@dataclass
class Split:
    issue5: tuple
    validation: tuple
    test: tuple
    reserved: frozenset


def distinct_layouts(random_layout, n, rng, reserved):
    result = []
    while len(result) < n:
        layout = random_layout(rng)
        if layout not in reserved:
            reserved.add(layout)
            result.append(layout)
    return tuple(result)


def make_split(random_layout, validation_size, test_size):
    issue5_train, issue5 = make_layouts(random_layout)
    reserved = set(issue5_train) | set(issue5) | {FIXED_LAYOUT}
    validation = distinct_layouts(
        random_layout, validation_size, Random(VALIDATION_SEED), reserved
    )
    test = distinct_layouts(random_layout, test_size, Random(FINAL_TEST_SEED), reserved)
    return Split(tuple(issue5), validation, test, frozenset(reserved))


def training_layouts(random_layout, seed, count, reserved):
    rng = Random(TRAIN_SEED_BASE + seed)
    layouts = []
    while len(layouts) < count:
        layout = random_layout(rng)
        if layout not in reserved:
            layouts.append(layout)
    return tuple(layouts)


@dataclass
class State:
    model: object
    rng: object
    actions: int
    seconds: float


def init_worker(notebook_path, split):
    global NB, SPLIT
    NB = load_notebook(notebook_path)
    SPLIT = split


@lru_cache(maxsize=12)
def training_stream(seed, count):
    return training_layouts(NB.random_layout, seed, count, SPLIT.reserved)


def classifier_model(policy, seed, rules=()):
    if policy == "relative_lcs":
        return IndexedLCS(RelativeInputs(NB), seed, 3, rules)
    return IndexedLCS(DeadReckoning(), seed, 4, rules)


def baseline_rules():
    return (NB.RETREAT, NB.RISK_EARLY, NB.RISK_LATE)


def baseline_model(state=None):
    values = NB.RuleValues()
    if state is not None:
        for rule, (value, count) in zip(baseline_rules(), state.model, strict=True):
            values.mean[rule], values.count[rule] = value, count
    return values


def train_model(policy, seed, previous, start, stop, stream_size):
    is_baseline = policy == "notebook_baseline"
    if is_baseline:
        model, rng = baseline_model(previous), Random(seed)
    else:
        model = classifier_model(
            policy, seed, () if previous is None else previous.model
        )
        rng = model.rng
    actions = seconds = 0
    if previous is not None:
        actions, seconds = previous.actions, previous.seconds
        rng.setstate(previous.rng)
        if not is_baseline:
            model.t = actions

    layouts = training_stream(seed, stream_size)[start:stop]
    t0 = perf_counter()
    for layout in layouts:
        if is_baseline:
            _, _, trace = NB.play(
                layout, model, rng, epsilon=0.4, learn=True, max_steps=MAX_STEPS
            )
            actions += len(trace)
        else:
            model.episode(make_cave(layout))
    if not is_baseline:
        actions = model.t
    elapsed = seconds + perf_counter() - t0
    state = (
        [(model.mean[rule], model.count[rule]) for rule in baseline_rules()]
        if is_baseline
        else model.rules
    )
    return model, State(state, rng.getstate(), actions, elapsed)


def evaluate(policy, model, layouts, limit=TEST_ACTION_LIMIT):
    rows = []
    for layout in layouts:
        if policy == "notebook_baseline":
            result = run_notebook_policy(NB, model, layout, limit=limit)
            outcome, score, actions = result.outcome, result.score, len(result.trace)
        else:
            outcome, score, actions, _, _ = model.greedy_episode(
                make_cave(layout), limit
            )
        rows.append({"outcome": outcome, "score": score, "actions": actions})
    return rows


def summary(rows):
    counts = Counter(row["outcome"] for row in rows)
    return {
        "caves": len(rows),
        "mean_score": mean(row["score"] for row in rows),
        **{outcome: counts[outcome] for outcome in OUTCOMES},
    }


def training_chunk(task):
    policy, seed, state, start, stop, stream_size = task
    model, updated = train_model(policy, seed, state, start, stop, stream_size)
    return updated, {
        "policy": policy,
        "seed": seed,
        "episodes": stop,
        "training_actions": updated.actions,
        "rules": len(updated.model),
        "train_seconds": round(updated.seconds, 3),
        **summary(evaluate(policy, model, SPLIT.validation)),
    }


def qualifying_extensions(previous, current):
    candidates = []
    for policy in POLICIES[:2]:
        old = {row["seed"]: row for row in previous if row["policy"] == policy}
        new = {row["seed"]: row for row in current if row["policy"] == policy}
        delta = [
            (
                (new[s]["gold"] - old[s]["gold"]) / new[s]["caves"],
                new[s]["mean_score"] - old[s]["mean_score"],
            )
            for s in sorted(new)
        ]
        if (
            mean(gold for gold, _ in delta) >= MIN_GOLD_GAIN
            and mean(score for _, score in delta) > 0
            and sum(gold > 0 and score > 0 for gold, score in delta) > len(delta) / 2
        ):
            candidates.append(policy)
    return candidates


def restore(policy, seed, state):
    if policy == "notebook_baseline":
        return baseline_model(state)
    return classifier_model(policy, seed, state.model)


def train_fixed_reference(seed):
    model = classifier_model("fixed_reference", seed)
    cave = make_cave(FIXED_LAYOUT)
    for _ in range(FIXED_EPISODES):
        model.episode(cave)
    assert model.greedy_episode(cave, TEST_ACTION_LIMIT)[:3] == ("gold", 988, 12)
    return model


def final_evaluation(task):
    policy, seed, state, stop, horizon_check = task
    if policy == "fixed_reference":
        model, episodes = train_fixed_reference(seed), FIXED_EPISODES
    else:
        model, episodes = restore(policy, seed, state), stop

    rows, results, horizons = [], [], []
    for name, layouts in (("issue5", SPLIT.issue5), ("fresh_test", SPLIT.test)):
        meta = {"policy": policy, "seed": seed, "episodes": episodes, "split": name}
        outcome = evaluate(policy, model, layouts)
        rows.extend({**meta, "cave_id": i, **row} for i, row in enumerate(outcome, 1))
        results.append({**meta, **summary(outcome)})
    if horizon_check:
        for limit in (120, 240):
            horizons.append(
                {
                    "policy": policy,
                    "seed": seed,
                    "episodes": episodes,
                    "cap": limit,
                    **summary(evaluate(policy, model, SPLIT.validation, limit)),
                }
            )
    return rows, results, horizons


def trace_examples(split, seed, states, rows):
    outcomes = {
        (row["policy"], row["cave_id"]): row["outcome"]
        for row in rows
        if row["split"] == "fresh_test" and row["seed"] == seed
    }
    choices = {}
    for i, layout in enumerate(split.test, 1):
        absolute, relative, baseline = (outcomes[policy, i] for policy in POLICIES)
        labels = {
            "relative_gold_coordinate_not_gold": relative == "gold"
            and absolute != "gold",
            "coordinate_gold_relative_not_gold": absolute == "gold"
            and relative != "gold",
            "baseline_gold_both_lcs_not_gold": baseline == "gold"
            and "gold" not in (absolute, relative),
            "relative_timeout": relative == "timeout",
        }
        for label, condition in labels.items():
            if condition:
                choices.setdefault(label, (i, layout))

    models = {
        policy: restore(policy, seed, states[policy, seed]) for policy in POLICIES
    }
    traces = {}
    for label, (i, layout) in choices.items():
        example = {"seed": seed, "cave_id": i, "layout": layout, "policies": {}}
        for policy, model in models.items():
            if policy == "notebook_baseline":
                result = run_notebook_policy(NB, model, layout)
            else:
                result = model.trace_rollout(make_cave(layout), TEST_ACTION_LIMIT)
            example["policies"][policy] = vars(result)
        traces[label] = example
    return traces


def show_checkpoint(rows):
    for policy in POLICIES:
        runs = [row for row in rows if row["policy"] == policy]
        print(
            f"{runs[0]['episodes']:>6} {policy:>18}  "
            f"validation score={mean(r['mean_score'] for r in runs):7.1f}  "
            f"gold={mean(r['gold'] / r['caves'] for r in runs):5.1%}  "
            f"timeout={mean(r['timeout'] / r['caves'] for r in runs):5.1%}  "
            f"rules={mean(r['rules'] for r in runs):.0f}",
            flush=True,
        )


def run(args):
    notebook = load_notebook(args.notebook)
    split = make_split(notebook.random_layout, args.validation_size, args.test_size)
    budgets = args.checkpoints + args.extensions
    model_keys = [(policy, seed) for policy in POLICIES for seed in args.seeds]
    states, curves, decisions = {}, [], []
    previous, current, last = [], [], 0
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_worker,
        initargs=(args.notebook, split),
    ) as pool:
        for target in budgets:
            if target in args.extensions:
                allowed = qualifying_extensions(previous, current) if previous else []
                decisions.append(
                    {
                        "target": target,
                        "from": last,
                        "run": bool(allowed),
                        "qualifying_policies": allowed,
                    }
                )
                if not allowed:
                    print(
                        f"Stopped at {last}: no LCS met the validation extension criterion.",
                        flush=True,
                    )
                    break
            jobs = [
                (p, s, states.get((p, s)), last, target, budgets[-1])
                for p, s in model_keys
            ]
            results = list(pool.map(training_chunk, jobs))
            previous, current = current, [row for _, row in results]
            states = {
                key: state for key, (state, _) in zip(model_keys, results, strict=True)
            }
            curves.extend(current)
            last = target
            show_checkpoint(current)

        overtime = {
            policy
            for policy in POLICIES
            if mean(
                row["timeout"] / row["caves"]
                for row in current
                if row["policy"] == policy
            )
            >= TIMEOUT_TRIGGER
        }
        print(
            f"Frozen evaluation at {last} episodes; horizon check: {sorted(overtime)}",
            flush=True,
        )
        jobs = [
            (p, s, states.get((p, s)), last, p in overtime)
            for p, s in model_keys + [("fixed_reference", s) for s in args.seeds]
        ]
        results = list(pool.map(final_evaluation, jobs))

    rows = [row for data, _, _ in results for row in data]
    final = [row for _, data, _ in results for row in data]
    horizon = [row for _, _, data in results for row in data]
    global NB
    NB = notebook
    traces = trace_examples(split, args.seeds[0], states, rows)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "validation-curve.csv", curves)
    write_csv(out / "test-summary.csv", final)
    write_csv(out / "test-episodes.csv", rows)
    if horizon:
        write_csv(out / "horizon-validation.csv", horizon)
    write_csv(
        out / "layouts.csv",
        [
            {"split": name, "cave_id": i, "layout": json.dumps(layout)}
            for name in ("issue5", "validation", "test")
            for i, layout in enumerate(getattr(split, name), 1)
        ],
    )
    write_json(out / "traces.json", traces)
    write_json(
        out / "config.json",
        {
            "lcs_source_revision": UPSTREAM_REVISION,
            "notebook_revision": NOTEBOOK_REVISION,
            "python": sys.version.split()[0],
            "seeds": args.seeds,
            "training_layout_seed_base": TRAIN_SEED_BASE,
            "training_action_limit": MAX_STEPS,
            "test_action_limit": TEST_ACTION_LIMIT,
            "validation_seed": VALIDATION_SEED,
            "test_seed": FINAL_TEST_SEED,
            "validation_caves": len(split.validation),
            "fresh_test_caves": len(split.test),
            "required_checkpoints": args.checkpoints,
            "possible_extensions": args.extensions,
            "extension_rule": "At least +2 pp mean validation gold and positive mean score gain, both positive in a majority of seeds for either LCS.",
            "extension_decisions": decisions,
            "final_training_episodes": last,
            "timeout_trigger": TIMEOUT_TRIGGER,
            "horizon_checked_on_validation": sorted(overtime),
            "ga_enabled": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebook", type=Path, help="exact pinned notebook, for offline use"
    )
    parser.add_argument("--output", type=Path, default=Path("results/issue-2"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=list(CHECKPOINTS))
    parser.add_argument("--extensions", type=int, nargs="*", default=list(EXTENSIONS))
    parser.add_argument("--validation-size", type=int, default=300)
    parser.add_argument("--test-size", type=int, default=500)
    args = parser.parse_args()
    budgets = args.checkpoints + args.extensions
    if budgets != sorted(set(budgets)) or budgets[0] < 1:
        parser.error("training budgets must be positive and strictly increasing")
    if (
        len(set(args.seeds)) != len(args.seeds)
        or min(args.validation_size, args.test_size, args.workers) < 1
    ):
        parser.error(
            "seeds must be distinct and split sizes and worker count must be positive"
        )
    run(args)


if __name__ == "__main__":
    main()
