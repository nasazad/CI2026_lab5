"""Train recovery rules from natural starts; evaluate on validation by default."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from random import Random
from time import perf_counter

import benchmark_rule_search as runner
from benchmark_navigation import gold_distance, measured_cave
from benchmark_transfer import (
    NOTEBOOK_REVISION,
    NOTEBOOK_SHA256,
    load_notebook,
    write_csv,
    write_json,
)
from learned_recovery import RecoveryPolicy, RecoverySettings
from recovery_router import RecoveryRouter, collect_examples
from shared_rule_lcs import SharedRuleLCS, average_rule_tables


PLAIN = RecoverySettings(history=False)
HISTORY = RecoverySettings()
CAUTIOUS = replace(HISTORY, death_cost=800, timeout_cost=125)
RISK = replace(
    HISTORY, risk_weight=0.65, timeout_cost=125, extra_history=True, replay=True
)
RECIPES = {
    "plain": PLAIN,
    "history": HISTORY,
    "cautious": CAUTIOUS,
    "early": replace(CAUTIOUS, early=12),
    "risk": RISK,
    "seek": replace(RISK, risk_weight=0.45, novelty=45.0),
    "reopen": replace(RISK, reopen=True),
    "discover": replace(RISK, reopen=True, risk_weight=0.30, novelty=8.0),
}
READOUTS = {
    "greedy": (0.0, 0.0),
    "soft15": (0.015, 0.0),
    "soft35": (0.035, 0.0),
    "soft65": (0.065, 0.0),
    "adaptive": (0.020, 0.018),
}
BASE_PROFILES = {
    "published": {},
    "inventory": {"inventory_memory": True},
    "smooth": {"target_mix": 0.10},
}


def evaluate(agent, layouts, seed, *, reference=None, exact=False):
    rows = []
    for cave_id, layout in enumerate(layouts, 1):
        cave = measured_cave(layout)
        if isinstance(agent, (RecoveryPolicy, RecoveryRouter)):
            outcome, score, steps, activation = agent.evaluate(
                cave, runner.EVAL_CAP, seed=seed * 100003 + cave_id
            )
        else:
            outcome, score, steps, _ = agent.evaluate(cave, runner.EVAL_CAP)
            activation = 0
        row = {
            "cave_id": cave_id,
            "outcome": outcome,
            "score": score,
            "actions": steps,
            "picked_gold": int(cave.has_gold),
            "distinct_cells": len(cave.reached),
            "immediate_retreat": int(outcome == "retreat" and steps == 1),
            "stalled_timeout": int(
                outcome == "timeout" and steps - cave.last_new >= 30
            ),
            "gold_distance": gold_distance(layout),
            "activated_at": activation,
            "exploratory_arm": int(getattr(agent, "last_exploratory", False)),
        }
        if reference is not None:
            baseline = reference[cave_id - 1]
            row["baseline_outcome"] = baseline["outcome"]
            if exact and baseline["outcome"] != "timeout" and baseline != row:
                raise AssertionError(
                    f"exact-repeat gate changed a non-timeout: seed {seed}, cave {cave_id}"
                )
        else:
            row["baseline_outcome"] = outcome
        rows.append(row)
    return rows


def pack(policy, seed, panel, meta):
    return (
        {
            "policy": policy,
            "seed": seed,
            **meta,
            **runner.summarize(panel),
            "activation": sum(bool(r["activated_at"]) for r in panel),
            "arm_activations": sum(r["exploratory_arm"] for r in panel),
            "recovered_gold": sum(
                r["baseline_outcome"] == "timeout" and r["outcome"] == "gold"
                for r in panel
            ),
            "recovered_retreat": sum(
                r["baseline_outcome"] == "timeout" and r["outcome"] == "retreat"
                for r in panel
            ),
            "recovered_death": sum(
                r["baseline_outcome"] == "timeout" and r["outcome"] == "death"
                for r in panel
            ),
        },
        [{"policy": policy, "seed": seed, **r} for r in panel],
    )


def example_trace(agent, base, layouts, rows, seed):
    candidates = [
        r for r in rows if r["baseline_outcome"] == "timeout" and r["outcome"] == "gold"
    ]
    if not candidates:
        return None
    example = max(candidates, key=lambda r: (r["score"], -r["cave_id"]))
    cave_id = example["cave_id"]
    layout = layouts[cave_id - 1]
    trace = []
    result = agent.evaluate(
        measured_cave(layout),
        runner.EVAL_CAP,
        seed=seed * 100003 + cave_id,
        trace=trace,
    )
    _, _, _, old_steps = base.evaluate(
        measured_cave(layout), runner.EVAL_CAP, trace=True
    )
    before = example["activated_at"] - 1
    if result != (example["outcome"], example["score"], example["actions"], before + 1):
        raise AssertionError("the recorded example did not reproduce")
    assert [s["command"] for s in trace[:before]] == [
        s["action"] for s in old_steps[:before]
    ]
    return {
        **example,
        "layout_for_reader_only": layout,
        "recovery_trace": trace,
        "baseline_commands": [s["action"] for s in old_steps],
    }


def worker(job):
    args, seed, validation, test, reserved = job
    runner.worker_init(args.notebook, validation, test, reserved)
    profile = runner.PROFILES["clean_ensemble3"]
    base_settings = replace(profile.settings, **BASE_PROFILES[args.base_profile])
    agents = [
        SharedRuleLCS(seed + 10003 * m, base_settings) for m in range(profile.members)
    ]
    base_actions = 0
    start = perf_counter()
    for layout in runner.training_stream(seed, args.base_episodes, profile.curriculum):
        for member in agents:
            _, steps = member.train_episode(measured_cave(layout), runner.TRAIN_CAP)
            base_actions += steps
    base = average_rule_tables(agents)
    panel = test if args.test else validation
    original = evaluate(base, panel, seed)
    empty = {
        "recovery_episodes": 0,
        "recovery_actions": 0,
        "training_activations": 0,
        "router_actions": 0,
        "router_examples": 0,
    }
    output = [pack("original", seed, original, {"base_actions": base_actions, **empty})]
    traces = {}
    print(
        f"seed {seed}: trained base ({base_actions:,} actions; {perf_counter() - start:.1f}s)",
        flush=True,
    )
    layouts = runner.training_stream(seed + 20009, args.episodes, False)
    heads = {}
    for name in args.profiles:
        settings = replace(RECIPES[name], episodes=args.episodes)
        model = RecoveryPolicy(base, seed + 78701, settings)
        actions = activations = 0
        for layout in layouts:
            _, _, steps, active = model.train_episode(
                measured_cave(layout), runner.EVAL_CAP
            )
            actions += steps
            activations += bool(active)
        meta = {
            **empty,
            "base_actions": base_actions,
            "recovery_episodes": args.episodes,
            "recovery_actions": actions,
            "training_activations": activations,
        }
        heads[name] = (model, settings, meta)
        for readout in args.readouts:
            if args.report and f"{name}_{readout}" not in args.report:
                continue
            temperature, reheat = READOUTS[readout]
            model.settings = replace(settings, temperature=temperature, reheat=reheat)
            rows = evaluate(
                model,
                panel,
                seed,
                reference=original,
                exact=not (settings.early or settings.reopen),
            )
            policy_name = f"{name}_{readout}"
            output.append(pack(policy_name, seed, rows, meta))
            if args.test:
                example = example_trace(model, base, panel, rows, seed)
                if example:
                    traces[policy_name] = example
        print(
            f"seed {seed}: {name} ({actions:,} extra actions; {activations:,} activations)",
            flush=True,
        )
    if args.router_episodes:
        safe, safe_settings, safe_meta = heads["cautious"]
        risk, risk_settings, risk_meta = heads[args.exploratory_arm]
        temperature, reheat = READOUTS["adaptive"]
        safe.settings = replace(safe_settings, temperature=temperature, reheat=reheat)
        risk.settings = replace(risk_settings, temperature=temperature, reheat=reheat)
        router = RecoveryRouter(safe, risk)
        examples, actions = collect_examples(
            safe,
            risk,
            runner.training_stream(seed + 30011, args.router_episodes, False),
            runner.EVAL_CAP,
            seed + 33341,
        )
        router.fit(examples, seed + 2413)
        meta = {
            "base_actions": base_actions,
            "recovery_episodes": safe_meta["recovery_episodes"]
            + risk_meta["recovery_episodes"],
            "recovery_actions": safe_meta["recovery_actions"]
            + risk_meta["recovery_actions"],
            "training_activations": safe_meta["training_activations"]
            + risk_meta["training_activations"],
            "router_actions": actions,
            "router_examples": len(examples),
        }
        for margin in (0, 30, 80):
            name = f"router_{margin:02d}"
            if args.report and name not in args.report:
                continue
            router.margin = margin / 1000
            rows = evaluate(router, panel, seed, reference=original, exact=True)
            output.append(pack(name, seed, rows, meta))
            if args.test:
                example = example_trace(router, base, panel, rows, seed)
                if example:
                    traces[name] = example
        print(
            f"seed {seed}: router ({actions:,} extra actions; {len(examples):,} paired examples)",
            flush=True,
        )
    if args.notebook_control:
        nb = runner.NB
        table, rng = nb.RuleValues(), Random(seed)
        steps = 0
        for layout in runner.training_stream(seed, args.base_episodes, False):
            _, _, trace = nb.play(
                layout, table, rng, epsilon=0.4, learn=True, max_steps=runner.TRAIN_CAP
            )
            steps += len(trace)
        notebook_test = [
            {
                "cave_id": i,
                **runner.rollout(table, runner.PROFILES["notebook"], layout)[0],
            }
            for i, layout in enumerate(panel, 1)
        ]
        rows = [
            {
                **row,
                "activated_at": 0,
                "exploratory_arm": 0,
                "baseline_outcome": original[i]["outcome"],
            }
            for i, row in enumerate(notebook_test)
        ]
        output.append(
            pack(
                "notebook",
                seed,
                rows,
                {"base_actions": steps, **empty},
            )
        )
    return output, traces


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--notebook", type=Path)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--profiles", nargs="+", choices=RECIPES, required=True)
    cli.add_argument(
        "--readouts",
        nargs="+",
        choices=READOUTS,
        default=["greedy", "soft15", "soft35"],
    )
    cli.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    cli.add_argument("--workers", type=int, default=5)
    cli.add_argument("--episodes", type=int, default=12000)
    cli.add_argument("--base-episodes", type=int, default=16000)
    cli.add_argument("--base-profile", choices=BASE_PROFILES, default="published")
    cli.add_argument("--router-episodes", type=int, default=0)
    cli.add_argument("--exploratory-arm", choices=("risk", "seek"), default="risk")
    cli.add_argument(
        "--report", nargs="+", help="report only these named candidates, plus controls"
    )
    cli.add_argument("--notebook-control", action="store_true")
    cli.add_argument(
        "--test",
        action="store_true",
        help="open the reserved final set after locking a recipe",
    )
    args = cli.parse_args()
    if args.test and not args.report:
        cli.error("lock the candidate names with --report before opening the test set")
    if args.episodes < 1 or args.base_episodes < 1 or args.workers < 1:
        cli.error("episodes, base episodes and workers must be positive")
    for values in (args.profiles, args.readouts, args.seeds):
        if len(set(values)) != len(values):
            cli.error("profiles, readouts and seeds must be unique")
    candidates = {
        f"{name}_{readout}" for name in args.profiles for readout in args.readouts
    }
    if args.router_episodes:
        if args.router_episodes < 0 or not {"cautious", args.exploratory_arm} <= set(
            args.profiles
        ):
            cli.error(
                "router needs positive episodes and both cautious and exploratory profiles"
            )
        candidates.update(("router_00", "router_30", "router_80"))
    if args.report and not set(args.report) <= candidates:
        cli.error("report must name candidates trained by this invocation")
    nb = load_notebook(args.notebook)
    validation, test, reserved = runner.evaluation_maps(nb, "timeout")
    jobs = [(args, seed, validation, test, reserved) for seed in args.seeds]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(worker, jobs))
    summary = [row for group, _ in results for row, _ in group]
    panel = [r for group, _ in results for _, rows in group for r in rows]
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output / "config.json",
        {
            **vars(args),
            "output": args.output.name
            if args.output.is_absolute()
            else str(args.output),
            "notebook": "Wumpus_World.ipynb",
            "notebook_revision": NOTEBOOK_REVISION,
            "notebook_sha256": NOTEBOOK_SHA256,
            "validation_seed": runner.TIMEOUT_VALIDATION_SEED,
            "test_seed": runner.TIMEOUT_TEST_SEED,
            "evaluation_cap": runner.EVAL_CAP,
            "recipes": {name: asdict(RECIPES[name]) for name in args.profiles},
            "base_settings": asdict(
                replace(
                    runner.PROFILES["clean_ensemble3"].settings,
                    **BASE_PROFILES[args.base_profile],
                )
            ),
            "readout_temperatures": {name: READOUTS[name] for name in args.readouts},
        },
    )
    write_csv(
        args.output / "layouts.csv",
        [
            {"cave_id": i, "layout": json.dumps(layout)}
            for i, layout in enumerate(test if args.test else validation, 1)
        ],
    )
    write_csv(args.output / "summary.csv", summary)
    write_csv(args.output / "episodes.csv", panel)
    if args.test:
        write_json(
            args.output / "traces.json",
            {
                f"{name}:{seed}": trace
                for seed, (_, traces) in zip(args.seeds, results, strict=True)
                for name, trace in traces.items()
            },
        )
    for policy in dict.fromkeys(r["policy"] for r in summary):
        rows = [r for r in summary if r["policy"] == policy]
        count = sum(r["caves"] for r in rows)
        rates = [
            sum(r[key] for r in rows) / count for key in ("gold", "death", "timeout")
        ]
        score = sum(r["mean_score"] for r in rows) / len(rows)
        print(
            f"{policy:>20} | gold {rates[0]:6.2%} | death {rates[1]:6.2%} | timeout {rates[2]:6.2%} | score {score:7.2f}"
        )


if __name__ == "__main__":
    main()
