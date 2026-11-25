"""Certify deterministic traps in the published, frozen primitive LCS."""

import argparse
import csv
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import benchmark_rule_search as runner
from benchmark_navigation import MeasuredCave, measured_cave
from benchmark_transfer import load_notebook, write_csv, write_json
from primitive_memory import RawPerceptMemory
from shared_rule_lcs import SharedRuleLCS, average_rule_tables


class AuditCave(MeasuredCave):
    """Privileged state is recorded offline; the policy gets only percepts."""

    def reset(self):
        self.before_action = []
        return super().reset()

    def step(self, action):
        self.before_action.append(
            (
                self.pos,
                self.heading,
                self.has_gold,
                self.arrow,
                self.wumpus_alive,
                self.bump,
                self.scream,
            )
        )
        return super().step(action)


def first_repeat(states):
    """Return zero-based (first occurrence, repeat) of the first exact cycle."""
    seen = {}
    for index, state in enumerate(states):
        if state in seen:
            return seen[state], index
        seen[state] = index
    return None


def audit_states(cave, trace):
    """Rebuild all mutable policy inputs; local action-score aliases are not cycles."""
    memory = RawPerceptMemory()
    percepts = [frozenset(step["percepts"]) for step in trace] + [cave.sensors()]
    memory.prime(percepts[0])
    memory.reset()
    for i, (env, step) in enumerate(zip(cave.before_action, trace, strict=True)):
        pose = memory.pose
        if (pose.x, pose.y) != tuple(step["square"]):
            raise ValueError("audit did not reconstruct the policy's own pose")
        if memory.messages() != frozenset(step["memory"]):
            raise ValueError("audit did not reconstruct the policy's memory")
        yield (
            env,
            pose.x,
            pose.y,
            pose.heading,
            pose.has_gold,
            memory.last_action,
            frozenset(memory.observations.items()),
        )
        memory.update(step["action"], percepts[i], percepts[i + 1])


def cycle_kind(env, actions):
    if len({s[0] for s in env}) > 1:
        return "moving_cycle"
    if len(set(actions)) > 1:
        return "stationary_mixed"
    return {
        "fwd": "wall_bump",
        "climb": "climb_away",
        "grab": "empty_grab",
        "shoot": "spent_arrow",
        "turn_l": "turn_in_place",
        "turn_r": "turn_in_place",
    }[actions[0]]


def audit_timeout(agent, layout):
    gold, wumpus, pits = layout
    cave = AuditCave(wumpus=wumpus, gold=gold, pits=pits)
    outcome, score, steps, trace = agent.evaluate(cave, runner.EVAL_CAP, trace=True)
    if outcome != "timeout":
        raise ValueError("recorded timeout could not be reproduced")
    cycle = first_repeat(audit_states(cave, trace))
    last_new = cave.last_new
    details = {
        "outcome": outcome,
        "score": score,
        "actions": steps,
        "picked_gold": int(cave.has_gold),
        "saw_glitter": int(any("glitter" in s["percepts"] for s in trace)),
        "distinct_cells": len(cave.reached),
        "last_new": last_new,
        "kind": "no_exact_repeat",
        "cycle_start_action": "",
        "period": "",
        "cycle_commands": "",
        "cap_240_outcome": outcome,
    }
    if cycle:
        first, repeat = cycle
        actions = [s["action"] for s in trace[first:repeat]]
        details.update(
            kind=cycle_kind(cave.before_action[first:repeat], actions),
            cycle_start_action=first + 1,
            period=repeat - first,
            cycle_commands=" ".join(actions),
        )
    else:
        details["cap_240_outcome"] = agent.evaluate(cave, 240)[0]
    return details


def audit_seed(job):
    notebook, seed, reference = job
    nb = load_notebook(notebook)
    val, test, reserved = runner.evaluation_maps(nb, "progress")
    runner.worker_init(notebook, val, test, reserved)
    profile = runner.PROFILES["clean_ensemble3"]
    agents = [
        SharedRuleLCS(seed + 10003 * m, profile.settings)
        for m in range(profile.members)
    ]
    for layout in runner.training_stream(seed, 16000, profile.curriculum):
        for agent in agents:
            agent.train_episode(measured_cave(layout), runner.TRAIN_CAP)
    agent = average_rule_tables(agents)
    audited = []
    for row in reference:
        cave_id = int(row["cave_id"])
        layout = test[cave_id - 1]
        result, _ = runner.rollout(agent, profile, layout)
        for k in ("outcome", "score", "actions", "picked_gold", "distinct_cells"):
            if str(result[k]) != row[k]:
                raise ValueError(f"historical seed {seed} cave {cave_id}: changed {k}")
        if row["outcome"] == "timeout":
            audited.append(
                {"seed": seed, "cave_id": cave_id, **audit_timeout(agent, layout)}
            )
    return audited


def summarize(rows):
    return {
        "timeouts": len(rows),
        "kinds": Counter(row["kind"] for row in rows),
        "periods": Counter(str(row["period"]) for row in rows if row["period"]),
        "common_cycles": Counter(
            row["cycle_commands"] for row in rows if row["period"]
        ).most_common(10),
        "before_pickup": sum(not row["picked_gold"] for row in rows),
        "after_pickup": sum(row["picked_gold"] for row in rows),
        "saw_glitter_without_pickup": sum(
            row["saw_glitter"] and not row["picked_gold"] for row in rows
        ),
        "cap_240_outcomes": Counter(row["cap_240_outcome"] for row in rows),
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--notebook", type=Path)
    cli.add_argument(
        "--reference",
        type=Path,
        default=Path("results/learned-exploration/final/test-episodes.csv"),
    )
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--workers", type=int, default=5)
    args = cli.parse_args()
    by_seed = {}
    with args.reference.open(newline="") as file:
        for row in csv.DictReader(file):
            if row["policy"] == "clean_ensemble3":
                by_seed.setdefault(int(row["seed"]), []).append(row)
    if len(by_seed) != 5 or any(len(rows) != 2000 for rows in by_seed.values()):
        raise ValueError("expected the complete historical 5 × 2,000 grid")
    jobs = [(args.notebook, seed, rows) for seed, rows in sorted(by_seed.items())]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(audit_seed, jobs))
    rows = [row for group in results for row in group]
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "timeouts.csv", rows)
    summary = summarize(rows)
    write_json(args.output / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
