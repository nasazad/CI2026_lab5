import os
import unittest
from pathlib import Path
from random import Random

import benchmark_randomized as benchmark
from benchmark_transfer import load_notebook, make_cave, run_fixed_policy
from cave import Cave
from learning import DeadReckoning, LearningCS
from randomized_lcs import IndexedLCS, RelativeInputs


def rule_signature(rules):
    return [(rule.condition, rule.action, rule.strength, rule.stamp) for rule in rules]


class RandomizedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        local = os.environ.get("WUMPUS_BASELINE_NOTEBOOK")
        cls.path = Path(local) if local else None
        cls.nb = load_notebook(cls.path)

    def test_reserved_splits_and_extension_prefix(self):
        split = benchmark.make_split(self.nb.random_layout, 30, 50)
        self.assertEqual(len(set(split.issue5 + split.validation + split.test)), 580)
        short = benchmark.training_layouts(self.nb.random_layout, 0, 10, split.reserved)
        long = benchmark.training_layouts(
            self.nb.random_layout, 0, 20000, split.reserved
        )
        self.assertEqual(short, long[:10])
        self.assertFalse(set(long) & split.reserved)

    def test_fast_matcher_preserves_learning_and_greedy_rollouts(self):
        regular = LearningCS(memory=DeadReckoning(), seed=9)
        fast = IndexedLCS(DeadReckoning(), seed=9, arity=4)
        rng = Random(28)
        layouts = [self.nb.random_layout(rng) for _ in range(60)]
        for layout in layouts:
            self.assertEqual(
                regular.episode(make_cave(layout)), fast.episode(make_cave(layout))
            )
        self.assertEqual(regular.rng.getstate(), fast.rng.getstate())
        self.assertEqual(regular.t, fast.t)
        signature = rule_signature(fast.rules)
        self.assertEqual(rule_signature(regular.rules), signature)
        for layout in layouts:
            slow = run_fixed_policy(regular.rules, layout, dead_reckoning=True)
            result = fast.trace_rollout(make_cave(layout), 120)
            self.assertEqual((slow.outcome, slow.score), (result.outcome, result.score))
            self.assertEqual(
                [r["action"] for r in slow.trace], [r["action"] for r in result.trace]
            )
        self.assertEqual(rule_signature(fast.rules), signature)

    def test_relative_states_share_rules_across_coordinates(self):
        cave = Cave(gold=(4, 4), wumpus=(4, 3), pits=())
        memory = RelativeInputs(self.nb)
        before = cave.reset()
        memory.prime(before)
        memory.reset()
        for _ in range(2):
            after, _, _ = cave.step("fwd")
            memory.update("fwd", before, after)
            before = after
            self.assertEqual(
                memory.messages(), {"no_gold", "ahead_safe", "home_behind"}
            )

    def test_initial_percepts_and_scream_change_beliefs(self):
        cave = Cave(gold=(3, 3), wumpus=(2, 1), pits=())
        memory = RelativeInputs(self.nb)
        before = cave.reset()
        memory.prime(before)
        memory.reset()
        self.assertIn("ahead_uncertain", memory.messages())
        after, _, _ = cave.step("shoot")
        memory.update("shoot", before, after)
        self.assertIn("ahead_safe", memory.messages())

    def test_grab_does_not_invent_gold(self):
        cave = Cave(gold=(2, 1), wumpus=(4, 4), pits=())
        memory = RelativeInputs(self.nb)
        before = cave.reset()
        memory.prime(before)
        memory.reset()
        for action, flag in (
            ("grab", "no_gold"),
            ("fwd", "no_gold"),
            ("grab", "has_gold"),
        ):
            after, _, _ = cave.step(action)
            memory.update(action, before, after)
            before = after
            self.assertIn(flag, memory.messages())

    def test_split_training_equals_uninterrupted_training(self):
        split = benchmark.make_split(self.nb.random_layout, 10, 20)
        benchmark.init_worker(self.path, split)
        benchmark.training_stream.cache_clear()
        for policy in benchmark.POLICIES:
            with self.subTest(policy=policy):
                _, uninterrupted = benchmark.train_model(policy, 0, None, 0, 35, 35)
                _, middle = benchmark.train_model(policy, 0, None, 0, 13, 35)
                _, resumed = benchmark.train_model(policy, 0, middle, 13, 35, 35)
                self.assertEqual(
                    (resumed.rng, resumed.actions),
                    (uninterrupted.rng, uninterrupted.actions),
                )
                signature = list if policy == "notebook_baseline" else rule_signature
                self.assertEqual(
                    signature(resumed.model), signature(uninterrupted.model)
                )

    def test_deeper_training_requires_gold_and_score_gains(self):
        before = [
            {"policy": policy, "seed": seed, "caves": 100, "gold": 10, "mean_score": 0}
            for policy in benchmark.POLICIES[:2]
            for seed in range(3)
        ]
        after = [
            {
                **row,
                "gold": 12,
                "mean_score": 10 if row["policy"] == "relative_lcs" else -10,
            }
            for row in before
        ]
        self.assertEqual(
            benchmark.qualifying_extensions(before, after), ["relative_lcs"]
        )


if __name__ == "__main__":
    unittest.main()
