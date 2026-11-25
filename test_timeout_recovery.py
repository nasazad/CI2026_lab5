import os
import unittest
from pathlib import Path

import benchmark_rule_search as benchmark
from analyze_timeout_recovery import score_parts
from benchmark_navigation import MeasuredCave
from benchmark_transfer import load_notebook
from learned_recovery import RecoveryPolicy, RecoverySettings, observed_state
from primitive_memory import RawPerceptMemory
from recovery_router import RecoveryRouter, collect_examples
from shared_rule_lcs import ACTION_IDS, SENSOR_BITS, Settings, SharedRuleLCS
from timeout_diagnostics import audit_timeout, first_repeat


LAYOUT = ((4, 4), (4, 3), ((3, 3),))


def fixture(command):
    base = SharedRuleLCS(0, Settings(wider_memory=True))
    scores = [0.0] * 6
    scores[ACTION_IDS[command]] = 1.0
    base.weights[("landmarks", False, SENSOR_BITS["at_start"])] = scores
    return base


class TimeoutRecoveryTests(unittest.TestCase):
    def test_exact_repeat_not_just_repeated_room(self):
        memory = RawPerceptMemory()
        sensors = frozenset({"at_start"})
        memory.prime(sensors)
        memory.reset()
        a = observed_state(memory, sensors, False, False)
        b = observed_state(memory, sensors, True, False)
        c = observed_state(memory, sensors, True, True)
        self.assertEqual(first_repeat([a, b, c, b]), (1, 3))
        self.assertIsNone(first_repeat([a, b, c]))

    def test_certifies_rotation_cycle_from_complete_state(self):
        result = audit_timeout(fixture("turn_l"), LAYOUT)
        self.assertEqual((result["kind"], result["period"]), ("turn_in_place", 4))
        self.assertEqual(result["cap_240_outcome"], "timeout")

    def test_non_cycle_exactly_preserves_command_and_score(self):
        base = fixture("climb")
        policy = RecoveryPolicy(base, 17)
        cave = MeasuredCave(gold=LAYOUT[0], wumpus=LAYOUT[1], pits=LAYOUT[2])
        self.assertEqual(policy.evaluate(cave, 120), ("retreat", -1, 1, 0))
        self.assertEqual(policy.train_episode(cave, 120), ("retreat", -1, 1, 0))
        self.assertFalse(policy.weights)

    def test_learnt_scores_can_exit_after_cycle_without_mutating_base(self):
        base = fixture("turn_l")
        model = RecoveryPolicy(base, 17, RecoverySettings(base_strength=0))
        key = ("landmarks", False, SENSOR_BITS["at_start"])
        before = base.weights[key][:]
        model.weights[key] = [0, 0, 0, 0, 0, 1]
        cave = MeasuredCave(gold=LAYOUT[0], wumpus=LAYOUT[1], pits=LAYOUT[2])
        self.assertEqual(model.evaluate(cave, 120), ("retreat", -6, 6, 6))
        self.assertEqual(base.weights[key], before)
        self.assertEqual(model.weights[key], [0, 0, 0, 0, 0, 1])
        self.assertEqual(model.episode_index, 0)

    def test_risk_and_score_updates_are_separate(self):
        base = fixture("climb")
        model = RecoveryPolicy(base, 17, RecoverySettings(alpha=0.5))
        rule, risk = [0.0] * 6, [0.0] * 6
        model.back_up([([rule], [risk], 2, 0, -1.001, -1.001)], death=True)
        self.assertAlmostEqual(rule[2], -0.5005)
        self.assertEqual(risk[2], 0.5)
        model.back_up([([rule], [risk], 2, 0, -1.001, -1.001)], imitate=True)
        self.assertAlmostEqual(rule[2], -0.5005)
        self.assertEqual(risk[2], 0.5)

    def test_score_parts_recover_arrow_cost(self):
        self.assertEqual(
            score_parts({"outcome": "gold", "actions": "24", "score": "966"}),
            (1000, 0, -24, -10),
        )

    def test_routing_uses_complete_trials_from_the_entrance(self):
        base = fixture("turn_l")
        cautious = RecoveryPolicy(base, 0, RecoverySettings(base_strength=0))
        exploratory = RecoveryPolicy(base, 1, RecoverySettings(base_strength=0))
        key = ("landmarks", False, SENSOR_BITS["at_start"])
        cautious.weights[key] = [0, 0, 0, 0, 0, 1]
        exploratory.weights[key] = [0, 1, 0, 0, 0, 0]
        examples, steps = collect_examples(cautious, exploratory, [LAYOUT], 15, 0)
        self.assertEqual(steps, 6 + 15)
        _, gain, risk = examples[0]
        self.assertEqual((gain, risk), (-0.009, 0))
        router = RecoveryRouter(cautious, exploratory)
        router.fit(examples, 0)
        cave = MeasuredCave(gold=LAYOUT[0], wumpus=LAYOUT[1], pits=LAYOUT[2])
        self.assertEqual(router.evaluate(cave, 15), ("retreat", -6, 6, 6))
        self.assertFalse(router.last_exploratory)

    def test_routed_specialist_sees_the_same_pretrigger_attempts(self):
        base = fixture("turn_l")
        cautious = RecoveryPolicy(base, 0, RecoverySettings(base_strength=0))
        exploratory = RecoveryPolicy(
            base, 1, RecoverySettings(base_strength=0, extra_history=True)
        )
        key = ("landmarks", False, SENSOR_BITS["at_start"])
        exploratory.weights[key] = [0, 1, 0, 0, 0, 0.5]
        # Turning left at the repeated state has already been tried once.
        exploratory.weights["attempts", False, frozenset({"at_start"}), 1] = [
            0,
            -1,
            0,
            0,
            0,
            0,
        ]
        cave = MeasuredCave(gold=LAYOUT[0], wumpus=LAYOUT[1], pits=LAYOUT[2])
        stand_alone, through_router = [], []
        expected = exploratory.evaluate(cave, 15, trace=stand_alone)
        actual = cautious.evaluate(
            cave, 15, select=lambda *_: exploratory, trace=through_router
        )
        self.assertEqual(expected, ("retreat", -6, 6, 6))
        self.assertEqual(actual, expected)
        self.assertEqual(stand_alone, through_router)

    def test_inventory_is_observed_and_smoothed_target_is_optional(self):
        agent = SharedRuleLCS(0, Settings(inventory_memory=True, target_mix=0.1))
        cave = MeasuredCave(gold=LAYOUT[0], wumpus=LAYOUT[1], pits=LAYOUT[2])
        sensors = cave.reset()
        agent.memory.prime(sensors)
        agent.memory.reset()
        self.assertFalse(agent.memory.spent)
        after, _, _ = cave.step("shoot")
        agent.memory.update("shoot", sensors, after)
        self.assertTrue(agent.memory.spent)
        self.assertFalse(agent.memory.heard)
        rules = [[2, 0, 0, 0, 0, 0]]
        self.assertAlmostEqual(agent.target_value(rules), 1.8 + 0.2 / 6)
        self.assertEqual(fixture("climb").target_value(rules), 2)

    def test_timeout_holdouts_are_disjoint_from_published_progress_split(self):
        path = os.environ.get("WUMPUS_BASELINE_NOTEBOOK")
        nb = load_notebook(Path(path) if path else None)
        _, _, old = benchmark.evaluation_maps(nb, "progress")
        val, test, all_reserved = benchmark.evaluation_maps(nb, "timeout")
        self.assertEqual((len(val), len(test)), (600, 2000))
        self.assertFalse(set(val) & set(test))
        self.assertFalse((set(val) | set(test)) & old)
        self.assertTrue((set(val) | set(test) | old) <= all_reserved)


if __name__ == "__main__":
    unittest.main()
