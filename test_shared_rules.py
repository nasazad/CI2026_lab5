import unittest

from cave import Cave, DEATH, EFFECTORS, STEP_COST
from primitive_memory import RawPerceptMemory
from shared_rule_lcs import Settings, SharedRuleLCS, condition_keys, reward_for_learning


class SharedRuleTests(unittest.TestCase):
    def test_conditions_reuse_rules_across_memory_records(self):
        memory = RawPerceptMemory()
        sensors = frozenset({"breeze", "at_start"})
        memory.prime(sensors)
        memory.reset()
        first = set(condition_keys(memory, sensors))
        memory.last_action = "turn_r"
        next_keys = set(condition_keys(memory, sensors))
        self.assertFalse(first == next_keys)
        self.assertIn(("sensors", False, 33), first & next_keys)
        self.assertFalse(any(key[0] == "exact" for key in first & next_keys))

    def test_covering_includes_every_command_and_evaluation_is_frozen(self):
        cave = Cave()
        agent = SharedRuleLCS(0, Settings())
        sensors = cave.reset()
        agent.memory.prime(sensors)
        agent.memory.reset()
        rules = agent.matching_rules(sensors, create=True)
        self.assertTrue(all(len(weights) == len(EFFECTORS) for weights in rules))
        rules[0][EFFECTORS.index("climb")] = 1.0
        snapshot = {key: tuple(values) for key, values in agent.weights.items()}
        outcome, score, steps, trace = agent.evaluate(cave, 120, trace=True)
        self.assertEqual((outcome, score, steps), ("retreat", -1, 1))
        self.assertEqual(set(trace[0]["action_values"]), set(EFFECTORS))
        self.assertEqual(
            snapshot, {key: tuple(values) for key, values in agent.weights.items()}
        )

    def test_early_penalty_and_novelty_do_not_override_actual_scoring(self):
        settings = Settings(penalty=25, novel_reward=20)
        self.assertEqual(reward_for_learning(-1, "retreat", 1, 1, settings), -26)
        self.assertEqual(reward_for_learning(-1, "retreat", 2, 2, settings), -1)
        self.assertEqual(reward_for_learning(-1, None, 1, 2, settings), 19)
        self.assertEqual(reward_for_learning(-1001, "death", 1, 2, settings), -1001)
        self.assertEqual(reward_for_learning(999, "gold", 4, 4, settings), 999)

    def test_n_step_target_is_truncated_and_bootstraps(self):
        agent = SharedRuleLCS(0, Settings(n=2, gamma=0.5, alpha=0.5))
        states = [[0.0] * len(EFFECTORS) for _ in range(3)]
        agent.back_up(
            [([weights], 0, reward) for weights, reward in zip(states, (1, 2, 4))]
        )
        self.assertEqual([weights[0] for weights in states], [1.25, 2, 2])
        self.assertTrue(all(not any(weights[1:]) for weights in states))

    def test_death_sized_retreat_penalty_cannot_be_waited_out(self):
        settings = Settings(penalty=-DEATH)
        self.assertEqual(
            reward_for_learning(STEP_COST, "retreat", 1, 1, settings),
            STEP_COST + DEATH,
        )
        self.assertEqual(
            reward_for_learning(STEP_COST, "retreat", 2, 2, settings), STEP_COST
        )

    def test_wider_memory_only_emits_records_seen_by_agent(self):
        memory = RawPerceptMemory()
        sensors = frozenset({"at_start"})
        memory.prime(sensors)
        memory.reset()
        self.assertFalse(
            any(
                key[0] == "record"
                for key in condition_keys(memory, sensors, wider_memory=True)
            )
        )
        memory.observations[3, 1] = "breeze"
        records = [
            key
            for key in condition_keys(memory, sensors, wider_memory=True)
            if key[0] == "record"
        ]
        self.assertEqual(records, [("record", False, 32, 2, 0, "breeze")])


if __name__ == "__main__":
    unittest.main()
