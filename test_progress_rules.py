import unittest

from cave import Cave, EFFECTORS
from level_replay import LevelReplay
from shared_rule_lcs import (
    ProgressMemory,
    Settings,
    SharedRuleLCS,
    condition_keys,
    average_rule_tables,
    loop_free_indices,
    potential_delta,
)


class ProgressRuleTests(unittest.TestCase):
    def test_n_step_truncation_uses_final_observation_value(self):
        # Cross-checked independently with Claude Opus 5.5.
        agent = SharedRuleLCS(0, Settings(n=3, gamma=0.5, alpha=1))
        rows = [[0] * len(EFFECTORS) for _ in range(3)]
        agent.back_up([([row], 0, -1) for row in rows], bootstrap=8)
        self.assertEqual([row[0] for row in rows], [-0.75, 0.5, 3])

    def test_last_action_death_overrides_timeout_bootstrap(self):
        agent = SharedRuleLCS(0, Settings(bootstrap_truncation=True, epsilon=0))
        # Stub learned rule values, not the environment. Force fwd into the pit
        # on the last allowed action; there must be no continuation estimate.
        agent.matching_rules = lambda *_, **__: [[1, 0, 0, 0, 0, 0]]
        cave = Cave(gold=(4, 4), wumpus=(4, 3), pits=((2, 1),))
        outcome, score, steps, _, history, tail, _, _ = agent._play(cave, 1, train=True)
        self.assertEqual((outcome, score, steps, tail), ("death", -1001, 1, 0))
        self.assertAlmostEqual(history[-1][2], -1.001)

    def test_novelty_potential_stops_at_a_true_terminal(self):
        first = potential_delta(0.5, 1, 2, terminal=False)
        death = potential_delta(0.5, 2, 3, terminal=True)
        cap = potential_delta(0.5, 2, 3, terminal=False)
        self.assertEqual((first, death, cap), (0.5, -1, 0))
        self.assertEqual(first + 0.5 * death, 0)
        self.assertEqual(first + 0.5 * cap, 0.5)

    def test_passive_memory_can_distinguish_repeated_turns(self):
        cave = Cave()
        memory = ProgressMemory()
        initial = cave.reset()
        memory.prime(initial)
        memory.reset()
        for action in ("turn_r", "turn_l"):
            before = cave.sensors()
            after, _, _ = cave.step(action)
            memory.update(action, before, after)
        self.assertEqual(
            (memory.since_new, memory.since_move, memory.turns_since_move), (2, 2, 2)
        )
        self.assertEqual(set(memory.visits), {(1, 1)})
        self.assertTrue(
            any(key[0] == "progress" for key in condition_keys(memory, cave.sensors()))
        )

    def test_evaluation_ignores_and_preserves_exploration_counts(self):
        cave = Cave()
        agent = SharedRuleLCS(0, Settings(count_bonus=100, bootstrap_truncation=True))
        sensors = cave.reset()
        agent.memory.prime(sensors)
        agent.memory.reset()
        rules = agent.matching_rules(sensors, create=True)
        rules[0][EFFECTORS.index("climb")] = 1
        agent.action_counts[sensors, agent.memory.messages()] = [0, 0, 0, 0, 0, 30000]
        before = {k: tuple(v) for k, v in agent.action_counts.items()}
        outcome, score, steps, _ = agent.evaluate(cave, 120)
        self.assertEqual((outcome, score, steps), ("retreat", -1, 1))
        self.assertEqual(before, {k: tuple(v) for k, v in agent.action_counts.items()})
        self.assertEqual(agent.episode_index, 0)

    def test_loop_erasure_keeps_nonrepeated_state_and_last_exit(self):
        self.assertEqual(loop_free_indices(["a", "b", "c", "b", "d"]), [0, 3, 4])
        states = [
            ("start", "arrow ready"),
            ("start", "arrow spent"),
            ("gold", "arrow spent"),
        ]
        self.assertEqual(loop_free_indices(states), [0, 1, 2])

    def test_level_replay_is_confined_to_seen_training_keys(self):
        levels = LevelReplay(0, 1, warmup=2, capacity=2)
        levels.record("old_a", 0, 0.2)
        self.assertEqual(levels.choose("fresh", 1), "fresh")
        levels.record("old_b", 1, 0.3)
        levels.record("old_c", 2, 0.8)
        self.assertTrue(
            all(levels.choose("fresh", i) in {"old_b", "old_c"} for i in range(3, 20))
        )

    def test_averaging_uses_zero_for_conditions_missing_from_a_member(self):
        left, right = (SharedRuleLCS(seed, Settings()) for seed in (0, 1))
        left.weights["shared"] = [2] * len(EFFECTORS)
        right.weights["shared"] = [4] * len(EFFECTORS)
        right.weights["right_only"] = [6] * len(EFFECTORS)
        result = average_rule_tables([left, right])
        self.assertEqual(result.weights, {"shared": [3] * 6, "right_only": [3] * 6})


if __name__ == "__main__":
    unittest.main()
