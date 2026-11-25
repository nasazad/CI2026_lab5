import unittest

from benchmark_transfer import (
    FIXED_LAYOUT,
    NOTEBOOK_SHA256,
    NotebookCaveAdapter,
    load_notebook,
    make_layouts,
    run_fixed_policy,
)
from classifiers import HAND_WRITTEN, rule


class TransferTests(unittest.TestCase):
    def test_layouts_exclude_training_lab_map_and_duplicates(self):
        candidates = [
            FIXED_LAYOUT,
            ((2, 1), (2, 2), ((3, 1), (3, 3), (4, 4))),
            ((2, 2), (3, 2), ((3, 1), (3, 3), (4, 4))),
            ((2, 3), (2, 4), ((3, 1), (3, 3), (4, 4))),
        ]
        scheduled = iter(candidates[:2] + candidates[:3] + candidates[2:])
        train, test = make_layouts(
            lambda _: next(scheduled), train_count=2, test_count=2
        )
        self.assertEqual(train, candidates[:2])
        self.assertEqual(test, candidates[2:])

    def test_notebook_pin_rejects_other_code(self):
        class DifferentNotebook:
            def read_bytes(self):
                return b'{"cells":[]}'

        self.assertEqual(len(NOTEBOOK_SHA256), 64)
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            load_notebook(DifferentNotebook())

    def test_adapter_returns_notebook_signals_and_rewards(self):
        cave = NotebookCaveAdapter((2, 1), (3, 1), ())
        self.assertEqual(cave.perceive(), {"at_start"})
        self.assertEqual(cave.step("shoot"), ({"at_start", "scream"}, -11, None))
        self.assertEqual(cave.step("fwd"), ({"glitter"}, -1, None))
        for action in ("grab", "turn_l", "turn_l", "fwd"):
            cave.step(action)
        self.assertEqual(cave.step("climb"), ({"at_start"}, 999, "gold"))
        self.assertEqual(sum(cave.rewards), 983)

    def test_unmatched_messages_and_timeout_are_distinct(self):
        layout = ((2, 1), (4, 4), ())
        rules = [rule("at_start", "fwd")]
        short = run_fixed_policy(rules, layout, dead_reckoning=True, limit=1)
        stopped = run_fixed_policy(rules, layout, dead_reckoning=True, limit=2)
        self.assertEqual(
            (short.outcome, stopped.outcome), ("timeout", "no_matching_action")
        )
        self.assertEqual(stopped.score, -1)
        self.assertEqual(
            set(stopped.unmatched_messages),
            {"glitter", "facing_east", "no_gold", "x2", "y1"},
        )

    def test_terminal_action_at_limit_is_not_a_timeout(self):
        result = run_fixed_policy([rule("", "climb")], FIXED_LAYOUT, limit=1)
        self.assertEqual((result.outcome, result.score), ("retreat", -1))

    def test_handwritten_reproduces_lab_route(self):
        result = run_fixed_policy(HAND_WRITTEN, FIXED_LAYOUT)
        self.assertEqual(
            (result.outcome, result.score, len(result.trace)), ("gold", 988, 12)
        )


if __name__ == "__main__":
    unittest.main()
