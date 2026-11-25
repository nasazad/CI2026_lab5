import unittest

from paired_wumpus import paired_intervals


def rows():
    for seed in range(3):
        for cave in range(4):
            for policy, outcome, shift in (("a", "gold", 3), ("b", "death", 0)):
                yield {
                    "policy": policy,
                    "seed": seed,
                    "cave_id": cave,
                    "outcome": outcome,
                    "score": 10 * seed + cave + shift,
                }


class PairedIntervalsTests(unittest.TestCase):
    def test_constant_delta_gives_point_interval(self):
        result = paired_intervals(rows(), "a", "b", n=30)
        self.assertEqual(result["gold_rate"], {"delta": 1, "lo": 1, "hi": 1})
        self.assertEqual(result["mean_score"], {"delta": 3, "lo": 3, "hi": 3})

    def test_requires_complete_identical_grids(self):
        data = list(rows())
        for invalid in (data[:-1], data[:-2], [*data, data[0]]):
            with self.assertRaises(ValueError):
                paired_intervals(invalid, "a", "b", n=30)


if __name__ == "__main__":
    unittest.main()
