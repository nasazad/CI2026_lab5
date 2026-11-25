import os
import unittest
from pathlib import Path

import benchmark_navigation as benchmark
from benchmark_transfer import load_notebook
from cave import Cave, EFFECTORS
from primitive_memory import RawPerceptMemory
from randomized_lcs import IndexedLCS


def observe(cave, memory, action):
    before = cave.sensors()
    after, _, _ = cave.step(action)
    memory.update(action, before, after)


class NavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        local = os.environ.get("WUMPUS_BASELINE_NOTEBOOK")
        cls.notebook = load_notebook(Path(local) if local else None)

    def test_unobserved_cell_is_not_an_inferred_wall_or_safe_square(self):
        memory = RawPerceptMemory()
        memory.prime(frozenset({"at_start", "breeze"}))
        memory.reset()
        self.assertEqual(
            memory.messages(),
            {
                "last_none",
                "no_gold",
                "remembered_front:unknown",
                "remembered_left:unknown",
                "remembered_back:unknown",
                "remembered_right:unknown",
            },
        )

    def test_memory_rotates_actual_observations_without_planning(self):
        cave = Cave(gold=(4, 3), wumpus=(4, 2), pits=((1, 2),))
        memory = RawPerceptMemory()
        memory.prime(cave.reset())
        memory.reset()
        observe(cave, memory, "fwd")
        self.assertIn("remembered_back:at_start+breeze", memory.messages())
        observe(cave, memory, "turn_r")
        self.assertIn("remembered_right:at_start+breeze", memory.messages())
        self.assertIn("last_turn_r", memory.messages())
        observe(cave, memory, "fwd")  # bump; pose and observation stay put
        self.assertEqual((memory.pose.x, memory.pose.y), (2, 1))
        self.assertIn("remembered_right:at_start+breeze", memory.messages())

    def test_primitive_covering_includes_all_actions(self):
        memory = RawPerceptMemory()
        cave = Cave()
        agent = IndexedLCS(memory, seed=0, arity=RawPerceptMemory.ARITY)
        initial = cave.reset()
        agent.reset_inputs(initial)
        rules = agent.match(initial | memory.messages())
        self.assertEqual(tuple(rule.effector for rule in rules), EFFECTORS)

    def test_holdouts_remain_reserved_and_curriculum_rejoins_the_schedule(self):
        nb = self.notebook
        maps = benchmark.make_maps(nb.random_layout, 30, 40)
        old = benchmark.make_split(nb.random_layout, 300, 500, climb_ablation=True)
        self.assertFalse((set(maps.validation) | set(maps.test)) & old.reserved)
        self.assertFalse(set(maps.validation) & set(maps.test))
        uniform = benchmark.training_layouts(nb.random_layout, 0, 3000, maps.reserved)
        curriculum = benchmark.curriculum_layouts(
            nb.random_layout, 0, uniform, maps.reserved
        )
        self.assertFalse((set(uniform) | set(curriculum)) & maps.reserved)
        self.assertEqual(curriculum[1600:], uniform[1600:])
        self.assertTrue(
            all(benchmark.gold_distance(layout) == 1 for layout in curriculum[:400])
        )
        self.assertTrue(
            all(benchmark.easy_layout(layout, 2) for layout in curriculum[400:1600])
        )
        self.assertEqual(
            benchmark.curriculum_layouts(
                nb.random_layout, 0, uniform[:800], maps.reserved
            ),
            curriculum[:800],
        )

    def test_measures_new_squares_not_turns_or_noops(self):
        cave = benchmark.MeasuredCave(gold=(2, 1), wumpus=(4, 4), pits=())
        cave.step("fwd")
        cave.step("climb")
        cave.step("grab")
        self.assertEqual((len(cave.reached), cave.last_new, cave.steps), (2, 1, 3))
        self.assertTrue(cave.has_gold)
        self.assertFalse(cave.done)


if __name__ == "__main__":
    unittest.main()
