"""A training-only level sampler; never reads the layout to assign a score."""

from collections import OrderedDict
from dataclasses import dataclass
from random import Random


@dataclass
class Level:
    priority: float
    episode: int


class LevelReplay:
    """Revisit promising training episodes or old levels from their normal start.

    A layout is an opaque key for the sampler. Its priority comes solely
    from the learner's positive TD errors on past attempts.
    """

    def __init__(self, seed, rate, *, warmup=1600, capacity=512, prioritized=True):
        self.rng = Random(seed)
        self.rate, self.warmup, self.capacity = rate, warmup, capacity
        self.levels = OrderedDict()
        self.replays = 0
        self.prioritized = prioritized

    def choose(self, fresh, episode):
        if episode < self.warmup or not self.levels or self.rng.random() >= self.rate:
            return fresh
        if not self.prioritized:
            self.replays += 1
            return self.rng.choice(tuple(self.levels))
        keys = self.rng.sample(tuple(self.levels), min(len(self.levels), 8))
        if self.rng.random() < 0.15:
            choice = min(keys, key=lambda k: self.levels[k].episode)
        else:
            keys.sort(key=lambda k: self.levels[k].priority, reverse=True)
            rank = int(len(keys) * self.rng.random() ** 3)
            choice = keys[rank]
        self.replays += 1
        return choice

    def record(self, layout, episode, priority):
        old = self.levels.pop(layout, None)
        if old is not None:
            priority = 0.6 * priority + 0.4 * old.priority
        self.levels[layout] = Level(priority, episode)
        if len(self.levels) > self.capacity:
            self.levels.popitem(last=False)
