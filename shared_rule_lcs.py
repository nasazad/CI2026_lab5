"""An LCS with reusable, automatically instantiated partial-match conditions."""

from collections import deque
from dataclasses import dataclass
from random import Random

from benchmark_transfer import terminal_outcome
from cave import DELTA, EFFECTORS, HEADINGS, SENSORS
from primitive_memory import RawPerceptMemory


ACTION_COUNT = len(EFFECTORS)
SENSOR_BITS = {name: 1 << i for i, name in enumerate(SENSORS)}
ACTION_IDS = {name: i for i, name in enumerate(EFFECTORS)}
LANDMARK_BITS = SENSOR_BITS["glitter"] | SENSOR_BITS["at_start"]
EVENT_BITS = SENSOR_BITS["bump"] | SENSOR_BITS["scream"]
PREVIEW_DIRECTIONS = tuple(
    tuple(DELTA[HEADINGS[(h + i) % 4]] for i in range(4)) for h in range(4)
)
HEAD_INDEX = {name: i for i, name in enumerate(HEADINGS)}


@dataclass(frozen=True)
class Settings:
    n: int = 12
    gamma: float = 0.98
    alpha: float = 0.10
    epsilon: float = 0.20
    penalty: float = 0.0
    novel_reward: float = 0.0
    replay: int = 0
    wider_memory: bool = False


def condition_keys(memory, sensors, *, wider_memory=False):
    """Generic partial views of memory, independent of action or cave layout."""
    pose = memory.pose
    x, y, h = pose.x, pose.y, HEAD_INDEX[pose.heading]
    mask = sum(SENSOR_BITS[name] for name in sensors)
    context = pose.has_gold, mask
    last = memory.last_action
    nearby = tuple(
        memory.observations.get((x + dx, y + dy), "?")
        for dx, dy in PREVIEW_DIRECTIONS[h]
    )
    # Slice the observed conjunction in several ways. Each active condition
    # contributes to all six action estimates, but receives credit for just
    # the chosen action. No template associates an observation with a command.
    features = [
        ("landmarks", pose.has_gold, mask & LANDMARK_BITS),
        ("sensors", *context),
        ("history", *context, last),
        ("front_back", *context, nearby[0], nearby[2]),
        ("sides", *context, nearby[1], nearby[3]),
        ("ring", *context, nearby),
        ("exact", *context, last, nearby),
    ]
    for i, clue in enumerate(nearby):
        features.append(("neighbor", i, *context, clue))
        features.append(
            ("neighbor_no_events", i, pose.has_gold, mask & ~EVENT_BITS, clue)
        )
    if wider_memory:
        # A raw observed record can be placed anywhere in an egocentric map.
        # In particular, the entrance mark is the remembered `at_start` bit.
        # Nothing derives a route, a best direction, or a safety label.
        ((fx, fy), (lx, ly), _, _) = PREVIEW_DIRECTIONS[h]
        for (ox, oy), clue in sorted(memory.observations.items()):
            dx, dy = ox - x, oy - y
            if dx == dy == 0:
                continue
            front, left = dx * fx + dy * fy, dx * lx + dy * ly
            features.append(("record", *context, front, left, clue))
    return features


def reward_for_learning(reward, outcome, before, after, settings):
    """Reward ablations are training-only and never alter action availability."""
    if outcome == "retreat" and after == 1:
        reward -= settings.penalty
    if outcome != "death" and after > before:
        reward += settings.novel_reward
    return reward


class SharedRuleLCS:
    """Q estimates from a set of overlapping condition–action rules.

    When an observed condition is new, the learner covers all six commands.
    The estimate for a command is the sum of contributions from matching
    rules. Each update divides its step size among the active conditions.
    """

    def __init__(self, seed, settings):
        self.settings = settings
        self.memory = RawPerceptMemory()
        self.weights = {}
        self.random = Random(seed)
        self.successes = deque(maxlen=128)

    def matching_rules(self, sensors, *, create):
        keys = condition_keys(
            self.memory, sensors, wider_memory=self.settings.wider_memory
        )
        if create:
            weight = self.weights
            found = []
            for key in keys:
                values = weight.get(key)
                if values is None:
                    values = [0.0] * ACTION_COUNT
                    weight[key] = values
                found.append(values)
            return found
        return [self.weights[key] for key in keys if key in self.weights]

    @staticmethod
    def estimate(rules):
        a = b = c = d = e = f = 0.0
        for v in rules:
            a += v[0]
            b += v[1]
            c += v[2]
            d += v[3]
            e += v[4]
            f += v[5]
        return a, b, c, d, e, f

    def train_episode(self, cave, cap):
        outcome, _, steps, _, history = self._play(cave, cap, train=True)
        self.back_up(history)
        if self.settings.replay:
            if outcome == "gold":
                self.successes.append(tuple(history))
            for _ in range(self.settings.replay if self.successes else 0):
                self.back_up(self.random.choice(self.successes))
        return outcome, steps

    def back_up(self, history):
        """Truncated n-step target, bootstrapped from the highest next value."""
        n, gamma, alpha = self.settings.n, self.settings.gamma, self.settings.alpha
        powers = [gamma**i for i in range(n + 1)]
        length = len(history)
        if not length:
            return
        # Accumulate discounted rewards in a reverse rolling window.
        segment = 0.0
        for i in range(length - 1, -1, -1):
            rules, action, reward = history[i]
            segment = reward + gamma * segment
            target = segment
            if i + n < length:
                segment -= powers[n] * history[i + n][2]
                target = segment + powers[n] * max(self.estimate(history[i + n][0]))
            q = sum(v[action] for v in rules)
            delta = alpha * (target - q) / len(rules)
            for v in rules:
                v[action] += delta

    def evaluate(self, cave, cap, *, trace=False):
        outcome, score, steps, records, _ = self._play(
            cave, cap, train=False, trace=trace
        )
        return outcome, score, steps, records

    def _play(self, cave, cap, *, train, trace=False):
        sensors = cave.reset()
        memory = self.memory
        memory.prime(sensors)
        memory.reset()
        history, records = [], []
        score = 0
        for t in range(cap):
            rules = self.matching_rules(sensors, create=train)
            if not rules:
                return "no_matching_action", score, t, records, history
            values = self.estimate(rules)
            if train and self.random.random() < self.settings.epsilon:
                action = self.random.randrange(ACTION_COUNT)
            else:
                best = max(values)
                tied = [i for i, q in enumerate(values) if q == best]
                action = (
                    self.random.choice(tied) if train and len(tied) > 1 else tied[0]
                )
            effector = EFFECTORS[action]
            if trace:
                records.append(
                    {
                        "square": (memory.pose.x, memory.pose.y),
                        "percepts": sorted(sensors),
                        "memory": sorted(memory.messages()),
                        "action": effector,
                        "action_values": dict(
                            zip(EFFECTORS, (round(1000 * q, 4) for q in values))
                        ),
                    }
                )
            before = len(memory.observations)
            after, reward, done = cave.step(effector)
            outcome = terminal_outcome(cave) if done else None
            memory.update(effector, sensors, after)
            score += reward
            if train:
                shaped = reward_for_learning(
                    reward, outcome, before, len(memory.observations), self.settings
                )
                history.append((rules, action, shaped / 1000))
            if trace:
                records[-1]["reward"] = reward
            if done:
                return outcome, score, t + 1, records, history
            sensors = after
        return "timeout", score, cap, records, history
