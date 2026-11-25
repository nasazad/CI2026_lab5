"""An LCS with reusable, automatically instantiated partial-match conditions."""

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from math import sqrt
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
    bootstrap_truncation: bool = False
    loop_memory: bool = False
    count_bonus: float = 0.0
    potential_points: float = 0.0
    anneal_start: int = 5000
    anneal_end: int = 10000
    balanced_replay: bool = False
    imitation: bool = False
    clean_replay: bool = False
    epsilon_final: float | None = None
    inventory_memory: bool = False
    target_mix: float = 0.0


class InventoryMemory(RawPerceptMemory):
    """Remember our own consumed arrow and whether its hit was observed."""

    def reset(self):
        super().reset()
        self.spent = self.heard = False

    def update(self, action, before, after):
        super().update(action, before, after)
        self.spent |= action == "shoot"
        self.heard |= "scream" in after

    def messages(self):
        return super().messages() | {
            f"arrow_spent:{self.spent}",
            f"scream_heard:{self.heard}",
        }


class ProgressMemory(RawPerceptMemory):
    """Summarize actually executed commands and observed movement."""

    def reset(self):
        super().reset()
        self.visits = Counter({(self.pose.x, self.pose.y): 1})
        self.since_new = self.since_move = self.turns_since_move = 0

    def update(self, action, before, after):
        old = self.pose.x, self.pose.y
        size = len(self.observations)
        super().update(action, before, after)
        current = self.pose.x, self.pose.y
        if current == old:
            self.since_move += 1
            self.turns_since_move += action in ("turn_l", "turn_r")
        else:
            self.visits[current] += 1
            self.since_move = self.turns_since_move = 0
        self.since_new = self.since_new + 1 if len(self.observations) == size else 0


def progress_bucket(steps):
    return 0 if steps == 0 else 1 if steps <= 3 else 2 if steps <= 8 else 3


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
    if isinstance(memory, ProgressMemory):
        front_x, front_y = PREVIEW_DIRECTIONS[h][0]
        here = min(memory.visits[x, y], 3)
        ahead = min(memory.visits[x + front_x, y + front_y], 2)
        since_new = progress_bucket(memory.since_new)
        since_move = progress_bucket(memory.since_move)
        turns = min(memory.turns_since_move, 2)
        features.extend(
            (
                ("visits", *context, here, ahead),
                ("progress", *context, since_new, ahead),
                ("delay", pose.has_gold, mask & LANDMARK_BITS, since_new, since_move),
                ("spinning", *context, turns, last),
                ("progress_front", *context, since_new, nearby[0]),
            )
        )
    if isinstance(memory, InventoryMemory):
        features.extend(
            (
                ("inventory", *context, memory.spent, memory.heard),
                ("inventory_last", *context, memory.spent, memory.heard, last),
                (
                    "inventory_axial",
                    *context,
                    memory.spent,
                    memory.heard,
                    nearby[0],
                    nearby[2],
                ),
            )
        )
    return features


def reward_for_learning(reward, outcome, before, after, settings):
    """Reward ablations are training-only and never alter action availability."""
    if outcome == "retreat" and after == 1:
        reward -= settings.penalty
    if outcome != "death" and after > before:
        reward += settings.novel_reward
    return reward


def potential_delta(gamma, before, after, terminal):
    """Count only observations made by the agent, with zero terminal potential."""
    next_potential = 0 if terminal else after - 1
    return gamma * next_potential - (before - 1)


def loop_free_indices(states):
    """Keep the last exit from a repeated state of a single observed episode."""
    kept, offsets = [], {}
    for i, state in enumerate(states):
        if state in offsets:
            cutoff = offsets[state]
            for old in kept[cutoff:]:
                del offsets[states[old]]
            del kept[cutoff:]
        offsets[state] = len(kept)
        kept.append(i)
    return kept


class SharedRuleLCS:
    """Q estimates from a set of overlapping condition–action rules.

    When an observed condition is new, the learner covers all six commands.
    The estimate for a command is the sum of contributions from matching
    rules. Each update divides its step size among the active conditions.
    """

    def __init__(self, seed, settings):
        if settings.inventory_memory and settings.loop_memory:
            raise ValueError("inventory and progress memory are separate ablations")
        if not 0 <= settings.target_mix <= 1:
            raise ValueError("target_mix must be between zero and one")
        if settings.clean_replay and (
            settings.loop_memory or settings.potential_points
        ):
            raise ValueError(
                "cycle erasure requires stationary raw-memory conditions and unshaped replay"
            )
        self.settings = settings
        memory_type = (
            ProgressMemory
            if settings.loop_memory
            else InventoryMemory
            if settings.inventory_memory
            else RawPerceptMemory
        )
        self.memory = memory_type()
        self.weights = {}
        self.random = Random(seed)
        self.successes = deque(maxlen=128)
        self.success_buckets = defaultdict(lambda: deque(maxlen=128))
        self.action_counts = {}
        self.episode_index = 0

    def annealing(self):
        begin, end = self.settings.anneal_start, self.settings.anneal_end
        if not 0 <= begin < end:
            raise ValueError("annealing requires 0 <= start < end")
        return min(1.0, max(0.0, (end - self.episode_index) / (end - begin)))

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

    def target_value(self, rules):
        values = self.estimate(rules)
        best = max(values)
        mix = self.settings.target_mix
        return (1 - mix) * best + mix * sum(values) / ACTION_COUNT if mix else best

    def train_episode(self, cave, cap):
        cycle_states = [] if self.settings.clean_replay else None
        outcome, _, steps, _, history, tail, factors, distance = self._play(
            cave, cap, train=True, cycle_states=cycle_states
        )
        scale = (
            self.settings.potential_points * self.annealing() / 1000
            if self.settings.potential_points
            else 0
        )
        self.last_learning_signal = self.back_up(
            history, bootstrap=tail, potential_factors=factors, scale=scale
        )
        if self.settings.replay:
            if outcome == "gold":
                kept = (
                    loop_free_indices(cycle_states)
                    if cycle_states is not None
                    else None
                )
                replay_history = (
                    tuple(history) if kept is None else tuple(history[i] for i in kept)
                )
                experience = (replay_history, tuple(factors))
                if self.settings.balanced_replay:
                    self.success_buckets[min(distance, 3)].append(experience)
                else:
                    self.successes.append(experience)
            if self.settings.balanced_replay:
                buckets = tuple(self.success_buckets.values())
                if buckets:
                    for _ in range(self.settings.replay):
                        past, past_factors = self.random.choice(
                            self.random.choice(buckets)
                        )
                        self.rehearse(past, past_factors, scale)
            else:
                for _ in range(self.settings.replay if self.successes else 0):
                    past, past_factors = self.random.choice(self.successes)
                    self.rehearse(past, past_factors, scale)
        self.episode_index += 1
        return outcome, steps

    def rehearse(self, history, factors, scale):
        if not self.settings.imitation:
            self.back_up(history, potential_factors=factors, scale=scale)
            return
        # Self-imitation only uses the observed return of a successful trip.
        # Do not increase a previously stronger estimate with a weaker replay.
        actual = 0.0
        for i in range(len(history) - 1, -1, -1):
            rules, action, reward = history[i]
            actual = reward + self.settings.gamma * actual
            if scale:
                actual += scale * factors[i]
            advantage = actual - sum(v[action] for v in rules)
            if advantage > 0:
                delta = self.settings.alpha * advantage / len(rules)
                for v in rules:
                    v[action] += delta

    def back_up(self, history, *, bootstrap=0.0, potential_factors=(), scale=0.0):
        """Use at most n rewards and bootstrap unless the trajectory terminated."""
        n, gamma, alpha = self.settings.n, self.settings.gamma, self.settings.alpha
        powers = [gamma**i for i in range(n + 1)]
        length = len(history)
        if not length:
            return 0.0
        # Accumulate discounted rewards in a reverse rolling window.
        segment = 0.0
        positive_error = 0.0
        for i in range(length - 1, -1, -1):
            rules, action, reward = history[i]
            if scale:
                reward += scale * potential_factors[i]
            segment = reward + gamma * segment
            target = segment
            if i + n < length:
                expired = history[i + n][2]
                if scale:
                    expired += scale * potential_factors[i + n]
                segment -= powers[n] * expired
                target = segment + powers[n] * self.target_value(history[i + n][0])
            elif bootstrap:
                target += powers[length - i] * bootstrap
            q = sum(v[action] for v in rules)
            positive_error += max(0.0, target - q)
            delta = alpha * (target - q) / len(rules)
            for v in rules:
                v[action] += delta
        return positive_error / length

    def evaluate(self, cave, cap, *, trace=False):
        outcome, score, steps, records, *_ = self._play(
            cave, cap, train=False, trace=trace
        )
        return outcome, score, steps, records

    def _play(self, cave, cap, *, train, trace=False, cycle_states=None):
        sensors = cave.reset()
        memory = self.memory
        memory.prime(sensors)
        memory.reset()
        history, records, factors = [], [], []
        score = 0
        distance = 0
        arrow_spent = scream_heard = False
        bonus = (
            self.settings.count_bonus * self.annealing()
            if train and self.settings.count_bonus
            else 0
        )
        epsilon = self.settings.epsilon
        if train and self.settings.epsilon_final is not None:
            end = self.settings.epsilon_final
            epsilon = end + (epsilon - end) * self.annealing()
        for t in range(cap):
            if cycle_states is not None:
                pose = memory.pose
                cycle_states.append(
                    (
                        pose.x,
                        pose.y,
                        pose.heading,
                        pose.has_gold,
                        memory.last_action,
                        sensors,
                        frozenset(memory.observations.items()),
                        arrow_spent,
                        scream_heard,
                    )
                )
            rules = self.matching_rules(sensors, create=train)
            if not rules:
                return (
                    "no_matching_action",
                    score,
                    t,
                    records,
                    history,
                    0,
                    factors,
                    distance,
                )
            values = self.estimate(rules)
            counts = None
            if bonus:
                key = sensors, memory.messages()
                counts = self.action_counts.setdefault(key, [0] * ACTION_COUNT)
                values = tuple(
                    q + bonus / sqrt(1 + count) for q, count in zip(values, counts)
                )
            if train and self.random.random() < epsilon:
                action = self.random.randrange(ACTION_COUNT)
            else:
                best = max(values)
                tied = [i for i, q in enumerate(values) if q == best]
                action = (
                    self.random.choice(tied) if train and len(tied) > 1 else tied[0]
                )
            effector = EFFECTORS[action]
            if counts is not None:
                counts[action] += 1
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
            if cycle_states is not None:
                arrow_spent |= effector == "shoot"
                scream_heard |= "scream" in after
            outcome = terminal_outcome(cave) if done else None
            carried = memory.pose.has_gold
            memory.update(effector, sensors, after)
            if self.settings.balanced_replay and not carried and memory.pose.has_gold:
                distance = abs(memory.pose.x - 1) + abs(memory.pose.y - 1)
            score += reward
            if train:
                shaped = reward_for_learning(
                    reward, outcome, before, len(memory.observations), self.settings
                )
                history.append((rules, action, shaped / 1000))
                if self.settings.potential_points:
                    factors.append(
                        potential_delta(
                            self.settings.gamma, before, len(memory.observations), done
                        )
                    )
            if trace:
                records[-1]["reward"] = reward
            if done:
                return outcome, score, t + 1, records, history, 0, factors, distance
            sensors = after
        tail = 0
        if train and self.settings.bootstrap_truncation:
            tail = self.target_value(self.matching_rules(sensors, create=True))
        return "timeout", score, cap, records, history, tail, factors, distance


def average_rule_tables(agents):
    """Freeze a single LCS with mean scores across identically defined rules."""
    first = agents[0]
    if any(agent.settings != first.settings for agent in agents):
        raise ValueError("rule-table averaging requires identical settings")
    merged = SharedRuleLCS(0, first.settings)
    for agent in agents:
        for key, values in agent.weights.items():
            combined = merged.weights.setdefault(key, [0.0] * ACTION_COUNT)
            for i, value in enumerate(values):
                combined[i] += value / len(agents)
    return merged
