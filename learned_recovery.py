"""Learn a residual rule table for recurrent observations; never filter commands."""

from collections import Counter, deque
from dataclasses import dataclass
from math import exp
from random import Random

from benchmark_transfer import terminal_outcome
from cave import EFFECTORS
from shared_rule_lcs import ACTION_COUNT, ACTION_IDS, condition_keys, progress_bucket


def observed_state(memory, sensors, spent, heard):
    """A repeat key, not a rule input: it compares the agent's own history."""
    pose = memory.pose
    return (
        pose.x,
        pose.y,
        pose.heading,
        pose.has_gold,
        memory.last_action,
        sensors,
        spent,
        heard,
        frozenset(memory.observations.items()),
    )


@dataclass(frozen=True)
class RecoverySettings:
    episodes: int = 12000
    history: bool = True
    alpha: float = 0.10
    gamma: float = 0.98
    base_strength: float = 0.40
    death_cost: float = 0.0
    timeout_cost: float = 0.0
    temperature: float = 0.0
    early: int = 0
    reheat: float = 0.0
    risk_weight: float = 0.0
    extra_history: bool = False
    reopen: bool = False
    novelty: float = 0.0
    replay: bool = False


class RecoveryPolicy:
    """One base and one learned correction. A repeated observation activates the latter."""

    def __init__(self, base, seed, settings=RecoverySettings()):
        if base.settings.loop_memory:
            raise ValueError("the exact-repeat gate requires stationary base memory")
        self.base = base
        self.settings = settings
        self.random = Random(seed)
        self.weights = {}
        self.death_weights = {}
        self.successes = deque(maxlen=128)
        self.episode_index = 0

    def residual_keys(
        self, memory, sensors, repeats, recent, idle, spent, heard, source
    ):
        keys = condition_keys(memory, sensors, wider_memory=True)
        if not self.settings.history:
            return keys
        gold = memory.pose.has_gold
        count, delay = progress_bucket(repeats), progress_bucket(idle)
        keys.extend(
            (
                ("recovery_repeat", gold, sensors, count),
                ("recovery_idle", gold, sensors, delay),
                ("recovery_history", gold, sensors, count, tuple(recent)),
                ("recovery_last", gold, sensors, delay, memory.last_action),
            )
        )
        if self.settings.extra_history:
            keys.extend(
                (
                    ("recovery_source", gold, source, sensors),
                    ("recovery_inventory", gold, sensors, spent, heard),
                    (
                        "recovery_inventory_last",
                        gold,
                        sensors,
                        spent,
                        heard,
                        memory.last_action,
                    ),
                )
            )
        return keys

    def evaluate(self, cave, cap, *, seed=0, select=None, trace=None):
        return self._episode(
            cave, cap, train=False, rng=Random(seed), select=select, trace=trace
        )

    def train_episode(self, cave, cap):
        outcome = self._episode(cave, cap, train=True, rng=self.random)
        self.episode_index += 1
        return outcome

    def back_up(self, records, *, death=False, imitate=False):
        """Monte Carlo credit on experienced transitions; no max-target bias."""
        actual = 0.0
        for rules, cost_rules, action, prior, learned, raw in reversed(records):
            actual = (raw if imitate else learned) + self.settings.gamma * actual
            prediction = prior + sum(rule[action] for rule in rules)
            error = actual - prediction
            if not imitate or error > 0:
                delta = self.settings.alpha * error / len(rules)
                for rule in rules:
                    rule[action] += delta
            if cost_rules and not imitate:
                error = int(death) - sum(rule[action] for rule in cost_rules)
                delta = self.settings.alpha * error / len(cost_rules)
                for rule in cost_rules:
                    rule[action] += delta

    @staticmethod
    def matching(table, keys, train):
        if train:
            return [table.setdefault(k, [0.0] * ACTION_COUNT) for k in keys]
        return [table[k] for k in keys if k in table]

    def _episode(self, cave, cap, *, train, rng, select=None, trace=None):
        settings = self.settings
        specialist = self
        # A routed specialist must get the same entire prefix it observes
        # when run by itself, even if the starting specialist ignores attempts.
        track_attempts = settings.extra_history or select is not None
        memory = type(self.base.memory)()
        sensors = cave.reset()
        memory.prime(sensors)
        memory.reset()
        seen = Counter()
        tried = Counter()
        recent = deque(maxlen=3)
        spent = heard = active = False
        source = "none"
        activated_at = idle = score = 0
        outcome = "timeout"
        records = []
        epsilon = max(
            0.08, 0.35 - 0.27 * self.episode_index / max(1, settings.episodes)
        )
        for index in range(cap):
            state = observed_state(memory, sensors, spent, heard)
            repeats = seen[state]
            seen[state] += 1
            if not active and (repeats or (settings.early and idle >= settings.early)):
                active = True
                activated_at = index + 1
                source = "repeat" if repeats else "idle"
            keys = condition_keys(
                memory, sensors, wider_memory=self.base.settings.wider_memory
            )
            base_rules = [self.base.weights[k] for k in keys if k in self.base.weights]
            if not active and not base_rules:
                outcome = "no_matching_action"
                break
            prior = self.base.estimate(base_rules)
            if (
                not active
                and settings.reopen
                and not memory.pose.has_gold
                and "at_start" in sensors
                and prior.index(max(prior)) == ACTION_IDS["climb"]
            ):
                active = True
                activated_at = index + 1
                source = "exit"
            if select is not None and activated_at == index + 1:
                selected = select(memory, sensors, repeats, recent, idle, spent, heard)
                if selected is not None:
                    if selected.base is not self.base or train:
                        raise ValueError(
                            "routing requires frozen specialists sharing one base"
                        )
                    specialist = selected
                    settings = selected.settings
            if active:
                keys = specialist.residual_keys(
                    memory, sensors, repeats, recent, idle, spent, heard, source
                )
                rules = self.matching(specialist.weights, keys, train)
                cost_rules = (
                    self.matching(specialist.death_weights, keys, train)
                    if settings.risk_weight
                    else []
                )
                if settings.extra_history:
                    attempt_keys = [
                        (
                            "attempts",
                            memory.pose.has_gold,
                            sensors,
                            progress_bucket(tried[state, a]),
                        )
                        for a in range(ACTION_COUNT)
                    ]
                    attempts = [
                        self.matching(specialist.weights, (k,), train)
                        for k in attempt_keys
                    ]
                    costs = (
                        [
                            self.matching(specialist.death_weights, (k,), train)
                            for k in attempt_keys
                        ]
                        if settings.risk_weight
                        else ()
                    )
                else:
                    attempts = costs = ()
                prior = tuple(settings.base_strength * value for value in prior)
                q = self.base.estimate(rules)
                risk = self.base.estimate(cost_rules)
                values = tuple(
                    prior[a]
                    + q[a]
                    + (sum(r[a] for r in attempts[a]) if attempts else 0.0)
                    - settings.risk_weight
                    * max(
                        0.0,
                        min(
                            1.0,
                            risk[a] + (sum(r[a] for r in costs[a]) if costs else 0.0),
                        ),
                    )
                    for a in range(ACTION_COUNT)
                )
            else:
                values = prior
            if active and train and rng.random() < epsilon:
                action = rng.randrange(ACTION_COUNT)
            elif active and settings.temperature and not train:
                top = max(values)
                temp = settings.temperature + settings.reheat * min(repeats, 4)
                weights = [exp((value - top) / temp) for value in values]
                action = rng.choices(range(ACTION_COUNT), weights, k=1)[0]
            else:
                action = values.index(max(values))
            command = EFFECTORS[action]
            if trace is not None:
                trace.append(
                    {
                        "command": command,
                        "square": [memory.pose.x, memory.pose.y],
                        "percepts": sorted(sensors),
                        "recovering": active,
                        "action_scores": dict(
                            zip(
                                EFFECTORS,
                                (round(v * 1000, 2) for v in values),
                                strict=True,
                            )
                        ),
                    }
                )
            if track_attempts:
                tried[state, action] += 1
            before, carried = len(memory.observations), memory.pose.has_gold
            after, reward, done = cave.step(command)
            if trace is not None:
                trace[-1]["reward"] = reward
            score += reward
            spent |= command == "shoot"
            heard |= "scream" in after
            memory.update(command, sensors, after)
            idle = (
                0
                if len(memory.observations) > before or carried != memory.pose.has_gold
                else idle + 1
            )
            recent.append(action)
            sensors = after
            if done:
                outcome = terminal_outcome(cave)
            if active and train:
                learned = reward
                if (
                    len(memory.observations) > before
                    and not carried
                    and outcome != "death"
                ):
                    learned += settings.novelty
                if outcome == "death":
                    learned -= settings.death_cost
                if not done and index + 1 == cap:
                    learned -= settings.timeout_cost
                records.append(
                    (
                        [*rules, *attempts[action]] if attempts else rules,
                        [*cost_rules, *costs[action]] if costs else cost_rules,
                        action,
                        prior[action],
                        learned / 1000,
                        reward / 1000,
                    )
                )
            if done:
                break
        if records:
            self.back_up(records, death=outcome == "death")
            if settings.replay:
                if outcome == "gold":
                    self.successes.append(records)
                if self.successes:
                    self.back_up(rng.choice(self.successes), imitate=True)
        return outcome, score, cave.steps, activated_at
