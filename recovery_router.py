"""A learned choice between two full recovery policies, with no action masking."""

from random import Random

from benchmark_navigation import measured_cave
from shared_rule_lcs import condition_keys, progress_bucket


def router_keys(memory, sensors, repeats, recent, idle, spent, heard):
    gold = memory.pose.has_gold
    observations = len(memory.observations)
    return (
        *condition_keys(memory, sensors, wider_memory=True),
        ("route_root", gold),
        ("route_context", gold, sensors, observations // 3, spent, heard),
        ("route_recent", gold, sensors, progress_bucket(idle), tuple(recent)),
    )


class RecoveryRouter:
    """Regularized contextual estimate of exploratory-minus-cautious returns."""

    def __init__(self, cautious, exploratory, *, margin=0.03, death_price=0.60):
        if cautious.base is not exploratory.base:
            raise ValueError("both arms must share one frozen pre-trigger policy")
        self.cautious = cautious
        self.exploratory = exploratory
        self.margin = margin
        self.death_price = death_price
        self.weights = {}
        self.last_exploratory = False

    def estimate(self, keys):
        weights = [self.weights[k] for k in keys if k in self.weights]
        return sum(w[0] for w in weights), sum(w[1] for w in weights)

    def fit(self, examples, seed):
        if not examples:
            raise ValueError("no recurrent training episodes to route")
        rng = Random(seed)
        samples = list(examples)
        for _ in range(8):
            rng.shuffle(samples)
            for keys, score, death in samples:
                predicted_score, predicted_death = self.estimate(keys)
                step = 0.12 / len(keys)
                for k in keys:
                    pair = self.weights.setdefault(k, [0.0, 0.0])
                    pair[0] += step * (score - predicted_score - 0.50 * pair[0])
                    pair[1] += step * (death - predicted_death - 0.50 * pair[1])

    def select(self, *observation):
        gain, risk = self.estimate(router_keys(*observation))
        if gain - self.death_price * max(0.0, risk) > self.margin:
            self.last_exploratory = True
            return self.exploratory
        return self.cautious

    def evaluate(self, cave, cap, *, seed=0, trace=None):
        self.last_exploratory = False
        return self.cautious.evaluate(
            cave, cap, seed=seed, select=self.select, trace=trace
        )


def collect_examples(cautious, exploratory, layouts, cap, seed):
    """Compare actually observed training outcomes, always from the entrance."""
    examples = []
    actions = 0
    for i, layout in enumerate(layouts):
        keys = None

        def collect(*observation):
            nonlocal keys
            keys = router_keys(*observation)

        token = seed * 100003 + i
        a, score_a, steps, trigger = cautious.evaluate(
            measured_cave(layout), cap, seed=token, select=collect
        )
        actions += steps
        if not trigger:
            continue
        b, score_b, steps, other_trigger = exploratory.evaluate(
            measured_cave(layout), cap, seed=token
        )
        if trigger != other_trigger or keys is None:
            raise ValueError(
                "specialists must share the same deterministic pre-trigger prefix"
            )
        actions += steps
        examples.append(
            (keys, (score_b - score_a) / 1000, int(b == "death") - int(a == "death"))
        )
    return examples, actions
