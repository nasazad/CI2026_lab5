"""Rule inputs and an exact matcher for the randomized-cave experiments."""

from collections import defaultdict

from benchmark_transfer import Rollout, terminal_outcome
from cave import SENSORS
from classifiers import auction
from learning import LearnedClassifier, LearningCS


SENSOR_NAMES = frozenset(SENSORS)


class RelativeInputs:
    """Supply belief features, never coordinates or hidden cave attributes."""

    def __init__(self, notebook):
        self.notebook = notebook

    def prime(self, sensors):
        self.initial_percepts = sensors

    def reset(self):
        self.memory = self.notebook.Memory()
        self.memory.observe(self.initial_percepts)
        self.geometry = {}
        self.hazards = None

    def update(self, effector, before, after):
        mem = self.memory
        was_dead = mem.wumpus_dead
        dest = (
            self.notebook.advance(mem.pos, mem.heading)
            if effector == "fwd" and "bump" not in after
            else mem.pos
        )
        old_clues = mem.seen.get(dest)
        # The notebook only attempts grab on glitter; the LCS can try it anywhere.
        action = None if effector == "grab" and "glitter" not in before else effector
        mem.observe(after, action)
        if mem.wumpus_dead != was_dead or mem.seen[dest] != old_clues:
            self.geometry.clear()
            self.hazards = None

    def messages(self):
        mem, nb = self.memory, self.notebook
        pose = mem.pos, mem.heading
        if pose not in self.geometry:
            if self.hazards is None:
                self.hazards = mem.hazards()
            pits, wumpus, confirmed = self.hazards
            ahead = nb.advance(*pose)
            if ahead not in nb.BOARD:
                front = "ahead_wall"
            elif ahead in confirmed or (len(wumpus) == 1 and ahead in wumpus):
                front = "ahead_known_hazard"
            elif ahead in pits or ahead in wumpus:
                front = "ahead_uncertain"
            else:
                front = "ahead_safe"

            home = nb.route(mem.pos, {nb.START}, mem.seen)
            if mem.pos == nb.START:
                direction = "home_here"
            elif home:
                dx, dy = home[0][0] - mem.pos[0], home[0][1] - mem.pos[1]
                turn = (nb.DIRECTIONS.index((dx, dy)) - mem.heading) % 4
                direction = ("home_ahead", "home_left", "home_behind", "home_right")[
                    turn
                ]
            else:
                direction = "home_unknown"
            self.geometry[pose] = front, direction
        front, direction = self.geometry[pose]
        gold = "has_gold" if "has_gold" in mem.flags else "no_gold"
        return frozenset((front, direction, gold))


class IndexedLCS(LearningCS):
    """Use exactly the upstream learner, with a faster ordered subset lookup.

    Each variant emits a fixed number of internal messages. One internal set
    can be a subset of another of that same size only when they are equal.
    Thus only the rules in that internal-message bucket need testing.
    """

    def __init__(self, memory, seed, arity, rules=(), *, climb_guard=False):
        super().__init__(memory=memory, seed=seed)
        self.arity = arity
        self.climb_guard = climb_guard
        self.buckets = defaultdict(list)
        for rule in rules:
            self._insert(rule)

    def _key(self, messages):
        internal = messages - SENSOR_NAMES
        if len(internal) != self.arity:
            raise ValueError(f"expected {self.arity} internal messages, got {internal}")
        return internal

    def _insert(self, rule):
        self.rules.append(rule)
        self.buckets[self._key(rule.condition)].append(rule)

    def applicable(self, messages):
        return [
            r
            for r in self.buckets.get(self._key(messages), ())
            if r.matches(messages) and self.action_available(r.action, messages)
        ]

    def action_available(self, action, messages):
        return not self.climb_guard or "climb" not in action or "at_start" in messages

    def match(self, messages):
        match = self.applicable(messages)
        present = {rule.action for rule in match}
        for action in self.actions:
            if action not in present and self.action_available(action, messages):
                rule = LearnedClassifier(
                    messages, action, self.initial_strength, stamp=self.t
                )
                self._insert(rule)
                match.append(rule)
        return match

    def reset_inputs(self, sensors):
        prime = getattr(self.memory, "prime", None)
        if prime is not None:
            prime(sensors)
        self.memory.reset()

    def episode(self, cave):
        prime = getattr(self.memory, "prime", None)
        if prime is not None:
            # Prime from the public reset observation. The upstream episode
            # resets the deterministic simulator again before its first action.
            prime(cave.reset())
        return super().episode(cave)

    def greedy_episode(self, cave, limit, *, trace=False):
        """Frozen upstream auction semantics, with optional diagnostic trace."""
        sensors = cave.reset()
        self.reset_inputs(sensors)
        score, steps = 0, []
        for t in range(limit):
            messages = sensors | self.memory.messages()
            winner = auction(self.applicable(messages))
            if winner is None:
                return "no_matching_action", score, t, steps, sorted(messages)
            action, square = winner.effector, cave.pos
            after, reward, done = cave.step(action)
            if trace:
                steps.append(
                    {
                        "square": square,
                        "messages": sorted(messages),
                        "condition": sorted(winner.condition),
                        "action": action,
                        "reward": reward,
                    }
                )
            score += reward
            self.memory.update(action, sensors, after)
            if done:
                return terminal_outcome(cave), score, t + 1, steps, []
            sensors = after
        return "timeout", score, limit, steps, []

    def trace_rollout(self, cave, limit):
        outcome, score, _, steps, unmatched = self.greedy_episode(
            cave, limit, trace=True
        )
        return Rollout(outcome, score, steps, tuple(unmatched))
