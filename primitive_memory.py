"""Remember local observations without making a navigation decision."""

from cave import DELTA, HEADINGS
from learning import DeadReckoning


OBSERVED = frozenset(("breeze", "stench", "at_start"))
SLOTS = ("front", "left", "back", "right")


class RawPerceptMemory:
    """Remember each square's last static clues and the most recent command.

    Coordinates synchronize observations internally. Rules see the observed
    bits in four relative slots, not coordinates, safety, or a route. A slot
    with no observation is unknown whether in bounds or beyond a wall.
    """

    ARITY = len(SLOTS) + 2

    def __init__(self):
        self.pose = DeadReckoning()

    def prime(self, sensors):
        self.initial = sensors

    def reset(self):
        self.pose.reset()
        self.last_action = "none"
        self.observations = {(self.pose.x, self.pose.y): self.clues(self.initial)}

    @staticmethod
    def clues(sensors):
        return "+".join(sorted(OBSERVED & sensors)) or "none"

    def update(self, action, before, after):
        self.pose.update(action, before, after)
        self.last_action = action
        self.observations[self.pose.x, self.pose.y] = self.clues(after)

    def messages(self):
        x, y, heading = self.pose.x, self.pose.y, HEADINGS.index(self.pose.heading)
        facts = [
            f"last_{self.last_action}",
            "has_gold" if self.pose.has_gold else "no_gold",
        ]
        for i, name in enumerate(SLOTS):
            dx, dy = DELTA[HEADINGS[(heading + i) % len(HEADINGS)]]
            clues = self.observations.get((x + dx, y + dy), "unknown")
            facts.append(f"remembered_{name}:{clues}")
        return frozenset(facts)
