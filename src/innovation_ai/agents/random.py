"""Seeded random baseline agent."""

from __future__ import annotations

import random

from innovation_ai.innovation.actions import Decision, SemanticAction

AGENT_RNG_VERSION = "python-mt19937-randrange-v1"


class RandomAgent:
    """Select legal actions uniformly with an isolated seeded RNG."""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._rng = random.Random(seed)

    def choose_action(self, decision: Decision, /) -> SemanticAction:
        """Choose a legal action without touching Python's module-global RNG."""

        return decision.legal_actions[self._rng.randrange(len(decision.legal_actions))]


SeededRandomAgent = RandomAgent
