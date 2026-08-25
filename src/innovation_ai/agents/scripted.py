"""Explicit scripted policy for deterministic fixtures."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable

from innovation_ai.innovation.actions import Decision, SemanticAction

type ScriptStep = SemanticAction | Callable[[Decision], SemanticAction]


class ScriptedAgentError(RuntimeError):
    """Base class for invalid or incomplete fixture scripts."""


class ScriptExhaustedError(ScriptedAgentError):
    """A decision was requested after the fixture script ended."""


class ScriptedIllegalActionError(ScriptedAgentError):
    """A fixture script produced an action that is not legal now."""


class ScriptedAgent:
    """Consume exact actions or decision-aware selectors in FIFO order."""

    def __init__(self, steps: Iterable[ScriptStep]) -> None:
        self._steps: deque[ScriptStep] = deque(steps)

    @property
    def remaining(self) -> int:
        """Return the number of unconsumed script steps."""

        return len(self._steps)

    def choose_action(self, decision: Decision, /) -> SemanticAction:
        """Consume the next script step and validate its semantic action."""

        if not self._steps:
            raise ScriptExhaustedError(f"script exhausted before decision {decision.decision_id}")
        step = self._steps.popleft()
        action = step(decision) if callable(step) else step
        if action not in decision.legal_actions:
            raise ScriptedIllegalActionError(
                f"scripted action {action.kind.value} is illegal for decision "
                f"{decision.decision_id}"
            )
        return action

    def assert_consumed(self) -> None:
        """Fail a fixture that completed before consuming its full script."""

        if self._steps:
            raise ScriptedAgentError(f"script has {len(self._steps)} unconsumed step(s)")
