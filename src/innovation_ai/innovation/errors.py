"""Typed engine errors shared by the protocol, effect VM, and dogma orchestration.

These live below :mod:`innovation_ai.innovation.protocol` so the effect executor can raise and
catch them without importing the paid-turn protocol, which imports the executor.
"""

from __future__ import annotations

from innovation_ai.innovation.actions import Action, Decision


class InnovationEngineError(RuntimeError):
    """Base class for recoverable protocol errors and engine defects."""


class IllegalAction(InnovationEngineError):
    """An agent submitted an action outside the current legal-action set."""

    def __init__(self, action: Action, decision: Decision) -> None:
        self.action = action
        self.decision = decision
        super().__init__(f"illegal {action.kind.value} action for decision {decision.decision_id}")


class EngineInvariantError(InnovationEngineError):
    """The engine reached a state that violates the transition protocol."""
