"""Minimal policy boundary used by game runners."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from innovation_ai.innovation.actions import Decision, SemanticAction


@runtime_checkable
class Agent(Protocol):
    """Choose one legal semantic action from a player-safe decision."""

    def choose_action(self, decision: Decision, /) -> SemanticAction:
        """Return one of ``decision.legal_actions``."""
