"""Deterministic seeded protocol fuzzing built on the reusable WP10 invariants."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass

from innovation_ai.innovation.actions import DogmaAction, SemanticAction, action_payload
from innovation_ai.innovation.catalog import CardRegistry, load_card_registry
from innovation_ai.innovation.invariants import assert_state_properties, checked_apply_action
from innovation_ai.innovation.protocol import current_decisions
from innovation_ai.innovation.state import TerminalResult, build_setup_state, state_hash


class ProtocolFuzzError(RuntimeError):
    """A deterministic fuzz run stalled or exceeded its safety ceiling."""


@dataclass(frozen=True, slots=True)
class ProtocolFuzzStep:
    """One reproducible submitted action and its state hashes."""

    number: int
    action: SemanticAction
    before_hash: str
    after_hash: str


@dataclass(frozen=True, slots=True)
class ProtocolFuzzResult:
    """Compact deterministic record of one completed protocol fuzz game."""

    seed: int
    steps: tuple[ProtocolFuzzStep, ...]
    terminal: TerminalResult
    final_state_hash: str

    @property
    def trace_digest(self) -> str:
        """Return a stable digest suitable for small golden-record assertions."""

        payload = [
            {
                "number": step.number,
                "action": action_payload(step.action),
                "before_hash": step.before_hash,
                "after_hash": step.after_hash,
            }
            for step in self.steps
        ]
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _supported_actions(actions: tuple[SemanticAction, ...]) -> tuple[SemanticAction, ...]:
    """Exclude Dogma until WP4 supplies a resumable effect transition path."""

    return tuple(action for action in actions if not isinstance(action, DogmaAction))


def run_protocol_fuzz(
    seed: int,
    registry: CardRegistry | None = None,
    *,
    max_steps: int = 512,
) -> ProtocolFuzzResult:
    """Play a deterministic legal game over the complete currently executable WP3 protocol.

    Setup choices and paid Draw/Meld/Achieve actions are selected from enumerated legal actions.
    Dogma is intentionally not selected: WP3 only creates a placeholder pending frame and card
    effects do not yet have a resume API.  The fuzzer still verifies Dogma action enumeration via
    the legal-action-completeness invariant on every state.
    """

    if max_steps < 1:
        raise ValueError("protocol fuzz max_steps must be positive")
    registry = registry or load_card_registry()
    rng = random.Random(seed)
    state = build_setup_state(seed, registry)
    assert_state_properties(state, registry)
    steps: list[ProtocolFuzzStep] = []

    for number in range(1, max_steps + 1):
        if state.terminal_result is not None:
            return ProtocolFuzzResult(
                seed,
                tuple(steps),
                state.terminal_result,
                state_hash(state),
            )
        decisions = current_decisions(state, registry)
        if not decisions:
            raise ProtocolFuzzError(
                f"seed {seed} stalled after {len(steps)} steps with no protocol decision"
            )
        decision = decisions[rng.randrange(len(decisions))]
        actions = _supported_actions(decision.legal_actions)
        if not actions:
            raise ProtocolFuzzError(
                f"seed {seed} decision {decision.decision_id} has no WP3-executable action"
            )
        action = actions[rng.randrange(len(actions))]
        before_hash = state_hash(state)
        transition = checked_apply_action(state, action, registry)
        state = transition.state
        steps.append(ProtocolFuzzStep(number, action, before_hash, state_hash(state)))

    raise ProtocolFuzzError(f"seed {seed} exceeded the {max_steps}-step ceiling")


def run_protocol_fuzz_seeds(
    seeds: range | tuple[int, ...],
    registry: CardRegistry | None = None,
    *,
    max_steps: int = 512,
) -> tuple[ProtocolFuzzResult, ...]:
    """Run a deterministic seed batch, reusing one immutable card registry."""

    registry = registry or load_card_registry()
    return tuple(run_protocol_fuzz(seed, registry, max_steps=max_steps) for seed in seeds)
