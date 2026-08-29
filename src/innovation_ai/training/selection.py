"""Deterministic learned afterstate selection without engine-state access.

This module owns only the value aggregation and stochastic selector.  Candidate
states are expanded elsewhere from information-set samples; the selector never
accepts a :class:`~innovation_ai.innovation.state.GameState` and therefore has
no route to score the live game's hidden state.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from innovation_ai.harness.afterstates import CandidateExpansion
from innovation_ai.harness.policy import CandidateRoute, PolicySelection
from innovation_ai.innovation.actions import SemanticAction
from innovation_ai.innovation.types import PlayerId

SELECTION_RNG_VERSION = "sha256-domain-separated-v1"
SELECTOR_VERSION = "temperature-softmax-v1"


class SelectionError(ValueError):
    """Candidate values or selector inputs violate the learned-policy contract."""


class SelectionRngError(SelectionError):
    """A versioned policy RNG input is malformed."""


class SelectionDomain(StrEnum):
    """Independent uses of a per-decision policy random stream."""

    DETERMINIZATION = "determinizations"
    TEMPERATURE = "temperature"


@dataclass(frozen=True, slots=True)
class ActionValue:
    """All sampled values and their exact-semantic-action mean."""

    action: SemanticAction
    sample_values: tuple[float, ...]
    mean_value: float

    def __post_init__(self) -> None:
        if not self.sample_values:
            raise SelectionError("an action requires at least one sampled value")
        if not math.isfinite(self.mean_value) or not 0.0 <= self.mean_value <= 1.0:
            raise SelectionError("an action mean must be a finite probability")
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in self.sample_values):
            raise SelectionError("sampled action values must be finite probabilities")


@dataclass(frozen=True, slots=True)
class PolicyRngFactory:
    """Derive batch-order-independent SHA-256 streams for one policy decision.

    The required identity fields are intentionally explicit.  A game is assigned
    the same sampler and temperature stream whether it is evaluated alone or in
    an arbitrarily rebucketed multi-game inference batch.
    """

    run_seed: int | str | bytes
    generation: int
    version: str = SELECTION_RNG_VERSION

    def __post_init__(self) -> None:
        _seed_bytes(self.run_seed)
        if isinstance(self.generation, bool) or not isinstance(self.generation, int):
            raise SelectionRngError("generation must be an integer")
        if self.generation < 0:
            raise SelectionRngError("generation cannot be negative")
        if self.version != SELECTION_RNG_VERSION:
            raise SelectionRngError(f"unsupported selection RNG version {self.version!r}")

    def for_decision(
        self,
        *,
        game_id: str,
        chooser: PlayerId,
        decision_id: int,
        domain: SelectionDomain,
        policy_id: str = "",
    ) -> DecisionRng:
        """Return a fresh deterministic stream for one domain of one decision."""

        if not game_id:
            raise SelectionRngError("game ID cannot be empty")
        if decision_id < 1:
            raise SelectionRngError("decision ID must be positive")
        payload = b"\0".join(
            (
                b"innovation-ai",
                self.version.encode("ascii"),
                _seed_bytes(self.run_seed),
                str(self.generation).encode("ascii"),
                game_id.encode("utf-8"),
                chooser.value.encode("ascii"),
                str(decision_id).encode("ascii"),
                domain.value.encode("ascii"),
                policy_id.encode("utf-8"),
            )
        )
        return DecisionRng(hashlib.sha256(payload).digest(), self.version, domain)

    def seed_for_decision(
        self,
        *,
        game_id: str,
        chooser: PlayerId,
        decision_id: int,
        domain: SelectionDomain,
        policy_id: str = "",
    ) -> bytes:
        """Return a stable seed for a component that owns a separate RNG class."""

        return self.for_decision(
            game_id=game_id,
            chooser=chooser,
            decision_id=decision_id,
            domain=domain,
            policy_id=policy_id,
        ).seed


class DecisionRng:
    """Small SHA-256 counter RNG used only by one policy-decision domain."""

    def __init__(self, seed: bytes, version: str, domain: SelectionDomain) -> None:
        if not seed:
            raise SelectionRngError("decision RNG seed cannot be empty")
        self.seed = seed
        self._version = version
        self._domain = domain
        self._counter = 0
        self._buffer = b""

    def _bytes(self, count: int) -> bytes:
        if count < 0:
            raise SelectionRngError("random byte count cannot be negative")
        while len(self._buffer) < count:
            payload = b"\0".join(
                (
                    b"innovation-ai",
                    self._version.encode("ascii"),
                    b"decision-rng",
                    self._domain.value.encode("ascii"),
                    self.seed,
                    self._counter.to_bytes(16, "big"),
                )
            )
            self._buffer += hashlib.sha256(payload).digest()
            self._counter += 1
        value, self._buffer = self._buffer[:count], self._buffer[count:]
        return value

    def unit_interval(self) -> float:
        """Return a deterministic value in ``[0, 1)`` using 53 unbiased bits."""

        value = int.from_bytes(self._bytes(8), "big") >> 11
        return value / (1 << 53)


def _seed_bytes(seed: int | str | bytes) -> bytes:
    if isinstance(seed, bool):
        raise SelectionRngError("run seed cannot be boolean")
    if isinstance(seed, int):
        return f"int:{seed}".encode("ascii")
    if isinstance(seed, str):
        return b"str:" + seed.encode("utf-8")
    if isinstance(seed, bytes):
        return b"bytes:" + seed
    raise SelectionRngError("run seed must be an int, string, or bytes")


def _probability(value: float, *, name: str) -> float:
    scalar = float(value)
    if not math.isfinite(scalar):
        raise SelectionError(f"{name} must be finite")
    # Models should already cross the evaluator boundary in [0, 1].  Clamp only
    # a tiny numerical overshoot rather than changing the distribution's scale.
    tolerance = 1e-12
    if scalar < -tolerance or scalar > 1.0 + tolerance:
        raise SelectionError(f"{name} must be in [0, 1]")
    return min(1.0, max(0.0, scalar))


def aggregate_candidate_values(
    legal_actions: Sequence[SemanticAction],
    routes: Sequence[CandidateRoute],
    evaluated_values: Sequence[float],
    *,
    terminal_values: Sequence[tuple[CandidateRoute, float]] = (),
) -> tuple[ActionValue, ...]:
    """Group candidate values by exact semantic action and average samples.

    Every legal action must have the same complete set of sample indices.  This
    makes accidental grouping by action-list index, missing terminal samples, or
    cross-decision routing an immediate error rather than a silent bias.
    """

    actions = tuple(legal_actions)
    if not actions:
        raise SelectionError("selection requires legal actions")
    if len(set(actions)) != len(actions):
        raise SelectionError("legal semantic actions cannot repeat")
    if len(routes) != len(evaluated_values):
        raise SelectionError("candidate routes and evaluated values differ in length")

    values_by_action: dict[SemanticAction, dict[int, float]] = {action: {} for action in actions}

    def add(route: CandidateRoute, value: float, source: str) -> None:
        try:
            samples = values_by_action[route.action]
        except KeyError as error:
            message = f"{source} route is not a current legal semantic action"
            raise SelectionError(message) from error
        if route.sample_index in samples:
            raise SelectionError(f"semantic action has duplicate sample index {route.sample_index}")
        samples[route.sample_index] = _probability(value, name=f"{source} candidate value")

    for route, value in zip(routes, evaluated_values, strict=True):
        add(route, value, "evaluator")
    for route, value in terminal_values:
        add(route, value, "terminal")

    expected_indices: tuple[int, ...] | None = None
    output: list[ActionValue] = []
    for action in actions:
        samples = values_by_action[action]
        indices = tuple(sorted(samples))
        if not indices:
            raise SelectionError("each legal action requires at least one sampled value")
        if expected_indices is None:
            expected_indices = indices
        elif indices != expected_indices:
            raise SelectionError("legal actions do not have identical sampled-index coverage")
        sample_values = tuple(samples[index] for index in indices)
        output.append(ActionValue(action, sample_values, sum(sample_values) / len(sample_values)))
    return tuple(output)


def aggregate_expansion_values(
    expansion: CandidateExpansion,
    evaluated_values: Sequence[float],
) -> tuple[ActionValue, ...]:
    """Aggregate one trusted candidate expansion without evaluating terminals."""

    return aggregate_candidate_values(
        tuple(dict.fromkeys(route.action for route in expansion.all_routes)),
        expansion.routes,
        evaluated_values,
        terminal_values=tuple(
            (candidate.route, candidate.utility) for candidate in expansion.terminal_candidates
        ),
    )


def choose_temperature_action(
    action_values: Sequence[ActionValue],
    temperature: float,
    rng: DecisionRng | None = None,
) -> ActionValue:
    """Choose with stable temperature softmax, preserving legal order for argmax ties."""

    candidates = tuple(action_values)
    if not candidates:
        raise SelectionError("temperature selection requires candidate values")
    if isinstance(temperature, bool):
        raise SelectionError("temperature must be numeric")
    scalar_temperature = float(temperature)
    if not math.isfinite(scalar_temperature) or scalar_temperature < 0.0:
        raise SelectionError("temperature must be finite and non-negative")

    maximum = max(item.mean_value for item in candidates)
    if scalar_temperature == 0.0:
        # ``max`` is intentionally not used here: the loop's strict comparison
        # retains the original legal semantic-action order on equal values.
        selected = candidates[0]
        for item in candidates[1:]:
            if item.mean_value > selected.mean_value:
                selected = item
        return selected

    if rng is None:
        raise SelectionError("positive-temperature selection requires a policy RNG")
    weights = tuple(
        math.exp((item.mean_value - maximum) / scalar_temperature) for item in candidates
    )
    total = sum(weights)
    if not math.isfinite(total) or total <= 0.0:  # pragma: no cover - guarded by finite values
        raise SelectionError("temperature softmax produced invalid weights")
    threshold = rng.unit_interval() * total
    cumulative = 0.0
    for item, weight in zip(candidates, weights, strict=True):
        cumulative += weight
        if threshold < cumulative:
            return item
    # Rounding can leave a threshold infinitesimally above the final cumulative sum.
    return candidates[-1]


def select_expansion_action(
    *,
    policy_id: str,
    game_id: str,
    decision_id: int,
    legal_actions: Sequence[SemanticAction],
    expansion: CandidateExpansion,
    evaluated_values: Sequence[float],
    temperature: float,
    rng: DecisionRng | None = None,
) -> PolicySelection:
    """Turn trusted sampled afterstates into one auditable policy selection."""

    if not policy_id or not game_id:
        raise SelectionError("policy and game IDs cannot be empty")
    values = aggregate_candidate_values(
        legal_actions,
        expansion.routes,
        evaluated_values,
        terminal_values=tuple(
            (candidate.route, candidate.utility) for candidate in expansion.terminal_candidates
        ),
    )
    selected = choose_temperature_action(values, temperature, rng)
    return PolicySelection(
        policy_id,
        game_id,
        decision_id,
        selected.action,
        selected.mean_value,
        float(temperature),
    )


__all__ = [
    "SELECTION_RNG_VERSION",
    "SELECTOR_VERSION",
    "ActionValue",
    "DecisionRng",
    "PolicyRngFactory",
    "SelectionDomain",
    "SelectionError",
    "SelectionRngError",
    "aggregate_candidate_values",
    "aggregate_expansion_values",
    "choose_temperature_action",
    "select_expansion_action",
]
