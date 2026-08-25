"""Hash-verifying replay and recording abstractions for Innovation game logs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from innovation_ai.innovation.actions import (
    ACTION_SCHEMA_VERSION,
    DECISION_SCHEMA_VERSION,
    Decision,
    SemanticAction,
)
from innovation_ai.innovation.catalog import CardRegistry, load_card_registry
from innovation_ai.innovation.logs import (
    ENGINE_VERSION,
    GAME_LOG_FORMAT,
    GAME_LOG_SCHEMA_VERSION,
    GameLog,
    LoggedTransition,
    ReplayOutcome,
)
from innovation_ai.innovation.observations import OBSERVATION_SCHEMA_VERSION
from innovation_ai.innovation.protocol import (
    InnovationEngineError,
    apply_action,
    current_decisions,
)
from innovation_ai.innovation.state import (
    INFORMATION_POLICY_VERSION,
    RULES_VERSION,
    SETUP_RNG_VERSION,
    STATE_SCHEMA_VERSION,
    TERMINAL_SCHEMA_VERSION,
    GamePhase,
    GameState,
    SetupProvenance,
    TerminalResult,
    build_setup_state_from_piles,
    state_hash,
)


class ReplayError(RuntimeError):
    """Base class for replay compatibility and divergence failures."""


class ReplayCompatibilityError(ReplayError):
    """A log cannot be interpreted by this engine/catalog version."""


class ReplayDivergenceError(ReplayError):
    """Replay did not reproduce a recorded decision, state hash, or outcome."""


class ReplayRecordingError(ReplayError):
    """A caller attempted to record from a non-replayable boundary."""


class ReplayAdapter(Protocol):
    """Engine boundary used by recorder/replay, replaceable by WP4 integration.

    A later adapter may resolve serializable effect frames inside ``apply`` before returning the
    next decision boundary. Logs and replay remain independent of the concrete effect executor.
    """

    def initial_state(self, setup: SetupProvenance, registry: CardRegistry) -> GameState:
        """Reconstruct the initial state from explicit setup provenance."""

    def decisions(self, state: GameState, registry: CardRegistry) -> tuple[Decision, ...]:
        """Return all semantic choices at this replay boundary."""

    def apply(self, state: GameState, action: SemanticAction, registry: CardRegistry) -> GameState:
        """Apply one logged semantic action and return its post-transition state."""

    def outcome(self, state: GameState) -> ReplayOutcome:
        """Classify a state boundary for the log's final integrity marker."""


class DefaultReplayAdapter:
    """Replay adapter for the current setup/paid-action protocol."""

    def initial_state(self, setup: SetupProvenance, registry: CardRegistry) -> GameState:
        state = build_setup_state_from_piles(
            setup.shuffled_piles,
            seed=setup.seed,
            registry=registry,
        )
        if state.setup != setup:
            raise ReplayCompatibilityError(
                "setup provenance uses an unsupported RNG/deal convention"
            )
        return state

    def decisions(self, state: GameState, registry: CardRegistry) -> tuple[Decision, ...]:
        return current_decisions(state, registry)

    def apply(self, state: GameState, action: SemanticAction, registry: CardRegistry) -> GameState:
        return apply_action(state, action, registry).state

    def outcome(self, state: GameState) -> ReplayOutcome:
        if state.phase is GamePhase.TERMINAL:
            return ReplayOutcome.TERMINAL
        if state.pending_effects:
            return ReplayOutcome.EFFECT_RESOLUTION_PENDING
        return ReplayOutcome.DECISION


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """Verified final state and summary from replaying one game log."""

    state: GameState
    transitions_replayed: int
    outcome: ReplayOutcome
    terminal_result: TerminalResult | None


def check_game_log_compatibility(log: GameLog, registry: CardRegistry | None = None) -> None:
    """Reject log/header versions or card data unsupported by this engine."""

    registry = registry or load_card_registry()
    expected_versions: tuple[tuple[str, object, object], ...] = (
        ("format", log.format, GAME_LOG_FORMAT),
        ("game-log schema", log.schema_version, GAME_LOG_SCHEMA_VERSION),
        ("engine", log.engine_version, ENGINE_VERSION),
        ("rules", log.rules_version, RULES_VERSION),
        (
            "information policy",
            log.information_policy_version,
            INFORMATION_POLICY_VERSION,
        ),
        ("state schema", log.state_schema_version, STATE_SCHEMA_VERSION),
        ("action schema", log.action_schema_version, ACTION_SCHEMA_VERSION),
        ("decision schema", log.decision_schema_version, DECISION_SCHEMA_VERSION),
        (
            "observation schema",
            log.observation_schema_version,
            OBSERVATION_SCHEMA_VERSION,
        ),
        ("terminal schema", log.terminal_schema_version, TERMINAL_SCHEMA_VERSION),
        ("setup RNG", log.setup.rng_version, SETUP_RNG_VERSION),
        ("card-data fingerprint", log.card_data_fingerprint, registry.data_fingerprint),
    )
    for name, actual, expected in expected_versions:
        if actual != expected:
            raise ReplayCompatibilityError(
                f"incompatible {name}: log has {actual!r}, engine expects {expected!r}"
            )


def _decision_for_action(
    decisions: tuple[Decision, ...], action: SemanticAction, sequence: int
) -> Decision:
    matches = tuple(
        decision for decision in decisions if decision.decision_id == action.decision_id
    )
    if len(matches) != 1:
        raise ReplayDivergenceError(
            f"transition {sequence}: decision {action.decision_id} is not pending"
        )
    return matches[0]


def replay_game_log(
    log: GameLog,
    registry: CardRegistry | None = None,
    *,
    adapter: ReplayAdapter | None = None,
) -> ReplayResult:
    """Replay every action and verify decisions, hashes, terminal data, and final markers."""

    registry = registry or load_card_registry()
    selected_adapter = adapter or DefaultReplayAdapter()
    check_game_log_compatibility(log, registry)
    state = selected_adapter.initial_state(log.setup, registry)
    actual_initial_hash = state_hash(state)
    if actual_initial_hash != log.initial_state_hash:
        raise ReplayDivergenceError(
            "initial state hash differs: "
            f"recorded {log.initial_state_hash}, reproduced {actual_initial_hash}"
        )

    for entry in log.transitions:
        decisions = selected_adapter.decisions(state, registry)
        actual_decision = _decision_for_action(decisions, entry.action, entry.sequence)
        if actual_decision != entry.decision:
            raise ReplayDivergenceError(
                f"transition {entry.sequence}: recorded decision differs from engine decision"
            )
        if entry.action not in actual_decision.legal_actions:
            raise ReplayDivergenceError(
                f"transition {entry.sequence}: recorded action is not legal"
            )
        try:
            state = selected_adapter.apply(state, entry.action, registry)
        except (InnovationEngineError, ValueError) as error:
            raise ReplayDivergenceError(
                f"transition {entry.sequence}: action application failed: {error}"
            ) from error
        actual_hash = state_hash(state)
        if actual_hash != entry.state_hash:
            raise ReplayDivergenceError(
                f"transition {entry.sequence}: state hash differs: "
                f"recorded {entry.state_hash}, reproduced {actual_hash}"
            )

    actual_final_hash = state_hash(state)
    if actual_final_hash != log.final_state_hash:
        raise ReplayDivergenceError(
            "final state hash differs: "
            f"recorded {log.final_state_hash}, reproduced {actual_final_hash}"
        )
    outcome = selected_adapter.outcome(state)
    if outcome is not log.final_outcome:
        raise ReplayDivergenceError(
            f"final outcome differs: recorded {log.final_outcome}, reproduced {outcome}"
        )
    if state.terminal_result != log.terminal_result:
        raise ReplayDivergenceError("recorded terminal result differs from replayed state")
    return ReplayResult(state, len(log.transitions), outcome, state.terminal_result)


class GameLogRecorder:
    """Record semantic submissions and post-transition hashes without invoking an agent."""

    def __init__(
        self,
        initial_state: GameState,
        registry: CardRegistry | None = None,
        *,
        adapter: ReplayAdapter | None = None,
    ) -> None:
        self._registry = registry or load_card_registry()
        self._adapter = adapter or DefaultReplayAdapter()
        if initial_state.setup.card_data_fingerprint != self._registry.data_fingerprint:
            raise ReplayRecordingError("initial state's card-data fingerprint is incompatible")
        reconstructed = self._adapter.initial_state(initial_state.setup, self._registry)
        if state_hash(reconstructed) != state_hash(initial_state):
            raise ReplayRecordingError("recorder must start at the explicit setup boundary")
        self._initial_state = initial_state
        self._state = initial_state
        self._transitions: list[LoggedTransition] = []

    @property
    def state(self) -> GameState:
        """Return the recorder's current immutable authoritative state."""

        return self._state

    def decisions(self) -> tuple[Decision, ...]:
        """Return all currently pending semantic decisions."""

        return self._adapter.decisions(self._state, self._registry)

    def submit(self, action: SemanticAction) -> GameState:
        """Apply and append one currently pending semantic action."""

        sequence = len(self._transitions) + 1
        decision = _decision_for_action(self.decisions(), action, sequence)
        if action not in decision.legal_actions:
            raise ReplayRecordingError(
                f"action is not legal for pending decision {decision.decision_id}"
            )
        try:
            new_state = self._adapter.apply(self._state, action, self._registry)
        except (InnovationEngineError, ValueError) as error:
            raise ReplayRecordingError(f"could not record action: {error}") from error
        self._transitions.append(
            LoggedTransition(sequence, decision, action, state_hash(new_state))
        )
        self._state = new_state
        return new_state

    def game_log(self) -> GameLog:
        """Freeze the recording with an explicit final state/outcome marker."""

        outcome = self._adapter.outcome(self._state)
        return GameLog(
            engine_version=ENGINE_VERSION,
            rules_version=self._initial_state.rules_version,
            information_policy_version=self._initial_state.information_policy_version,
            card_data_fingerprint=self._registry.data_fingerprint,
            setup=self._initial_state.setup,
            initial_state_hash=state_hash(self._initial_state),
            transitions=tuple(self._transitions),
            transition_count=len(self._transitions),
            final_state_hash=state_hash(self._state),
            final_outcome=outcome,
            terminal_result=self._state.terminal_result,
        )
