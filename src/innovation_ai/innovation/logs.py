"""Versioned deterministic Innovation game-log schema and file I/O."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

from innovation_ai import __version__
from innovation_ai.innovation.actions import (
    ACTION_SCHEMA_VERSION,
    DECISION_SCHEMA_VERSION,
    Decision,
    SemanticAction,
    action_payload,
    decision_payload,
)
from innovation_ai.innovation.observations import OBSERVATION_SCHEMA_VERSION
from innovation_ai.innovation.serialization import (
    JsonObject,
    JsonValue,
    SerializationError,
    action_from_payload,
    canonical_json,
    decision_from_payload,
    parse_json,
    setup_from_payload,
    setup_payload,
    terminal_from_payload,
    terminal_payload,
)
from innovation_ai.innovation.state import (
    STATE_SCHEMA_VERSION,
    TERMINAL_SCHEMA_VERSION,
    SetupProvenance,
    TerminalResult,
)

GAME_LOG_FORMAT = "innovation-ai-game-log"
GAME_LOG_SCHEMA_VERSION = 2
ENGINE_VERSION = __version__


class ReplayOutcome(StrEnum):
    """The engine boundary present after the final logged transition.

    There is no "effect resolution pending" boundary: a paused dogma action always exposes a
    decision, so it is an ordinary ``DECISION`` boundary.
    """

    DECISION = "decision"
    TERMINAL = "terminal"


class GameLogError(ValueError):
    """A game-log document is malformed or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class LoggedTransition:
    """One submitted decision/action and its authoritative post-state hash."""

    sequence: int
    decision: Decision
    action: SemanticAction
    state_hash: str

    def __post_init__(self) -> None:
        if self.sequence < 1:
            raise ValueError("logged transition sequence must be positive")
        if self.action.decision_id != self.decision.decision_id:
            raise ValueError("logged action and decision IDs differ")
        if not self.state_hash.startswith("sha256:"):
            raise ValueError("logged state hash must be a tagged SHA-256 digest")


@dataclass(frozen=True, slots=True)
class GameLog:
    """Portable action log with explicit setup provenance and integrity markers."""

    engine_version: str
    rules_version: str
    information_policy_version: str
    card_data_fingerprint: str
    effects_fingerprint: str
    setup: SetupProvenance
    initial_state_hash: str
    transitions: tuple[LoggedTransition, ...]
    transition_count: int
    final_state_hash: str
    final_outcome: ReplayOutcome
    terminal_result: TerminalResult | None
    format: str = GAME_LOG_FORMAT
    schema_version: int = GAME_LOG_SCHEMA_VERSION
    state_schema_version: int = STATE_SCHEMA_VERSION
    action_schema_version: int = ACTION_SCHEMA_VERSION
    decision_schema_version: int = DECISION_SCHEMA_VERSION
    observation_schema_version: int = OBSERVATION_SCHEMA_VERSION
    terminal_schema_version: int = TERMINAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.format != GAME_LOG_FORMAT:
            raise ValueError(f"unsupported game-log format {self.format!r}")
        if self.schema_version != GAME_LOG_SCHEMA_VERSION:
            raise ValueError(f"unsupported game-log schema version {self.schema_version}")
        if self.transition_count != len(self.transitions):
            raise ValueError("game-log transition count does not match its contents")
        expected_sequences = tuple(range(1, self.transition_count + 1))
        if tuple(item.sequence for item in self.transitions) != expected_sequences:
            raise ValueError("game-log transition sequence is not contiguous")
        if self.setup.card_data_fingerprint != self.card_data_fingerprint:
            raise ValueError("game-log setup and header fingerprints differ")
        if not self.initial_state_hash.startswith("sha256:"):
            raise ValueError("initial state hash must be a tagged SHA-256 digest")
        if not self.final_state_hash.startswith("sha256:"):
            raise ValueError("final state hash must be a tagged SHA-256 digest")
        if (self.final_outcome is ReplayOutcome.TERMINAL) != (self.terminal_result is not None):
            raise ValueError("terminal outcome and terminal result must be supplied together")


def game_log_payload(log: GameLog) -> JsonObject:
    """Return the canonical JSON-compatible game-log payload."""

    return {
        "format": log.format,
        "schema_version": log.schema_version,
        "engine_version": log.engine_version,
        "rules_version": log.rules_version,
        "information_policy_version": log.information_policy_version,
        "card_data_fingerprint": log.card_data_fingerprint,
        "effects_fingerprint": log.effects_fingerprint,
        "state_schema_version": log.state_schema_version,
        "action_schema_version": log.action_schema_version,
        "decision_schema_version": log.decision_schema_version,
        "observation_schema_version": log.observation_schema_version,
        "terminal_schema_version": log.terminal_schema_version,
        "setup": cast(JsonValue, setup_payload(log.setup)),
        "initial_state_hash": log.initial_state_hash,
        "transitions": [
            {
                "sequence": transition.sequence,
                "decision": cast(JsonValue, decision_payload(transition.decision)),
                "action": cast(JsonValue, action_payload(transition.action)),
                "state_hash": transition.state_hash,
            }
            for transition in log.transitions
        ],
        "transition_count": log.transition_count,
        "final_state_hash": log.final_state_hash,
        "final_outcome": log.final_outcome.value,
        "terminal_result": (
            None
            if log.terminal_result is None
            else cast(JsonValue, terminal_payload(log.terminal_result))
        ),
    }


def dumps_game_log(log: GameLog) -> str:
    """Serialize a game log to deterministic single-line JSON."""

    return canonical_json(cast(JsonValue, game_log_payload(log)))


def _object(value: object, path: str) -> JsonObject:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise GameLogError(f"{path} must be an object")
    return cast(JsonObject, value)


def _array(value: JsonValue, path: str) -> list[JsonValue]:
    if not isinstance(value, list):
        raise GameLogError(f"{path} must be an array")
    return value


def _string(value: JsonValue, path: str) -> str:
    if not isinstance(value, str):
        raise GameLogError(f"{path} must be a string")
    return value


def _integer(value: JsonValue, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GameLogError(f"{path} must be an integer")
    return value


def _exact_keys(payload: JsonObject, expected: set[str], path: str) -> None:
    missing = expected - payload.keys()
    extra = payload.keys() - expected
    if missing or extra:
        raise GameLogError(
            f"{path} keys differ: missing={sorted(missing)}, unexpected={sorted(extra)}"
        )


def game_log_from_payload(value: object) -> GameLog:
    """Decode a strict game-log payload without replaying it."""

    payload = _object(value, "game_log")
    expected = {
        "format",
        "schema_version",
        "engine_version",
        "rules_version",
        "information_policy_version",
        "card_data_fingerprint",
        "effects_fingerprint",
        "state_schema_version",
        "action_schema_version",
        "decision_schema_version",
        "observation_schema_version",
        "terminal_schema_version",
        "setup",
        "initial_state_hash",
        "transitions",
        "transition_count",
        "final_state_hash",
        "final_outcome",
        "terminal_result",
    }
    _exact_keys(payload, expected, "game_log")
    transitions: list[LoggedTransition] = []
    try:
        for raw_item in _array(payload["transitions"], "game_log.transitions"):
            item = _object(raw_item, "game_log.transitions[]")
            _exact_keys(
                item,
                {"sequence", "decision", "action", "state_hash"},
                "game_log.transitions[]",
            )
            transitions.append(
                LoggedTransition(
                    _integer(item["sequence"], "game_log.transitions[].sequence"),
                    decision_from_payload(item["decision"]),
                    action_from_payload(item["action"]),
                    _string(item["state_hash"], "game_log.transitions[].state_hash"),
                )
            )
        terminal_value = payload["terminal_result"]
        return GameLog(
            engine_version=_string(payload["engine_version"], "game_log.engine_version"),
            rules_version=_string(payload["rules_version"], "game_log.rules_version"),
            information_policy_version=_string(
                payload["information_policy_version"], "game_log.information_policy_version"
            ),
            card_data_fingerprint=_string(
                payload["card_data_fingerprint"], "game_log.card_data_fingerprint"
            ),
            effects_fingerprint=_string(
                payload["effects_fingerprint"], "game_log.effects_fingerprint"
            ),
            setup=setup_from_payload(payload["setup"]),
            initial_state_hash=_string(
                payload["initial_state_hash"], "game_log.initial_state_hash"
            ),
            transitions=tuple(transitions),
            transition_count=_integer(payload["transition_count"], "game_log.transition_count"),
            final_state_hash=_string(payload["final_state_hash"], "game_log.final_state_hash"),
            final_outcome=ReplayOutcome(
                _string(payload["final_outcome"], "game_log.final_outcome")
            ),
            terminal_result=(
                None if terminal_value is None else terminal_from_payload(terminal_value)
            ),
            format=_string(payload["format"], "game_log.format"),
            schema_version=_integer(payload["schema_version"], "game_log.schema_version"),
            state_schema_version=_integer(
                payload["state_schema_version"], "game_log.state_schema_version"
            ),
            action_schema_version=_integer(
                payload["action_schema_version"], "game_log.action_schema_version"
            ),
            decision_schema_version=_integer(
                payload["decision_schema_version"], "game_log.decision_schema_version"
            ),
            observation_schema_version=_integer(
                payload["observation_schema_version"], "game_log.observation_schema_version"
            ),
            terminal_schema_version=_integer(
                payload["terminal_schema_version"], "game_log.terminal_schema_version"
            ),
        )
    except (SerializationError, ValueError) as error:
        if isinstance(error, GameLogError):
            raise
        raise GameLogError(f"invalid game log: {error}") from error


def loads_game_log(text: str) -> GameLog:
    """Deserialize a game-log JSON document."""

    try:
        return game_log_from_payload(parse_json(text))
    except SerializationError as error:
        raise GameLogError(str(error)) from error


def save_game_log(log: GameLog, path: str | Path) -> None:
    """Write a deterministic UTF-8 game log, ending with one newline."""

    Path(path).write_text(f"{dumps_game_log(log)}\n", encoding="utf-8")


def load_game_log(path: str | Path) -> GameLog:
    """Read and decode a UTF-8 game log."""

    return loads_game_log(Path(path).read_text(encoding="utf-8"))
