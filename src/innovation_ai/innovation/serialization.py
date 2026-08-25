"""Strict deterministic JSON schemas for Innovation engine contracts."""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from enum import StrEnum
from typing import cast

from innovation_ai.innovation.actions import (
    ACTION_SCHEMA_VERSION,
    DECISION_SCHEMA_VERSION,
    AchieveAction,
    ActionKind,
    ChooseBranchAction,
    ChooseCardAction,
    ChooseCardsAction,
    ChooseColorAction,
    ChoosePlayerAction,
    ChooseSplayAction,
    ChooseStartingMeldAction,
    ChooseValueAction,
    Decision,
    DecisionContext,
    DecisionKind,
    DecisionSource,
    DeclineAction,
    DogmaAction,
    DrawAction,
    FinishSelectionAction,
    IncrementalSelectionKind,
    MeldAction,
    OrderCardsAction,
    SemanticAction,
    action_payload,
    decision_payload,
)
from innovation_ai.innovation.catalog import CardRegistry, load_card_registry
from innovation_ai.innovation.observations import (
    OBSERVATION_SCHEMA_VERSION,
    CoveredCardObservation,
    GameObservation,
    InformationPolicy,
    PlayerObservation,
    StackObservation,
    SupplyObservation,
    ZoneObservation,
)
from innovation_ai.innovation.state import (
    INFORMATION_POLICY_VERSION,
    RULES_VERSION,
    STATE_SCHEMA_VERSION,
    TERMINAL_SCHEMA_VERSION,
    Board,
    ColorStack,
    EffectFrameState,
    EffectVariable,
    GamePhase,
    GameState,
    NormalAchievementState,
    PlayerState,
    PlayerTurnCounters,
    RevealedCard,
    SetupProvenance,
    StateValue,
    SupplyState,
    TerminalReason,
    TerminalResult,
    TurnCounters,
    state_payload,
)
from innovation_ai.innovation.types import (
    CardId,
    Color,
    DogmaEffectId,
    Icon,
    NormalAchievementId,
    PlayerId,
    SpecialAchievementId,
    SplayDirection,
)
from innovation_ai.innovation.zones import assert_state_invariants

JsonScalar = str | int | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]


class SerializationError(ValueError):
    """A serialized contract is malformed or unsupported."""


class SchemaVersionError(SerializationError):
    """A serialized contract uses an unsupported schema version."""


def _object(value: object, path: str) -> JsonObject:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise SerializationError(f"{path} must be an object")
    return cast(JsonObject, value)


def _list(value: JsonValue, path: str) -> list[JsonValue]:
    if not isinstance(value, list):
        raise SerializationError(f"{path} must be an array")
    return value


def _string(value: JsonValue, path: str) -> str:
    if not isinstance(value, str):
        raise SerializationError(f"{path} must be a string")
    return value


def _integer(value: JsonValue, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SerializationError(f"{path} must be an integer")
    return value


def _optional_integer(value: JsonValue, path: str) -> int | None:
    return None if value is None else _integer(value, path)


def _boolean(value: JsonValue, path: str) -> bool:
    if not isinstance(value, bool):
        raise SerializationError(f"{path} must be a boolean")
    return value


def _optional_string(value: JsonValue, path: str) -> str | None:
    return None if value is None else _string(value, path)


def _keys(
    payload: JsonObject,
    required: set[str],
    path: str,
    *,
    optional: set[str] | None = None,
) -> None:
    allowed = required | (optional or set())
    missing = required - payload.keys()
    extra = payload.keys() - allowed
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if extra:
            details.append(f"unexpected {sorted(extra)}")
        raise SerializationError(f"{path} has {', '.join(details)}")


def _schema(payload: JsonObject, expected: int, path: str) -> None:
    actual = _integer(payload.get("schema_version"), f"{path}.schema_version")
    if actual != expected:
        raise SchemaVersionError(f"unsupported {path} schema version {actual}; expected {expected}")


def _enum[EnumT: StrEnum](enum_type: type[EnumT], value: JsonValue, path: str) -> EnumT:
    raw = _string(value, path)
    try:
        return enum_type(raw)
    except ValueError as error:
        raise SerializationError(f"{path} has unknown value {raw!r}") from error


def _cards(value: JsonValue, path: str) -> tuple[CardId, ...]:
    return tuple(CardId(_string(item, f"{path}[]")) for item in _list(value, path))


def _enums[EnumT: StrEnum](
    enum_type: type[EnumT], value: JsonValue, path: str
) -> tuple[EnumT, ...]:
    return tuple(_enum(enum_type, item, f"{path}[]") for item in _list(value, path))


def canonical_json(payload: JsonValue) -> str:
    """Encode one JSON-compatible payload with stable byte-for-byte formatting."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def parse_json(text: str) -> JsonValue:
    """Parse JSON while normalizing parser errors to ``SerializationError``."""

    try:
        return cast(JsonValue, json.loads(text))
    except (json.JSONDecodeError, RecursionError) as error:
        raise SerializationError(f"invalid JSON: {error}") from error


def _contract_value(value: object) -> JsonValue:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, CardId):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _contract_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [_contract_value(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"contract contains non-serializable value: {type(value).__name__}")


def terminal_payload(terminal: TerminalResult) -> JsonObject:
    """Return the canonical versioned terminal-result payload."""

    return cast(JsonObject, _contract_value(terminal))


def observation_payload(observation: GameObservation) -> JsonObject:
    """Return the canonical versioned player-observation payload."""

    return cast(JsonObject, _contract_value(observation))


def action_from_payload(value: object) -> SemanticAction:
    """Decode a strict version-1 semantic action payload."""

    payload = _object(value, "action")
    common = {"schema_version", "kind", "decision_id"}
    kind = _enum(ActionKind, payload.get("kind"), "action.kind")
    extra_by_kind = {
        ActionKind.CHOOSE_STARTING_MELD: {"card_id"},
        ActionKind.DRAW: set(),
        ActionKind.MELD: {"card_id"},
        ActionKind.DOGMA: {"card_id"},
        ActionKind.ACHIEVE: {"achievement_id"},
        ActionKind.CHOOSE_CARD: {"card_id"},
        ActionKind.CHOOSE_CARDS: {"card_ids"},
        ActionKind.CHOOSE_COLOR: {"color"},
        ActionKind.CHOOSE_PLAYER: {"player_id"},
        ActionKind.CHOOSE_VALUE: {"value"},
        ActionKind.CHOOSE_SPLAY: {"direction"},
        ActionKind.CHOOSE_BRANCH: {"branch_id"},
        ActionKind.ORDER_CARDS: {"card_ids"},
        ActionKind.DECLINE: set(),
        ActionKind.FINISH_SELECTION: set(),
    }
    _keys(payload, common | extra_by_kind[kind], "action")
    _schema(payload, ACTION_SCHEMA_VERSION, "action")
    decision_id = _integer(payload["decision_id"], "action.decision_id")
    if kind is ActionKind.CHOOSE_STARTING_MELD:
        return ChooseStartingMeldAction(
            decision_id, CardId(_string(payload["card_id"], "action.card_id"))
        )
    if kind is ActionKind.DRAW:
        return DrawAction(decision_id)
    if kind is ActionKind.MELD:
        return MeldAction(decision_id, CardId(_string(payload["card_id"], "action.card_id")))
    if kind is ActionKind.DOGMA:
        return DogmaAction(decision_id, CardId(_string(payload["card_id"], "action.card_id")))
    if kind is ActionKind.ACHIEVE:
        return AchieveAction(
            decision_id,
            _enum(NormalAchievementId, payload["achievement_id"], "action.achievement_id"),
        )
    if kind is ActionKind.CHOOSE_CARD:
        return ChooseCardAction(decision_id, CardId(_string(payload["card_id"], "action.card_id")))
    if kind is ActionKind.CHOOSE_CARDS:
        return ChooseCardsAction(decision_id, _cards(payload["card_ids"], "action.card_ids"))
    if kind is ActionKind.CHOOSE_COLOR:
        return ChooseColorAction(decision_id, _enum(Color, payload["color"], "action.color"))
    if kind is ActionKind.CHOOSE_PLAYER:
        return ChoosePlayerAction(
            decision_id, _enum(PlayerId, payload["player_id"], "action.player_id")
        )
    if kind is ActionKind.CHOOSE_VALUE:
        return ChooseValueAction(decision_id, _integer(payload["value"], "action.value"))
    if kind is ActionKind.CHOOSE_SPLAY:
        return ChooseSplayAction(
            decision_id, _enum(SplayDirection, payload["direction"], "action.direction")
        )
    if kind is ActionKind.CHOOSE_BRANCH:
        return ChooseBranchAction(decision_id, _string(payload["branch_id"], "action.branch_id"))
    if kind is ActionKind.ORDER_CARDS:
        return OrderCardsAction(decision_id, _cards(payload["card_ids"], "action.card_ids"))
    if kind is ActionKind.DECLINE:
        return DeclineAction(decision_id)
    return FinishSelectionAction(decision_id)


def _zone_observation(value: JsonValue, path: str) -> ZoneObservation:
    payload = _object(value, path)
    _keys(payload, {"values", "known_cards"}, path)
    values = tuple(_integer(item, f"{path}.values[]") for item in _list(payload["values"], path))
    return ZoneObservation(values, _cards(payload["known_cards"], f"{path}.known_cards"))


def _covered_observation(value: JsonValue, path: str) -> CoveredCardObservation:
    payload = _object(value, path)
    _keys(payload, {"card_id", "age", "visible_icons"}, path)
    card = _optional_string(payload["card_id"], f"{path}.card_id")
    return CoveredCardObservation(
        CardId(card) if card is not None else None,
        _optional_integer(payload["age"], f"{path}.age"),
        _enums(Icon, payload["visible_icons"], f"{path}.visible_icons"),
    )


def _stack_observation(value: JsonValue, path: str) -> StackObservation:
    payload = _object(value, path)
    _keys(payload, {"color", "top_card_id", "splay", "covered_cards", "covered_count"}, path)
    top = _optional_string(payload["top_card_id"], f"{path}.top_card_id")
    return StackObservation(
        _enum(Color, payload["color"], f"{path}.color"),
        CardId(top) if top is not None else None,
        _enum(SplayDirection, payload["splay"], f"{path}.splay"),
        tuple(
            _covered_observation(item, f"{path}.covered_cards[]")
            for item in _list(payload["covered_cards"], f"{path}.covered_cards")
        ),
        _optional_integer(payload["covered_count"], f"{path}.covered_count"),
    )


def _player_observation(value: JsonValue, path: str) -> PlayerObservation:
    payload = _object(value, path)
    _keys(
        payload,
        {
            "player_id",
            "hand",
            "score_pile",
            "board",
            "normal_achievements",
            "special_achievements",
        },
        path,
    )
    board = tuple(
        _stack_observation(item, f"{path}.board[]")
        for item in _list(payload["board"], f"{path}.board")
    )
    return PlayerObservation(
        _enum(PlayerId, payload["player_id"], f"{path}.player_id"),
        _zone_observation(payload["hand"], f"{path}.hand"),
        _zone_observation(payload["score_pile"], f"{path}.score_pile"),
        board,
        _enums(
            NormalAchievementId,
            payload["normal_achievements"],
            f"{path}.normal_achievements",
        ),
        _enums(
            SpecialAchievementId,
            payload["special_achievements"],
            f"{path}.special_achievements",
        ),
    )


def observation_from_payload(value: object) -> GameObservation:
    """Decode a strict version-1 player observation payload."""

    payload = _object(value, "observation")
    _keys(
        payload,
        {
            "viewer",
            "phase",
            "active_player",
            "turn_number",
            "paid_actions_remaining",
            "supplies",
            "players",
            "available_normal_achievements",
            "available_special_achievements",
            "information_policy",
            "rules_version",
            "revealed_cards",
            "schema_version",
        },
        "observation",
    )
    _schema(payload, OBSERVATION_SCHEMA_VERSION, "observation")
    active = _optional_string(payload["active_player"], "observation.active_player")
    supplies: list[SupplyObservation] = []
    for item in _list(payload["supplies"], "observation.supplies"):
        supply = _object(item, "observation.supplies[]")
        _keys(supply, {"age", "count"}, "observation.supplies[]")
        supplies.append(
            SupplyObservation(
                _integer(supply["age"], "observation.supplies[].age"),
                _integer(supply["count"], "observation.supplies[].count"),
            )
        )
    players = tuple(
        _player_observation(item, "observation.players[]")
        for item in _list(payload["players"], "observation.players")
    )
    if len(players) != 2:
        raise SerializationError("observation.players must contain exactly two players")
    return GameObservation(
        viewer=_enum(PlayerId, payload["viewer"], "observation.viewer"),
        phase=_enum(GamePhase, payload["phase"], "observation.phase"),
        active_player=PlayerId(active) if active is not None else None,
        turn_number=_integer(payload["turn_number"], "observation.turn_number"),
        paid_actions_remaining=_integer(
            payload["paid_actions_remaining"], "observation.paid_actions_remaining"
        ),
        supplies=tuple(supplies),
        players=players,
        available_normal_achievements=_enums(
            NormalAchievementId,
            payload["available_normal_achievements"],
            "observation.available_normal_achievements",
        ),
        available_special_achievements=_enums(
            SpecialAchievementId,
            payload["available_special_achievements"],
            "observation.available_special_achievements",
        ),
        information_policy=_enum(
            InformationPolicy, payload["information_policy"], "observation.information_policy"
        ),
        rules_version=_string(payload["rules_version"], "observation.rules_version"),
        revealed_cards=_cards(payload["revealed_cards"], "observation.revealed_cards"),
    )


def _decision_source(value: JsonValue) -> DecisionSource | None:
    if value is None:
        return None
    payload = _object(value, "decision.source")
    _keys(payload, {"card_id", "effect_id"}, "decision.source")
    card_id = CardId(_string(payload["card_id"], "decision.source.card_id"))
    raw_effect = payload["effect_id"]
    if raw_effect is None:
        effect_id = None
    else:
        effect = _object(raw_effect, "decision.source.effect_id")
        _keys(effect, {"card_id", "ordinal"}, "decision.source.effect_id")
        effect_id = DogmaEffectId(
            CardId(_string(effect["card_id"], "decision.source.effect_id.card_id")),
            _integer(effect["ordinal"], "decision.source.effect_id.ordinal"),
        )
    return DecisionSource(card_id, effect_id)


def decision_from_payload(value: object) -> Decision:
    """Decode a strict version-1 decision, including its detached observation."""

    payload = _object(value, "decision")
    _keys(
        payload,
        {
            "schema_version",
            "decision_id",
            "kind",
            "chooser",
            "executor",
            "source",
            "dogma_activator",
            "dogma_action_id",
            "context",
            "observation",
            "legal_actions",
        },
        "decision",
    )
    _schema(payload, DECISION_SCHEMA_VERSION, "decision")
    dogma_activator = _optional_string(payload["dogma_activator"], "decision.dogma_activator")
    return Decision(
        decision_id=_integer(payload["decision_id"], "decision.decision_id"),
        kind=_enum(DecisionKind, payload["kind"], "decision.kind"),
        chooser=_enum(PlayerId, payload["chooser"], "decision.chooser"),
        executor=_enum(PlayerId, payload["executor"], "decision.executor"),
        source=_decision_source(payload["source"]),
        dogma_activator=PlayerId(dogma_activator) if dogma_activator is not None else None,
        dogma_action_id=_optional_integer(payload["dogma_action_id"], "decision.dogma_action_id"),
        context=_decision_context(payload["context"]),
        observation=observation_from_payload(payload["observation"]),
        legal_actions=tuple(
            action_from_payload(item)
            for item in _list(payload["legal_actions"], "decision.legal_actions")
        ),
    )


def _decision_context(value: JsonValue) -> DecisionContext | None:
    if value is None:
        return None
    payload = _object(value, "decision.context")
    _keys(
        payload,
        {
            "demand",
            "shared",
            "nested",
            "featured_icon",
            "activator_icons",
            "opponent_icons",
            "minimum_count",
            "maximum_count",
            "selected_so_far",
            "incremental_selection",
        },
        "decision.context",
    )
    icon = payload["featured_icon"]
    return DecisionContext(
        demand=_boolean(payload["demand"], "decision.context.demand"),
        shared=_boolean(payload["shared"], "decision.context.shared"),
        nested=_boolean(payload["nested"], "decision.context.nested"),
        featured_icon=None if icon is None else _enum(Icon, icon, "decision.context.featured_icon"),
        activator_icons=_optional_integer(
            payload["activator_icons"], "decision.context.activator_icons"
        ),
        opponent_icons=_optional_integer(
            payload["opponent_icons"], "decision.context.opponent_icons"
        ),
        minimum_count=_integer(payload["minimum_count"], "decision.context.minimum_count"),
        maximum_count=_integer(payload["maximum_count"], "decision.context.maximum_count"),
        selected_so_far=_cards(payload["selected_so_far"], "decision.context.selected_so_far"),
        incremental_selection=_enum(
            IncrementalSelectionKind,
            payload["incremental_selection"],
            "decision.context.incremental_selection",
        ),
    )


def terminal_from_payload(value: object) -> TerminalResult:
    """Decode a strict version-1 terminal-result payload."""

    payload = _object(value, "terminal")
    _keys(payload, {"reason", "winners", "schema_version"}, "terminal")
    _schema(payload, TERMINAL_SCHEMA_VERSION, "terminal")
    return TerminalResult(
        _enum(TerminalReason, payload["reason"], "terminal.reason"),
        _enums(PlayerId, payload["winners"], "terminal.winners"),
    )


def _effect_value(value: JsonValue, path: str) -> StateValue:
    if isinstance(value, list):
        return tuple(_effect_value(item, f"{path}[]") for item in value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise SerializationError(f"{path} must contain only scalar values or arrays")


def _effect_variable(value: JsonValue, path: str) -> EffectVariable:
    payload = _object(value, path)
    _keys(payload, {"name", "value"}, path)
    raw = _effect_value(payload["value"], f"{path}.value")
    return EffectVariable(_string(payload["name"], f"{path}.name"), raw)


def _color_stack(value: JsonValue, path: str) -> ColorStack:
    payload = _object(value, path)
    _keys(payload, {"color", "cards", "splay"}, path)
    return ColorStack(
        _enum(Color, payload["color"], f"{path}.color"),
        _cards(payload["cards"], f"{path}.cards"),
        _enum(SplayDirection, payload["splay"], f"{path}.splay"),
    )


def _player_state(value: JsonValue, path: str) -> PlayerState:
    payload = _object(value, path)
    _keys(
        payload,
        {
            "player_id",
            "hand",
            "board",
            "score_pile",
            "normal_achievements",
            "special_achievements",
        },
        path,
    )
    board_payload = _object(payload["board"], f"{path}.board")
    _keys(board_payload, {"stacks"}, f"{path}.board")
    stacks = tuple(
        _color_stack(item, f"{path}.board.stacks[]")
        for item in _list(board_payload["stacks"], f"{path}.board.stacks")
    )
    return PlayerState(
        _enum(PlayerId, payload["player_id"], f"{path}.player_id"),
        _cards(payload["hand"], f"{path}.hand"),
        Board(stacks),
        _cards(payload["score_pile"], f"{path}.score_pile"),
        _enums(
            NormalAchievementId,
            payload["normal_achievements"],
            f"{path}.normal_achievements",
        ),
        _enums(
            SpecialAchievementId,
            payload["special_achievements"],
            f"{path}.special_achievements",
        ),
    )


def _setup(value: JsonValue, path: str = "setup") -> SetupProvenance:
    payload = _object(value, path)
    _keys(
        payload,
        {"seed", "card_data_fingerprint", "shuffled_piles", "deal_sequence", "rng_version"},
        path,
    )
    return SetupProvenance(
        seed=_integer(payload["seed"], f"{path}.seed"),
        card_data_fingerprint=_string(
            payload["card_data_fingerprint"], f"{path}.card_data_fingerprint"
        ),
        shuffled_piles=tuple(
            _cards(item, f"{path}.shuffled_piles[]")
            for item in _list(payload["shuffled_piles"], f"{path}.shuffled_piles")
        ),
        deal_sequence=_enums(PlayerId, payload["deal_sequence"], f"{path}.deal_sequence"),
        rng_version=_string(payload["rng_version"], f"{path}.rng_version"),
    )


def setup_payload(setup: SetupProvenance) -> JsonObject:
    """Return setup provenance as a canonical JSON-compatible object."""

    return cast(JsonObject, _contract_value(setup))


def setup_from_payload(value: object) -> SetupProvenance:
    """Decode explicit setup provenance embedded in a state or game log."""

    return _setup(cast(JsonValue, value))


def state_from_payload(
    value: object,
    registry: CardRegistry | None = None,
    *,
    check_compatibility: bool = True,
) -> GameState:
    """Decode and validate a complete authoritative state payload."""

    payload = _object(value, "state")
    _keys(
        payload,
        {
            "supply",
            "players",
            "normal_achievements",
            "removed_cards",
            "phase",
            "active_player",
            "turn_number",
            "paid_actions_remaining",
            "turn_counters",
            "pending_effects",
            "effect_variables",
            "revealed",
            "starting_meld_decision_ids",
            "starting_meld_choices",
            "next_decision_id",
            "next_event_id",
            "next_dogma_action_id",
            "setup",
            "terminal_result",
            "schema_version",
            "rules_version",
            "information_policy_version",
        },
        "state",
    )
    _schema(payload, STATE_SCHEMA_VERSION, "state")
    supply_payload = _object(payload["supply"], "state.supply")
    _keys(supply_payload, {"piles"}, "state.supply")
    supply = SupplyState(
        tuple(
            _cards(item, "state.supply.piles[]")
            for item in _list(supply_payload["piles"], "state.supply.piles")
        )
    )
    players = tuple(
        _player_state(item, "state.players[]")
        for item in _list(payload["players"], "state.players")
    )
    if len(players) != 2:
        raise SerializationError("state.players must contain exactly two players")
    achievements_payload = _object(payload["normal_achievements"], "state.normal_achievements")
    _keys(achievements_payload, {"cards"}, "state.normal_achievements")
    counters_payload = _object(payload["turn_counters"], "state.turn_counters")
    _keys(counters_payload, {"players"}, "state.turn_counters")
    counters: list[PlayerTurnCounters] = []
    for item in _list(counters_payload["players"], "state.turn_counters.players"):
        counter = _object(item, "state.turn_counters.players[]")
        _keys(counter, {"player_id", "tucked", "scored"}, "state.turn_counters.players[]")
        counters.append(
            PlayerTurnCounters(
                _enum(PlayerId, counter["player_id"], "state.turn_counters.players[].player_id"),
                _integer(counter["tucked"], "state.turn_counters.players[].tucked"),
                _integer(counter["scored"], "state.turn_counters.players[].scored"),
            )
        )
    if len(counters) != 2:
        raise SerializationError("state.turn_counters.players must contain exactly two players")
    pending_effects: list[EffectFrameState] = []
    for item in _list(payload["pending_effects"], "state.pending_effects"):
        frame = _object(item, "state.pending_effects[]")
        _keys(frame, {"kind", "step", "source_card_id", "variables"}, "state.pending_effects[]")
        source = _optional_string(frame["source_card_id"], "state.pending_effects[].source_card_id")
        pending_effects.append(
            EffectFrameState(
                _string(frame["kind"], "state.pending_effects[].kind"),
                _integer(frame["step"], "state.pending_effects[].step"),
                CardId(source) if source is not None else None,
                tuple(
                    _effect_variable(variable, "state.pending_effects[].variables[]")
                    for variable in _list(frame["variables"], "state.pending_effects[].variables")
                ),
            )
        )
    revealed: list[RevealedCard] = []
    for item in _list(payload["revealed"], "state.revealed"):
        marker = _object(item, "state.revealed[]")
        _keys(marker, {"card_id", "scope"}, "state.revealed[]")
        revealed.append(
            RevealedCard(
                CardId(_string(marker["card_id"], "state.revealed[].card_id")),
                _string(marker["scope"], "state.revealed[].scope"),
            )
        )
    choices: list[CardId | None] = []
    for item in _list(payload["starting_meld_choices"], "state.starting_meld_choices"):
        raw = _optional_string(item, "state.starting_meld_choices[]")
        choices.append(CardId(raw) if raw is not None else None)
    if len(choices) != 2:
        raise SerializationError("state.starting_meld_choices must contain exactly two values")
    terminal_value = payload["terminal_result"]
    active = _optional_string(payload["active_player"], "state.active_player")
    state = GameState(
        supply=supply,
        players=players,
        normal_achievements=NormalAchievementState(
            _cards(achievements_payload["cards"], "state.normal_achievements.cards")
        ),
        removed_cards=_cards(payload["removed_cards"], "state.removed_cards"),
        phase=_enum(GamePhase, payload["phase"], "state.phase"),
        active_player=PlayerId(active) if active is not None else None,
        turn_number=_integer(payload["turn_number"], "state.turn_number"),
        paid_actions_remaining=_integer(
            payload["paid_actions_remaining"], "state.paid_actions_remaining"
        ),
        turn_counters=TurnCounters(
            cast(tuple[PlayerTurnCounters, PlayerTurnCounters], tuple(counters))
        ),
        pending_effects=tuple(pending_effects),
        effect_variables=tuple(
            _effect_variable(item, "state.effect_variables[]")
            for item in _list(payload["effect_variables"], "state.effect_variables")
        ),
        revealed=tuple(revealed),
        starting_meld_decision_ids=cast(
            tuple[int, int],
            tuple(
                _integer(item, "state.starting_meld_decision_ids[]")
                for item in _list(
                    payload["starting_meld_decision_ids"], "state.starting_meld_decision_ids"
                )
            ),
        ),
        starting_meld_choices=(choices[0], choices[1]),
        next_decision_id=_integer(payload["next_decision_id"], "state.next_decision_id"),
        next_event_id=_integer(payload["next_event_id"], "state.next_event_id"),
        next_dogma_action_id=_integer(
            payload["next_dogma_action_id"], "state.next_dogma_action_id"
        ),
        setup=_setup(payload["setup"], "state.setup"),
        terminal_result=None if terminal_value is None else terminal_from_payload(terminal_value),
        rules_version=_string(payload["rules_version"], "state.rules_version"),
        information_policy_version=_string(
            payload["information_policy_version"], "state.information_policy_version"
        ),
    )
    from innovation_ai.innovation.effects.model import validate_effect_runtime_structure

    try:
        validate_effect_runtime_structure(state)
    except (ValueError, RuntimeError) as error:
        raise SerializationError(f"invalid effect runtime: {error}") from error
    if check_compatibility:
        registry = registry or load_card_registry()
        if state.rules_version != RULES_VERSION:
            raise SchemaVersionError(f"unsupported rules version {state.rules_version!r}")
        if state.information_policy_version != INFORMATION_POLICY_VERSION:
            raise SchemaVersionError(
                f"unsupported information-policy version {state.information_policy_version!r}"
            )
        if state.setup.card_data_fingerprint != registry.data_fingerprint:
            raise SchemaVersionError("state card-data fingerprint is incompatible")
        try:
            assert_state_invariants(state, registry)
        except RuntimeError as error:
            raise SerializationError(f"state invariant validation failed: {error}") from error
    return state


def dumps_state(state: GameState) -> str:
    """Serialize a state to canonical JSON."""

    return canonical_json(cast(JsonValue, state_payload(state)))


def loads_state(text: str, registry: CardRegistry | None = None) -> GameState:
    """Deserialize a canonical state JSON document."""

    return state_from_payload(parse_json(text), registry)


def dumps_action(action: SemanticAction) -> str:
    """Serialize a semantic action to canonical JSON."""

    return canonical_json(cast(JsonValue, action_payload(action)))


def loads_action(text: str) -> SemanticAction:
    """Deserialize a semantic action JSON document."""

    return action_from_payload(parse_json(text))


def dumps_decision(decision: Decision) -> str:
    """Serialize a decision to canonical JSON."""

    return canonical_json(cast(JsonValue, decision_payload(decision)))


def loads_decision(text: str) -> Decision:
    """Deserialize a decision JSON document."""

    return decision_from_payload(parse_json(text))


def dumps_terminal(terminal: TerminalResult) -> str:
    """Serialize a terminal result to canonical JSON."""

    return canonical_json(terminal_payload(terminal))


def loads_terminal(text: str) -> TerminalResult:
    """Deserialize a terminal-result JSON document."""

    return terminal_from_payload(parse_json(text))
