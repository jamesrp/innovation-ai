"""Versioned flat encoding of player-safe value positions."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, replace
from functools import cached_property
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from innovation_ai.harness.policy import (
    PUBLIC_BOUNDARY_SCHEMA_VERSION,
    VALUE_POSITION_SCHEMA_VERSION,
    PlayerRelation,
    ValuePosition,
    ValuePositionKind,
)
from innovation_ai.innovation.actions import (
    DECISION_SCHEMA_VERSION,
    DecisionKind,
    IncrementalSelectionKind,
)
from innovation_ai.innovation.catalog import CardRegistry, load_card_registry
from innovation_ai.innovation.observations import (
    OBSERVATION_SCHEMA_VERSION,
    StackObservation,
    ZoneObservation,
)
from innovation_ai.innovation.state import (
    INFORMATION_POLICY_VERSION,
    RULES_VERSION,
    SUPPORTED_INFORMATION_POLICY_VERSIONS,
    GamePhase,
)
from innovation_ai.innovation.types import (
    CardId,
    Color,
    Icon,
    NormalAchievementId,
    PlayerId,
    SpecialAchievementId,
    SplayDirection,
)

ENCODER_VERSION = "flat-observation-v1"
ENCODER_MANIFEST_SCHEMA_VERSION = 1
TURN_NUMBER_SCALE = 100.0
CARD_COUNT_SCALE = 15.0
STACK_COUNT_SCALE = 21.0
GENERAL_COUNT_SCALE = 105.0
ICON_COUNT_SCALE = 21.0


class EncoderCompatibilityError(ValueError):
    """A position or manifest is incompatible with this encoder."""


class _Setter(Protocol):
    def __call__(self, name: str, value: float = 1.0) -> None: ...


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """One scalar feature at a stable vector offset."""

    name: str
    offset: int
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not self.name or self.offset < 0 or self.scale <= 0.0:
            raise ValueError("invalid encoder feature specification")


@dataclass(frozen=True)
class EncoderManifest:
    """Complete frozen layout and compatibility metadata for one encoder version."""

    encoder_version: str
    card_data_fingerprint: str
    rules_version: str
    information_policy_version: str
    card_order: tuple[str, ...]
    color_order: tuple[str, ...]
    icon_order: tuple[str, ...]
    splay_order: tuple[str, ...]
    normal_achievement_order: tuple[str, ...]
    special_achievement_order: tuple[str, ...]
    decision_kind_order: tuple[str, ...]
    relation_order: tuple[str, ...]
    incremental_selection_order: tuple[str, ...]
    feature_specs: tuple[FeatureSpec, ...]
    input_dimension: int
    layout_fingerprint: str
    schema_version: int = ENCODER_MANIFEST_SCHEMA_VERSION
    observation_schema_version: int = OBSERVATION_SCHEMA_VERSION
    decision_schema_version: int = DECISION_SCHEMA_VERSION
    value_position_schema_version: int = VALUE_POSITION_SCHEMA_VERSION
    public_boundary_schema_version: int = PUBLIC_BOUNDARY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ENCODER_MANIFEST_SCHEMA_VERSION:
            raise EncoderCompatibilityError(
                f"unsupported encoder manifest schema {self.schema_version}"
            )
        if self.input_dimension != len(self.feature_specs):
            raise ValueError("encoder dimension does not match scalar feature count")
        if tuple(spec.offset for spec in self.feature_specs) != tuple(range(self.input_dimension)):
            raise ValueError("encoder feature offsets must be contiguous")
        if len({spec.name for spec in self.feature_specs}) != self.input_dimension:
            raise ValueError("encoder feature names must be unique")
        if self.layout_fingerprint and self.layout_fingerprint != _manifest_fingerprint(
            replace(self, layout_fingerprint="")
        ):
            raise EncoderCompatibilityError("encoder layout fingerprint is invalid")

    @cached_property
    def offsets(self) -> dict[str, int]:
        """Return feature-name offsets for inspection and encoding."""

        return {spec.name: spec.offset for spec in self.feature_specs}

    def payload(self) -> dict[str, object]:
        """Return a canonical JSON-compatible manifest payload."""

        payload = asdict(self)
        payload["feature_specs"] = [asdict(spec) for spec in self.feature_specs]
        for key in (
            "card_order",
            "color_order",
            "icon_order",
            "splay_order",
            "normal_achievement_order",
            "special_achievement_order",
            "decision_kind_order",
            "relation_order",
            "incremental_selection_order",
        ):
            payload[key] = list(cast(tuple[str, ...], getattr(self, key)))
        return cast(dict[str, object], payload)

    def dumps(self) -> str:
        """Serialize the manifest deterministically."""

        return json.dumps(self.payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    def save(self, path: str | Path) -> None:
        """Write the deterministic manifest with one trailing newline."""

        Path(path).write_text(f"{self.dumps()}\n", encoding="utf-8")


def _fingerprint_payload(manifest: EncoderManifest) -> dict[str, object]:
    payload = manifest.payload()
    payload.pop("layout_fingerprint")
    return payload


def _manifest_fingerprint(manifest: EncoderManifest) -> str:
    encoded = json.dumps(
        _fingerprint_payload(manifest),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _feature_names(registry: CardRegistry) -> tuple[tuple[str, float], ...]:
    cards = tuple(sorted(registry.cards, key=lambda card: str(card.id)))
    names: list[tuple[str, float]] = []

    def add(name: str, scale: float = 1.0) -> None:
        names.append((name, scale))

    def enum_bits(prefix: str, values: tuple[str, ...]) -> None:
        for value in values:
            add(f"{prefix}.{value}")

    enum_bits("global.phase", tuple(phase.value for phase in GamePhase))
    enum_bits("global.active_relation", tuple(relation.value for relation in PlayerRelation))
    add("global.paid_actions_remaining", 2.0)
    add("global.turn_number", TURN_NUMBER_SCALE)
    for age in range(1, 11):
        add(f"global.supply.age-{age}.count", CARD_COUNT_SCALE)
    for normal_achievement in NormalAchievementId:
        add(f"global.available_normal.{normal_achievement.value}")
    for special_achievement in SpecialAchievementId:
        add(f"global.available_special.{special_achievement.value}")
    for card in cards:
        add(f"global.revealed_card.{card.id}")
    for color in Color:
        add(f"global.revealed_color.{color.value}")

    for relation in ("self", "opponent"):
        for zone_name in ("hand", "score"):
            prefix = f"player.{relation}.{zone_name}"
            for age in range(1, 11):
                add(f"{prefix}.age-{age}.count", CARD_COUNT_SCALE)
            for card in cards:
                add(f"{prefix}.card.{card.id}.present")
                add(f"{prefix}.card.{card.id}.known")
        for normal_achievement in NormalAchievementId:
            add(f"player.{relation}.claimed_normal.{normal_achievement.value}")
        for special_achievement in SpecialAchievementId:
            add(f"player.{relation}.claimed_special.{special_achievement.value}")
        for color in Color:
            prefix = f"player.{relation}.stack.{color.value}"
            add(f"{prefix}.empty")
            for card in cards:
                add(f"{prefix}.top.{card.id}")
            enum_bits(f"{prefix}.splay", tuple(item.value for item in SplayDirection))
            add(f"{prefix}.covered_count", STACK_COUNT_SCALE)
            add(f"{prefix}.covered_count.known")
            for card in cards:
                add(f"{prefix}.covered.{card.id}.present")
                add(f"{prefix}.covered.{card.id}.known")
            for icon in Icon:
                add(f"{prefix}.visible_covered_icons.{icon.value}", ICON_COUNT_SCALE)

    enum_bits(
        "boundary.decision_kind",
        ("none", *(kind.value for kind in DecisionKind)),
    )
    for field in ("chooser", "executor", "dogma_activator"):
        enum_bits(
            f"boundary.{field}_relation",
            tuple(relation.value for relation in PlayerRelation),
        )
    add("boundary.source.present")
    for card in cards:
        add(f"boundary.source.card.{card.id}")
    enum_bits("boundary.source.effect", ("none", "1", "2", "3"))
    add("boundary.context.present")
    for flag in ("demand", "shared", "nested"):
        add(f"boundary.context.{flag}")
    enum_bits("boundary.context.featured_icon", ("none", *(icon.value for icon in Icon)))
    for count in ("activator_icons", "opponent_icons"):
        add(f"boundary.context.{count}", ICON_COUNT_SCALE)
        add(f"boundary.context.{count}.known")
    add("boundary.context.minimum_count", GENERAL_COUNT_SCALE)
    add("boundary.context.maximum_count", GENERAL_COUNT_SCALE)
    for card in cards:
        add(f"boundary.context.selected.{card.id}.present")
        add(f"boundary.context.selected.{card.id}.known")
    add("boundary.context.selected.unknown_count", GENERAL_COUNT_SCALE)
    enum_bits(
        "boundary.context.incremental_selection",
        tuple(item.value for item in IncrementalSelectionKind),
    )
    for relation in ("self", "opponent"):
        add(f"boundary.turn_progress.{relation}.tucked", GENERAL_COUNT_SCALE)
        add(f"boundary.turn_progress.{relation}.scored", GENERAL_COUNT_SCALE)
    enum_bits("position.kind", tuple(item.value for item in ValuePositionKind))
    return tuple(names)


def build_encoder_manifest(
    registry: CardRegistry | None = None,
    *,
    information_policy_version: str = INFORMATION_POLICY_VERSION,
) -> EncoderManifest:
    """Generate encoder-v1 layout for an explicit supported information policy."""

    registry = registry or load_card_registry()
    if information_policy_version not in SUPPORTED_INFORMATION_POLICY_VERSIONS:
        raise EncoderCompatibilityError(
            f"unsupported information-policy version {information_policy_version!r}"
        )
    named = _feature_names(registry)
    specs = tuple(FeatureSpec(name, offset, scale) for offset, (name, scale) in enumerate(named))
    provisional = EncoderManifest(
        encoder_version=ENCODER_VERSION,
        card_data_fingerprint=registry.data_fingerprint,
        rules_version=RULES_VERSION,
        information_policy_version=information_policy_version,
        card_order=tuple(
            str(card.id) for card in sorted(registry.cards, key=lambda item: str(item.id))
        ),
        color_order=tuple(item.value for item in Color),
        icon_order=tuple(item.value for item in Icon),
        splay_order=tuple(item.value for item in SplayDirection),
        normal_achievement_order=tuple(item.value for item in NormalAchievementId),
        special_achievement_order=tuple(item.value for item in SpecialAchievementId),
        decision_kind_order=tuple(item.value for item in DecisionKind),
        relation_order=tuple(item.value for item in PlayerRelation),
        incremental_selection_order=tuple(item.value for item in IncrementalSelectionKind),
        feature_specs=specs,
        input_dimension=len(specs),
        layout_fingerprint="",
    )
    return replace(provisional, layout_fingerprint=_manifest_fingerprint(provisional))


def encoder_manifest_from_payload(payload: object) -> EncoderManifest:
    """Decode and validate an exact encoder manifest payload."""

    if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
        raise EncoderCompatibilityError("encoder manifest must be an object")
    expected = {
        "encoder_version",
        "card_data_fingerprint",
        "rules_version",
        "information_policy_version",
        "card_order",
        "color_order",
        "icon_order",
        "splay_order",
        "normal_achievement_order",
        "special_achievement_order",
        "decision_kind_order",
        "relation_order",
        "incremental_selection_order",
        "feature_specs",
        "input_dimension",
        "layout_fingerprint",
        "schema_version",
        "observation_schema_version",
        "decision_schema_version",
        "value_position_schema_version",
        "public_boundary_schema_version",
    }
    if set(payload) != expected:
        raise EncoderCompatibilityError("encoder manifest fields differ from schema")

    def string(name: str) -> str:
        value = payload[name]
        if not isinstance(value, str):
            raise EncoderCompatibilityError(f"encoder manifest {name} must be a string")
        return value

    def integer(name: str) -> int:
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise EncoderCompatibilityError(f"encoder manifest {name} must be an integer")
        return value

    def strings(name: str) -> tuple[str, ...]:
        value = payload[name]
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise EncoderCompatibilityError(f"encoder manifest {name} must be a string array")
        return tuple(value)

    raw_specs = payload["feature_specs"]
    if not isinstance(raw_specs, list):
        raise EncoderCompatibilityError("encoder manifest feature_specs must be an array")
    specs: list[FeatureSpec] = []
    for raw in raw_specs:
        if not isinstance(raw, dict) or set(raw) != {"name", "offset", "scale"}:
            raise EncoderCompatibilityError("invalid encoder feature specification payload")
        name = raw["name"]
        offset = raw["offset"]
        scale = raw["scale"]
        if (
            not isinstance(name, str)
            or isinstance(offset, bool)
            or not isinstance(offset, int)
            or isinstance(scale, bool)
            or not isinstance(scale, (int, float))
        ):
            raise EncoderCompatibilityError("invalid encoder feature specification value")
        specs.append(FeatureSpec(name, offset, float(scale)))
    return EncoderManifest(
        encoder_version=string("encoder_version"),
        card_data_fingerprint=string("card_data_fingerprint"),
        rules_version=string("rules_version"),
        information_policy_version=string("information_policy_version"),
        card_order=strings("card_order"),
        color_order=strings("color_order"),
        icon_order=strings("icon_order"),
        splay_order=strings("splay_order"),
        normal_achievement_order=strings("normal_achievement_order"),
        special_achievement_order=strings("special_achievement_order"),
        decision_kind_order=strings("decision_kind_order"),
        relation_order=strings("relation_order"),
        incremental_selection_order=strings("incremental_selection_order"),
        feature_specs=tuple(specs),
        input_dimension=integer("input_dimension"),
        layout_fingerprint=string("layout_fingerprint"),
        schema_version=integer("schema_version"),
        observation_schema_version=integer("observation_schema_version"),
        decision_schema_version=integer("decision_schema_version"),
        value_position_schema_version=integer("value_position_schema_version"),
        public_boundary_schema_version=integer("public_boundary_schema_version"),
    )


def load_encoder_manifest(path: str | Path) -> EncoderManifest:
    """Load a strict encoder manifest from JSON."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EncoderCompatibilityError(f"could not load encoder manifest: {error}") from error
    return encoder_manifest_from_payload(payload)


class FlatObservationEncoder:
    """Encode only detached ``ValuePosition`` contracts as contiguous float32 vectors."""

    def __init__(
        self,
        registry: CardRegistry | None = None,
        manifest: EncoderManifest | None = None,
    ) -> None:
        self.registry = registry or load_card_registry()
        self.manifest = manifest or build_encoder_manifest(self.registry)
        expected = build_encoder_manifest(
            self.registry,
            information_policy_version=self.manifest.information_policy_version,
        )
        if self.manifest != expected:
            raise EncoderCompatibilityError(
                "encoder manifest does not match the installed card data and schema layout"
            )
        self._cards = tuple(CardId(value) for value in self.manifest.card_order)
        self._offsets = self.manifest.offsets

    def _validate(self, position: ValuePosition) -> None:
        observation = position.observation
        if position.schema_version != self.manifest.value_position_schema_version:
            raise EncoderCompatibilityError("value-position schema mismatch")
        if position.boundary.schema_version != self.manifest.public_boundary_schema_version:
            raise EncoderCompatibilityError("public-boundary schema mismatch")
        if observation.schema_version != self.manifest.observation_schema_version:
            raise EncoderCompatibilityError("observation schema mismatch")
        if observation.rules_version != self.manifest.rules_version:
            raise EncoderCompatibilityError("observation rules mismatch")
        if observation.information_policy.value != self.manifest.information_policy_version:
            raise EncoderCompatibilityError("observation information-policy mismatch")
        if tuple(supply.age for supply in observation.supplies) != tuple(range(1, 11)):
            raise EncoderCompatibilityError("observation supplies are not in canonical order")

    def encode(self, position: ValuePosition) -> NDArray[np.float32]:
        """Return one fixed-shape contiguous float32 vector."""

        self._validate(position)
        vector = np.zeros(self.manifest.input_dimension, dtype=np.float32)

        def put(name: str, value: float = 1.0) -> None:
            vector[self._offsets[name]] = np.float32(value)

        def one_hot(prefix: str, value: str) -> None:
            put(f"{prefix}.{value}")

        observation = position.observation
        viewer = position.viewer
        opponent = next(player for player in PlayerId if player is not viewer)
        one_hot("global.phase", observation.phase.value)
        active_relation = (
            PlayerRelation.NONE
            if observation.active_player is None
            else (
                PlayerRelation.SELF
                if observation.active_player is viewer
                else PlayerRelation.OPPONENT
            )
        )
        one_hot("global.active_relation", active_relation.value)
        put("global.paid_actions_remaining", min(observation.paid_actions_remaining, 2) / 2.0)
        put(
            "global.turn_number",
            min(observation.turn_number, int(TURN_NUMBER_SCALE)) / TURN_NUMBER_SCALE,
        )
        for supply in observation.supplies:
            put(f"global.supply.age-{supply.age}.count", min(supply.count, 15) / CARD_COUNT_SCALE)
        for normal_achievement in observation.available_normal_achievements:
            put(f"global.available_normal.{normal_achievement.value}")
        for special_achievement in observation.available_special_achievements:
            put(f"global.available_special.{special_achievement.value}")
        for card_id in observation.revealed_cards:
            put(f"global.revealed_card.{card_id}")
        for color in observation.revealed_colors:
            put(f"global.revealed_color.{color.value}")

        for relation, player_id in (("self", viewer), ("opponent", opponent)):
            player = observation.player(player_id)
            self._encode_zone(vector, put, f"player.{relation}.hand", player.hand)
            self._encode_zone(vector, put, f"player.{relation}.score", player.score_pile)
            for normal_achievement in player.normal_achievements:
                put(f"player.{relation}.claimed_normal.{normal_achievement.value}")
            for special_achievement in player.special_achievements:
                put(f"player.{relation}.claimed_special.{special_achievement.value}")
            if tuple(stack.color for stack in player.board) != tuple(Color):
                raise EncoderCompatibilityError("observation board is not in canonical color order")
            for stack in player.board:
                self._encode_stack(vector, put, relation, stack)

        boundary = position.boundary
        one_hot(
            "boundary.decision_kind",
            "none" if boundary.decision_kind is None else boundary.decision_kind.value,
        )
        one_hot("boundary.chooser_relation", boundary.chooser_relation.value)
        one_hot("boundary.executor_relation", boundary.executor_relation.value)
        one_hot("boundary.dogma_activator_relation", boundary.dogma_activator_relation.value)
        if boundary.source is None:
            one_hot("boundary.source.effect", "none")
        else:
            put("boundary.source.present")
            put(f"boundary.source.card.{boundary.source.card_id}")
            one_hot(
                "boundary.source.effect",
                (
                    "none"
                    if boundary.source.effect_id is None
                    else str(boundary.source.effect_id.ordinal)
                ),
            )
        context = boundary.context
        if context is not None:
            put("boundary.context.present")
            for flag in ("demand", "shared", "nested"):
                if cast(bool, getattr(context, flag)):
                    put(f"boundary.context.{flag}")
            one_hot(
                "boundary.context.featured_icon",
                "none" if context.featured_icon is None else context.featured_icon.value,
            )
            for name in ("activator_icons", "opponent_icons"):
                count = cast(int | None, getattr(context, name))
                if count is not None:
                    put(f"boundary.context.{name}", min(count, 21) / ICON_COUNT_SCALE)
                    put(f"boundary.context.{name}.known")
            put(
                "boundary.context.minimum_count",
                min(context.minimum_count, 105) / GENERAL_COUNT_SCALE,
            )
            put(
                "boundary.context.maximum_count",
                min(context.maximum_count, 105) / GENERAL_COUNT_SCALE,
            )
            visible = frozenset(context.visible_selected_cards)
            selection_complete = context.unknown_selected_count == 0
            for card_id in self._cards:
                if card_id in visible:
                    put(f"boundary.context.selected.{card_id}.present")
                if selection_complete or card_id in visible:
                    put(f"boundary.context.selected.{card_id}.known")
            put(
                "boundary.context.selected.unknown_count",
                min(context.unknown_selected_count, 105) / GENERAL_COUNT_SCALE,
            )
            one_hot(
                "boundary.context.incremental_selection",
                context.incremental_selection.value,
            )
        else:
            one_hot("boundary.context.featured_icon", "none")
            one_hot("boundary.context.incremental_selection", IncrementalSelectionKind.NONE.value)
        progress = boundary.turn_progress
        for relation, tucked, scored in (
            ("self", progress.self_tucked, progress.self_scored),
            ("opponent", progress.opponent_tucked, progress.opponent_scored),
        ):
            put(
                f"boundary.turn_progress.{relation}.tucked",
                min(tucked, 105) / GENERAL_COUNT_SCALE,
            )
            put(
                f"boundary.turn_progress.{relation}.scored",
                min(scored, 105) / GENERAL_COUNT_SCALE,
            )
        one_hot("position.kind", position.position_kind.value)
        return np.ascontiguousarray(vector, dtype=np.float32)

    def encode_batch(self, positions: tuple[ValuePosition, ...]) -> NDArray[np.float32]:
        """Encode a batch with shape ``[N, D]``."""

        if not positions:
            return np.empty((0, self.manifest.input_dimension), dtype=np.float32)
        return np.ascontiguousarray(np.stack(tuple(self.encode(item) for item in positions)))

    def inspect_nonzero(self, position: ValuePosition) -> tuple[tuple[str, float], ...]:
        """Return named nonzero features for compact debugging output."""

        vector = self.encode(position)
        return tuple(
            (spec.name, float(vector[spec.offset]))
            for spec in self.manifest.feature_specs
            if vector[spec.offset] != 0.0
        )

    def _encode_zone(
        self,
        vector: NDArray[np.float32],
        put: _Setter,
        prefix: str,
        zone: ZoneObservation,
    ) -> None:
        del vector
        call = put
        counts = Counter(zone.values)
        for age in range(1, 11):
            call(f"{prefix}.age-{age}.count", min(counts[age], 15) / CARD_COUNT_SCALE)
        visible = frozenset(zone.known_cards)
        complete = len(zone.known_cards) == zone.count
        for card_id in self._cards:
            if card_id in visible:
                call(f"{prefix}.card.{card_id}.present")
            if complete or card_id in visible:
                call(f"{prefix}.card.{card_id}.known")

    def _encode_stack(
        self,
        vector: NDArray[np.float32],
        put: _Setter,
        relation: str,
        stack: StackObservation,
    ) -> None:
        del vector
        call = put
        prefix = f"player.{relation}.stack.{stack.color.value}"
        if stack.top_card_id is None:
            call(f"{prefix}.empty")
        else:
            call(f"{prefix}.top.{stack.top_card_id}")
        call(f"{prefix}.splay.{stack.splay.value}")
        if stack.covered_count is not None:
            call(f"{prefix}.covered_count", min(stack.covered_count, 21) / STACK_COUNT_SCALE)
            call(f"{prefix}.covered_count.known")
        known_cards = frozenset(
            item.card_id for item in stack.covered_cards if item.card_id is not None
        )
        complete = stack.covered_count is not None and len(known_cards) == stack.covered_count
        for card_id in self._cards:
            if card_id in known_cards:
                call(f"{prefix}.covered.{card_id}.present")
            if complete or card_id in known_cards:
                call(f"{prefix}.covered.{card_id}.known")
        visible_icons = Counter(
            icon for covered in stack.covered_cards for icon in covered.visible_icons
        )
        for icon in Icon:
            call(
                f"{prefix}.visible_covered_icons.{icon.value}",
                min(visible_icons[icon], 21) / ICON_COUNT_SCALE,
            )
