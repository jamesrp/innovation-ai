"""Validated immutable catalog for the supplied Innovation cards."""

from __future__ import annotations

import csv
import hashlib
import io
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from types import MappingProxyType
from typing import Final

from innovation_ai.innovation.types import (
    CardId,
    Color,
    DogmaEffectId,
    Icon,
    IconSlot,
    SpecialAchievementId,
)

_DATA_PACKAGE: Final = "innovation_ai.innovation.data"
_CARD_DATA_FILE: Final = "cards.csv"
_ACHIEVEMENT_DATA_FILE: Final = "special_achievements.csv"
_CARD_COLUMNS: Final = (
    "Name",
    "Age",
    "Color",
    "Dogma 1",
    "Dogma 2",
    "Dogma 3",
    "Main Symbol",
    "Symbol 1",
    "Symbol 2",
    "Symbol 3",
    "Symbol 4",
    "Symbol 5",
    "Symbol 6",
)
_ACHIEVEMENT_COLUMNS: Final = ("Name", "Condition", "Linked Card")
_SLOT_COLUMNS: Final = MappingProxyType(
    {
        IconSlot.TOP_LEFT: "Symbol 1",
        IconSlot.BOTTOM_LEFT: "Symbol 4",
        IconSlot.BOTTOM_CENTER: "Symbol 5",
        IconSlot.BOTTOM_RIGHT: "Symbol 6",
    }
)
_SLOT_INDEX: Final = MappingProxyType({slot: index for index, slot in enumerate(IconSlot)})
_EXPECTED_AGE_COUNTS: Final = {1: 15, **{age: 10 for age in range(2, 11)}}
_EXPECTED_SPECIAL_CONDITIONS: Final = {
    SpecialAchievementId.EMPIRE: "Three or more icons of all six types",
    SpecialAchievementId.MONUMENT: "Tuck six or Score six in one turn",
    SpecialAchievementId.UNIVERSE: "Five top cards, each of value 8 or higher",
    SpecialAchievementId.WONDER: "Five colors, each splayed up or right",
    SpecialAchievementId.WORLD: "Twelve clocks",
}
_EXPECTED_LINKED_CARDS: Final = {
    SpecialAchievementId.EMPIRE: CardId.from_name("Construction"),
    SpecialAchievementId.MONUMENT: CardId.from_name("Masonry"),
    SpecialAchievementId.UNIVERSE: CardId.from_name("Astronomy"),
    SpecialAchievementId.WONDER: CardId.from_name("Invention"),
    SpecialAchievementId.WORLD: CardId.from_name("Translation"),
}


class CatalogValidationError(ValueError):
    """The packaged authoritative data violates a frozen catalog assumption."""


@dataclass(frozen=True, slots=True)
class PrintedDogmaEffect:
    """Reference text and stable identity for one printed dogma effect."""

    id: DogmaEffectId
    text: str


@dataclass(frozen=True, slots=True)
class Card:
    """Immutable printed data for one Innovation card."""

    id: CardId
    name: str
    age: int
    color: Color
    dogma_effects: tuple[PrintedDogmaEffect, ...]
    featured_icon: Icon
    face_symbols: tuple[Icon | None, Icon | None, Icon | None, Icon | None]
    image_slot: IconSlot

    def icon_at(self, slot: IconSlot) -> Icon | None:
        """Return the functional icon at a position, or ``None`` for the image."""

        return self.face_symbols[_SLOT_INDEX[slot]]

    @property
    def functional_icons(self) -> tuple[Icon, Icon, Icon]:
        """Return the three functional face icons in geometric slot order."""

        icons = tuple(symbol for symbol in self.face_symbols if symbol is not None)
        if len(icons) != 3:  # Defensive: validated during loading.
            raise CatalogValidationError(f"card {self.id} does not have three icons")
        return icons[0], icons[1], icons[2]


@dataclass(frozen=True, slots=True)
class SpecialAchievementDefinition:
    """The automatic public predicate metadata for one special achievement."""

    id: SpecialAchievementId
    name: str
    source_condition: str


@dataclass(frozen=True, slots=True)
class LinkedAchievementRoute:
    """A card-specific alternate route to a special achievement."""

    achievement_id: SpecialAchievementId
    source_effect_id: DogmaEffectId

    @property
    def source_card_id(self) -> CardId:
        """Return the card containing the alternate award route."""

        return self.source_effect_id.card_id


@dataclass(frozen=True, slots=True)
class CardRegistry:
    """Immutable validated card and achievement reference data."""

    cards: tuple[Card, ...]
    by_id: Mapping[CardId, Card]
    special_achievements: Mapping[SpecialAchievementId, SpecialAchievementDefinition]
    linked_achievement_routes: Mapping[SpecialAchievementId, LinkedAchievementRoute]
    data_fingerprint: str

    def card(self, card_id: CardId | str) -> Card:
        """Look up a card by canonical ID or a printed/clarified name."""

        normalized = card_id if isinstance(card_id, CardId) else CardId.from_name(card_id)
        try:
            return self.by_id[normalized]
        except KeyError as error:
            raise KeyError(f"unknown card: {card_id}") from error


def _data_bytes(filename: str) -> bytes:
    return resources.files(_DATA_PACKAGE).joinpath(filename).read_bytes()


def _rows(data: bytes, expected_columns: tuple[str, ...]) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""))
    if tuple(reader.fieldnames or ()) != expected_columns:
        raise CatalogValidationError(
            f"unexpected columns: {reader.fieldnames!r}; expected {expected_columns!r}"
        )
    return [dict(row) for row in reader]


def _icon(source_value: str) -> Icon:
    canonical = "lightbulb" if source_value == "bulb" else source_value
    try:
        return Icon(canonical)
    except ValueError as error:
        raise CatalogValidationError(f"unknown icon value: {source_value!r}") from error


def _parse_card(row: Mapping[str, str]) -> Card:
    name = row["Name"].strip()
    card_id = CardId.from_name(name)
    try:
        age = int(row["Age"])
        color = Color(row["Color"])
    except ValueError as error:
        raise CatalogValidationError(f"invalid age/color for {name}") from error

    if not 1 <= age <= 10:
        raise CatalogValidationError(f"invalid age for {name}: {age}")
    if row["Symbol 2"] or row["Symbol 3"]:
        raise CatalogValidationError(f"unused symbol columns are populated for {name}")

    symbols: list[Icon | None] = []
    image_slots: list[IconSlot] = []
    for slot, column in _SLOT_COLUMNS.items():
        source_symbol = row[column].strip()
        if source_symbol == "hex":
            symbols.append(None)
            image_slots.append(slot)
        else:
            symbols.append(_icon(source_symbol))
    if len(image_slots) != 1:
        raise CatalogValidationError(f"{name} must have exactly one image slot")

    raw_effects = tuple(row[f"Dogma {index}"].strip() for index in range(1, 4))
    if not raw_effects[0]:
        raise CatalogValidationError(f"{name} has no dogma effects")
    if any(raw_effects[index] and not raw_effects[index - 1] for index in range(1, 3)):
        raise CatalogValidationError(f"{name} has a gap between dogma effects")
    effects = tuple(
        PrintedDogmaEffect(DogmaEffectId(card_id, index), text)
        for index, text in enumerate(raw_effects, start=1)
        if text
    )

    return Card(
        id=card_id,
        name=name,
        age=age,
        color=color,
        dogma_effects=effects,
        featured_icon=_icon(row["Main Symbol"].strip()),
        face_symbols=(symbols[0], symbols[1], symbols[2], symbols[3]),
        image_slot=image_slots[0],
    )


def _parse_achievements(
    rows: list[dict[str, str]], by_id: Mapping[CardId, Card]
) -> tuple[
    Mapping[SpecialAchievementId, SpecialAchievementDefinition],
    Mapping[SpecialAchievementId, LinkedAchievementRoute],
]:
    definitions: dict[SpecialAchievementId, SpecialAchievementDefinition] = {}
    routes: dict[SpecialAchievementId, LinkedAchievementRoute] = {}
    for row in rows:
        try:
            achievement_id = SpecialAchievementId(row["Name"].strip().casefold())
        except ValueError as error:
            raise CatalogValidationError(f"unknown special achievement: {row['Name']!r}") from error
        if achievement_id in definitions:
            raise CatalogValidationError(f"duplicate special achievement: {achievement_id}")

        condition = row["Condition"].strip()
        linked_card = CardId.from_name(row["Linked Card"])
        if condition != _EXPECTED_SPECIAL_CONDITIONS[achievement_id]:
            raise CatalogValidationError(f"unexpected condition summary for {achievement_id}")
        if linked_card != _EXPECTED_LINKED_CARDS[achievement_id]:
            raise CatalogValidationError(f"unexpected linked card for {achievement_id}")
        if linked_card not in by_id:
            raise CatalogValidationError(f"linked achievement card is absent: {linked_card}")

        definitions[achievement_id] = SpecialAchievementDefinition(
            id=achievement_id,
            name=row["Name"].strip(),
            source_condition=condition,
        )
        routes[achievement_id] = LinkedAchievementRoute(
            achievement_id,
            DogmaEffectId(linked_card, 1 if achievement_id is SpecialAchievementId.MONUMENT else 2),
        )

    if set(definitions) != set(SpecialAchievementId):
        raise CatalogValidationError("special achievement set is incomplete")
    return MappingProxyType(definitions), MappingProxyType(routes)


def _fingerprint(card_data: bytes, achievement_data: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(b"innovation-ai-card-data-v1\0")
    for filename, data in (
        (_CARD_DATA_FILE, card_data),
        (_ACHIEVEMENT_DATA_FILE, achievement_data),
    ):
        digest.update(filename.encode("ascii"))
        digest.update(b"\0")
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return f"sha256:{digest.hexdigest()}"


@lru_cache(maxsize=1)
def load_card_registry() -> CardRegistry:
    """Load and validate the packaged registry, caching the immutable result."""

    card_data = _data_bytes(_CARD_DATA_FILE)
    achievement_data = _data_bytes(_ACHIEVEMENT_DATA_FILE)
    cards = tuple(_parse_card(row) for row in _rows(card_data, _CARD_COLUMNS))
    by_id_mutable = {card.id: card for card in cards}

    if len(cards) != 105 or len(by_id_mutable) != len(cards):
        raise CatalogValidationError("catalog must contain exactly 105 uniquely identified cards")
    if Counter(card.age for card in cards) != Counter(_EXPECTED_AGE_COUNTS):
        raise CatalogValidationError("card age histogram differs from the supplied game")
    if Counter(card.color for card in cards) != Counter({color: 21 for color in Color}):
        raise CatalogValidationError("each color must contain exactly 21 cards")
    for age in range(1, 11):
        expected_per_color = 3 if age == 1 else 2
        if Counter(card.color for card in cards if card.age == age) != Counter(
            {color: expected_per_color for color in Color}
        ):
            raise CatalogValidationError(f"age {age} has an invalid color histogram")
    if any(card.featured_icon not in card.functional_icons for card in cards):
        raise CatalogValidationError("a featured icon is absent from its card face")
    if Counter(len(card.dogma_effects) for card in cards) != Counter({1: 55, 2: 47, 3: 3}):
        raise CatalogValidationError("dogma-effect histogram differs from the supplied game")

    by_id: Mapping[CardId, Card] = MappingProxyType(by_id_mutable)
    achievements, routes = _parse_achievements(_rows(achievement_data, _ACHIEVEMENT_COLUMNS), by_id)
    return CardRegistry(
        cards=cards,
        by_id=by_id,
        special_achievements=achievements,
        linked_achievement_routes=routes,
        data_fingerprint=_fingerprint(card_data, achievement_data),
    )
