"""Lazy discovery and validation of the declarative card effect programs.

One module per card, grouped into ``cards/ageNN`` packages purely for navigation. Discovery is a
lazy function rather than decorator auto-registration, because import-time side effects are banned
and because a wave-scoped run must be able to ask which cards are actually implemented.

Each card module exposes exactly:

.. code-block:: python

    CARD_ID: Final[CardId]
    EFFECTS: Final[EffectProgram]                        # one program, all printed ordinals
    PREDICATES: Final[Mapping[str, NamedPredicate]] = {}  # optional bounded escape hatch
    VALUES: Final[Mapping[str, NamedValue]] = {}
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Mapping
from functools import lru_cache
from types import ModuleType

from innovation_ai.innovation.catalog import CardRegistry, load_card_registry
from innovation_ai.innovation.types import CardId

from .program import (
    ClaimAchievementNode,
    EffectProgram,
    EffectProgramRegistry,
    NamedPredicate,
    NamedValue,
    Predicate,
    PredicateKind,
    ValueRef,
    ValueRefKind,
)

_CARDS_PACKAGE = "innovation_ai.innovation.cards"
_DEMAND_PREFIX = "i demand"


class EffectRegistryError(ValueError):
    """A discovered card module violates the frozen card-module contract."""


def _age_packages() -> tuple[str, ...]:
    package = importlib.import_module(_CARDS_PACKAGE)
    return tuple(
        sorted(
            f"{_CARDS_PACKAGE}.{info.name}"
            for info in pkgutil.iter_modules(package.__path__)
            if info.ispkg and info.name.startswith("age")
        )
    )


def _card_modules() -> tuple[ModuleType, ...]:
    modules: list[ModuleType] = []
    for package_name in _age_packages():
        package = importlib.import_module(package_name)
        for info in sorted(pkgutil.iter_modules(package.__path__), key=lambda item: item.name):
            if info.ispkg or info.name.startswith("_"):
                continue
            modules.append(importlib.import_module(f"{package_name}.{info.name}"))
    return tuple(modules)


def _named_references(program: EffectProgram) -> tuple[frozenset[str], frozenset[str]]:
    predicates: set[str] = set()
    values: set[str] = set()

    def visit(item: object) -> None:
        if isinstance(item, Predicate):
            if item.kind is PredicateKind.NAMED and item.name is not None:
                predicates.add(item.name)
            for child in (item.operand, item.left, item.right, item.cards, item.match):
                if child is not None:
                    visit(child)
            return
        if isinstance(item, ValueRef):
            if item.kind is ValueRefKind.NAMED and item.name is not None:
                values.add(item.name)
            if item.selector is not None:
                visit(item.selector)
            return
        for attribute in (
            "predicate",
            "guard",
            "repeat_if",
            "cards",
            "first",
            "second",
            "count",
            "requested_age",
            "value",
            "value_expr",
            "relation",
            "reference",
        ):
            child = getattr(item, attribute, None)
            if child is not None and not isinstance(child, (str, int, bool, tuple)):
                visit(child)
        selector_predicate = getattr(item, "predicate", None)
        if isinstance(selector_predicate, str):
            predicates.add(selector_predicate)

    for node in program.nodes:
        visit(node)
    return frozenset(predicates), frozenset(values)


def _validate_program(
    program: EffectProgram,
    card_id: CardId,
    registry: CardRegistry,
    predicates: Mapping[str, NamedPredicate],
    values: Mapping[str, NamedValue],
) -> None:
    if program.source_card_id != card_id:
        raise EffectRegistryError(
            f"module for {card_id} declares a program for {program.source_card_id}"
        )
    card = registry.card(card_id)
    if len(program.effects) != len(card.dogma_effects):
        raise EffectRegistryError(
            f"{card_id} implements {len(program.effects)} effects but the card prints "
            f"{len(card.dogma_effects)}"
        )
    for implemented, printed in zip(program.effects, card.dogma_effects, strict=True):
        if implemented.effect_id != printed.id:
            raise EffectRegistryError(f"{card_id} effect ordinals do not match printed order")
        # The one place printed prose is touched, and only as an assertion.
        expected_demand = printed.text.casefold().startswith(_DEMAND_PREFIX)
        if implemented.demand != expected_demand:
            raise EffectRegistryError(
                f"{card_id} effect {printed.id.ordinal} demand flag disagrees with its text"
            )
    needed_predicates, needed_values = _named_references(program)
    missing_predicates = needed_predicates - set(predicates)
    missing_values = needed_values - set(values)
    if missing_predicates:
        raise EffectRegistryError(
            f"{card_id} references unknown named predicates: {sorted(missing_predicates)}"
        )
    if missing_values:
        raise EffectRegistryError(
            f"{card_id} references unknown named values: {sorted(missing_values)}"
        )
    unused_predicates = set(predicates) - needed_predicates
    unused_values = set(values) - needed_values
    if unused_predicates or unused_values:
        raise EffectRegistryError(
            f"{card_id} registers unused named callables: "
            f"{sorted(unused_predicates | unused_values)}"
        )
    for node in program.nodes:
        if isinstance(node, ClaimAchievementNode):
            route = registry.linked_achievement_routes[node.achievement_id]
            if route.source_card_id != card_id:
                raise EffectRegistryError(
                    f"{card_id} claims {node.achievement_id} but its linked route belongs to "
                    f"{route.source_card_id}"
                )


def build_effect_programs(registry: CardRegistry | None = None) -> EffectProgramRegistry:
    """Discover, validate, and assemble every implemented card effect program."""

    registry = registry or load_card_registry()
    programs: list[EffectProgram] = []
    predicates: dict[CardId, Mapping[str, NamedPredicate]] = {}
    values: dict[CardId, Mapping[str, NamedValue]] = {}
    for module in _card_modules():
        try:
            card_id = module.CARD_ID
            program = module.EFFECTS
        except AttributeError as error:
            raise EffectRegistryError(
                f"card module {module.__name__} must define CARD_ID and EFFECTS"
            ) from error
        if not isinstance(card_id, CardId):
            raise EffectRegistryError(f"{module.__name__}.CARD_ID must be a CardId")
        if not isinstance(program, EffectProgram):
            raise EffectRegistryError(f"{module.__name__}.EFFECTS must be an EffectProgram")
        module_predicates: Mapping[str, NamedPredicate] = getattr(module, "PREDICATES", {})
        module_values: Mapping[str, NamedValue] = getattr(module, "VALUES", {})
        _validate_program(program, card_id, registry, module_predicates, module_values)
        programs.append(program)
        if module_predicates:
            predicates[card_id] = module_predicates
        if module_values:
            values[card_id] = module_values
    return EffectProgramRegistry(tuple(programs), predicates=predicates, values=values)


@lru_cache(maxsize=1)
def load_effect_programs() -> EffectProgramRegistry:
    """Return the cached immutable production effect-program registry."""

    return build_effect_programs()


def implemented_card_ids() -> frozenset[CardId]:
    """Return every card whose effects are implemented right now."""

    return load_effect_programs().implemented_card_ids()


def effects_fingerprint() -> str:
    """Return the digest recorded in logs beside the card-data fingerprint."""

    return load_effect_programs().fingerprint()
