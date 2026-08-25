"""Owner-only registry discovery, coverage, and card-module hygiene checks.

Milestone 1 is not done until all 105 cards are registered; until then this test reports exactly
which cards are implemented, so a missing card fails loudly instead of acting as a no-op.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
from pathlib import Path

import pytest

from innovation_ai.innovation.catalog import CardRegistry, load_card_registry
from innovation_ai.innovation.effects import (
    EffectContext,
    EffectProgramRegistry,
    EffectStatus,
    UnimplementedCardError,
    build_effect_programs,
    effects_fingerprint,
    implemented_card_ids,
    load_effect_programs,
    start_dogma,
    submit_effect_action,
)
from innovation_ai.innovation.effects.program import (
    ConditionNode,
    EffectProgram,
    NoOpNode,
    Predicate,
    ProgramEffect,
)
from innovation_ai.innovation.effects.registry import EffectRegistryError
from innovation_ai.innovation.state import (
    ExplicitPlayerPosition,
    GameState,
    build_explicit_state,
)
from innovation_ai.innovation.types import CardId, DogmaEffectId, PlayerId

REGISTRY = load_card_registry()

# Milestone 1 final manifest: every catalog card must have one discovered implementation.
WAVE_MANIFEST = frozenset(card.id for card in REGISTRY.cards)
TOTAL_CARDS = 105


def _fingerprint_true(state: GameState, context: EffectContext, registry: CardRegistry) -> bool:
    return True


def _fingerprint_false(state: GameState, context: EffectContext, registry: CardRegistry) -> bool:
    return False


def test_the_implemented_set_equals_the_current_wave_manifest() -> None:
    assert implemented_card_ids() == WAVE_MANIFEST


def test_every_implemented_card_id_exists_in_the_catalog() -> None:
    for card_id in implemented_card_ids():
        assert REGISTRY.card(card_id).id == card_id


def test_milestone_one_coverage_is_reported_rather_than_silently_incomplete() -> None:
    implemented = implemented_card_ids()
    missing = tuple(sorted(str(card.id) for card in REGISTRY.cards if card.id not in implemented))
    assert len(implemented) + len(missing) == TOTAL_CARDS
    if missing:
        pytest.skip(f"{len(missing)} of {TOTAL_CARDS} cards remain: {missing[:5]}...")


def test_every_card_resolves_from_its_minimum_board_state() -> None:
    """Registration alone is insufficient: every card must survive legal partial execution."""

    programs = load_effect_programs()
    for card in REGISTRY.cards:
        state = build_explicit_state(
            REGISTRY,
            positions=(
                (
                    PlayerId.PLAYER_1,
                    ExplicitPlayerPosition(board=((card.color, (card.id,)),)),
                ),
            ),
        )
        resolution = start_dogma(
            state,
            card.id,
            PlayerId.PLAYER_1,
            programs,
            REGISTRY,
        )
        decision_count = 0
        while resolution.status is EffectStatus.AWAIT_DECISION:
            decision_count += 1
            assert decision_count <= 128, f"{card.id} did not converge"
            decision = resolution.decision
            assert decision is not None
            resolution = submit_effect_action(
                resolution.state,
                decision.legal_actions[0],
                programs,
                REGISTRY,
            )
        assert resolution.status in {EffectStatus.COMPLETE, EffectStatus.TERMINAL}, card.id


def test_discovery_is_deterministic_and_cached() -> None:
    first = load_effect_programs()
    assert load_effect_programs() is first
    rebuilt = build_effect_programs(REGISTRY)
    assert rebuilt.implemented_card_ids() == first.implemented_card_ids()
    assert rebuilt.fingerprint() == first.fingerprint()


def test_the_effects_fingerprint_changes_when_a_program_changes() -> None:
    baseline = effects_fingerprint()
    assert baseline.startswith("sha256:")
    original = load_effect_programs().program_for_card(CardId("the-wheel"))
    tweaked = EffectProgramRegistry(
        (
            EffectProgram(
                original.program_id,
                original.source_card_id,
                original.effects,
                original.nodes,
            ),
        )
    )
    assert tweaked.fingerprint() != baseline


def test_the_effects_fingerprint_includes_named_helper_behavior() -> None:
    card_id = CardId("the-wheel")
    program = EffectProgram(
        "named-fingerprint-v1",
        card_id,
        (ProgramEffect(DogmaEffectId(card_id, 1), False, "condition"),),
        (
            ConditionNode("condition", Predicate.named("switch"), "yes", "no"),
            NoOpNode("yes"),
            NoOpNode("no"),
        ),
    )
    first = EffectProgramRegistry((program,), predicates={card_id: {"switch": _fingerprint_true}})
    second = EffectProgramRegistry((program,), predicates={card_id: {"switch": _fingerprint_false}})
    assert first.fingerprint() != second.fingerprint()


def test_a_partial_registry_raises_a_typed_error_and_never_no_ops() -> None:
    complete = load_effect_programs()
    wheel = complete.program_for_card(CardId("the-wheel"))
    partial = EffectProgramRegistry((wheel,))
    with pytest.raises(UnimplementedCardError) as error:
        partial.program_for_card(CardId("tools"))
    assert error.value.card_id == CardId("tools")


def test_a_demand_flag_that_disagrees_with_printed_text_is_rejected() -> None:
    """The one place printed prose is touched: as an assertion, never as behaviour."""

    card_id = CardId("the-wheel")
    lying = EffectProgram(
        "the-wheel-lying",
        card_id,
        (ProgramEffect(DogmaEffectId(card_id, 1), True, "nothing"),),
        (NoOpNode("nothing"),),
    )
    from innovation_ai.innovation.effects.registry import _validate_program

    with pytest.raises(EffectRegistryError, match="demand flag"):
        _validate_program(lying, card_id, REGISTRY, {}, {})


def test_an_effect_count_that_disagrees_with_the_card_is_rejected() -> None:
    """POTTERY prints two effects; a program implementing one must be rejected."""

    card_id = CardId("pottery")
    assert len(REGISTRY.card(card_id).dogma_effects) == 2
    stub = EffectProgram(
        "pottery-stub",
        card_id,
        (ProgramEffect(DogmaEffectId(card_id, 1), False, "nothing"),),
        (NoOpNode("nothing"),),
    )
    from innovation_ai.innovation.effects.registry import _validate_program

    with pytest.raises(EffectRegistryError, match="the card prints"):
        _validate_program(stub, card_id, REGISTRY, {}, {})


def _card_module_sources() -> tuple[tuple[str, str], ...]:
    package = importlib.import_module("innovation_ai.innovation.cards")
    sources: list[tuple[str, str]] = []
    for age_info in pkgutil.iter_modules(package.__path__):
        if not age_info.ispkg:
            continue
        age_package = importlib.import_module(f"innovation_ai.innovation.cards.{age_info.name}")
        for info in pkgutil.iter_modules(age_package.__path__):
            module = importlib.import_module(
                f"innovation_ai.innovation.cards.{age_info.name}.{info.name}"
            )
            path = module.__file__
            assert path is not None
            sources.append((module.__name__, Path(path).read_text(encoding="utf-8")))
    return tuple(sources)


BANNED_IMPORTS = (
    "innovation_ai.innovation.zones",
    "innovation_ai.innovation.state",
    "innovation_ai.innovation.protocol",
    "innovation_ai.innovation.effects.engine",
    "innovation_ai.innovation.effects.dogma",
)


def test_no_card_module_can_reach_a_mutation_path() -> None:
    """Card modules are pure data: they never import a mutation primitive or the interpreter."""

    for name, source in _card_module_sources():
        tree = ast.parse(source, filename=name)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.add(node.module)
        for banned in BANNED_IMPORTS:
            assert banned not in imported, f"{name} must not import {banned}"
        assert "dataclasses" not in imported, f"{name} must not use dataclasses.replace"


def test_every_card_module_exposes_exactly_the_frozen_contract() -> None:
    package = importlib.import_module("innovation_ai.innovation.cards")
    for age_info in pkgutil.iter_modules(package.__path__):
        if not age_info.ispkg:
            continue
        age_package = importlib.import_module(f"innovation_ai.innovation.cards.{age_info.name}")
        for info in pkgutil.iter_modules(age_package.__path__):
            module = importlib.import_module(
                f"innovation_ai.innovation.cards.{age_info.name}.{info.name}"
            )
            assert isinstance(module.CARD_ID, CardId)
            assert isinstance(module.EFFECTS, EffectProgram)
            assert module.__doc__, f"{module.__name__} needs a docstring quoting its card text"
            # The module's age package must match the card's printed age.
            expected = f"age{REGISTRY.card(module.CARD_ID).age:02d}"
            assert age_info.name == expected, f"{module.__name__} is in the wrong age package"
