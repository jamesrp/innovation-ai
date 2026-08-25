"""PHYSICS: three-card reveal, duplicate-colour branch, ordering, and replay."""

from __future__ import annotations

from support import ScenarioBuilder, choose_card, resolve_dogma, scenario

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.effects import EffectStatus, load_effect_programs
from innovation_ai.innovation.state import state_hash
from innovation_ai.innovation.types import CardId, Color, PlayerId

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()
PROGRAMS = load_effect_programs()


def _solo() -> ScenarioBuilder:
    return (
        scenario(REGISTRY)
        .board(P1, Color.BLUE, ("physics",))
        .board(P2, Color.YELLOW, ("agriculture",))
    )


def test_three_distinct_colours_are_kept_and_become_private_again() -> None:
    state = _solo().supply(6, ("classification", "democracy", "atomic-theory")).build()
    result = resolve_dogma(state, "physics", registry=REGISTRY, programs=PROGRAMS)
    assert result.status is EffectStatus.COMPLETE
    assert result.decisions == ()
    assert set(result.state.player(P1).hand) == {
        CardId("classification"),
        CardId("democracy"),
        CardId("atomic-theory"),
    }
    assert result.state.revealed == ()


def test_a_duplicate_colour_returns_the_drawn_cards_and_the_entire_prior_hand() -> None:
    state = (
        _solo()
        .hand(P1, ("tools", "canning"))
        .supply(6, ("classification", "metric-system", "democracy"))
        .build()
    )
    result = resolve_dogma(
        state,
        "physics",
        choose_card("tools"),
        choose_card("canning"),
        choose_card("classification"),
        choose_card("democracy"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    assert not result.state.player(P1).hand
    assert result.state.revealed == ()
    assert all(decision.chooser is P1 for decision in result.decisions)
    assert len(result.decisions) == 4


def test_same_age_returns_are_ordered_by_the_executor_not_by_subset_canonicalization() -> None:
    state = _solo().supply(6, ("classification", "metric-system", "democracy")).build()
    result = resolve_dogma(
        state,
        "physics",
        choose_card("metric-system"),
        choose_card("classification"),
        registry=REGISTRY,
        programs=PROGRAMS,
    )
    pile = result.state.supply.pile(6)
    assert pile.index(CardId("metric-system")) < pile.index(CardId("classification"))
    assert pile.index(CardId("classification")) < pile.index(CardId("democracy"))


def test_the_duplicate_branch_replays_identically_across_every_decision_boundary() -> None:
    state = _solo().supply(6, ("classification", "metric-system", "democracy")).build()
    choices = (choose_card("classification"), choose_card("democracy"))
    first = resolve_dogma(
        state, "physics", *choices, registry=REGISTRY, programs=PROGRAMS, verify_resume=True
    )
    second = resolve_dogma(
        state, "physics", *choices, registry=REGISTRY, programs=PROGRAMS, verify_resume=True
    )
    assert state_hash(first.state) == state_hash(second.state)
    assert first.decisions == second.decisions


def test_a_stronger_opponent_shares_the_whole_effect_first() -> None:
    state = (
        scenario(REGISTRY)
        .board(P1, Color.BLUE, ("physics",))
        .board(P2, Color.PURPLE, ("philosophy",))
        .supply(
            6,
            (
                "classification",
                "democracy",
                "atomic-theory",
                "canning",
                "emancipation",
                "industrialization",
            ),
        )
        .supply(5, ("banking",))
        .build()
    )
    result = resolve_dogma(state, "physics", registry=REGISTRY, programs=PROGRAMS)
    assert result.decisions == ()
    assert set(result.state.player(P2).hand) == {
        CardId("classification"),
        CardId("democracy"),
        CardId("atomic-theory"),
    }
    assert set(result.state.player(P1).hand) == {
        CardId("canning"),
        CardId("emancipation"),
        CardId("industrialization"),
        CardId("banking"),
    }
