from __future__ import annotations

from dataclasses import replace

import pytest

from innovation_ai.innovation import (
    CardId,
    Color,
    ExplicitPlayerPosition,
    PlayerId,
    build_explicit_state,
    build_setup_state,
)
from innovation_ai.innovation.actions import DogmaAction
from innovation_ai.innovation.observations import observe
from innovation_ai.innovation.protocol import apply_action, current_decision, current_decisions
from innovation_ai.innovation.state import EffectVariable, SupplyState
from innovation_ai.search import (
    HiddenCardDomainKind,
    InformationSetSampler,
    InformationSetSpecBuilder,
    InformationSetSpecError,
    UnsupportedInformationSet,
    verify_sampled_state,
)


def _turn_state():
    return build_explicit_state(
        positions=(
            (
                PlayerId.PLAYER_1,
                ExplicitPlayerPosition(hand=(CardId("city-states"),)),
            ),
            (
                PlayerId.PLAYER_2,
                ExplicitPlayerPosition(
                    hand=(CardId("archery"),),
                    score_pile=(CardId("metalworking"),),
                    board=((Color.BLUE, (CardId("pottery"), CardId("tools"))),),
                ),
            ),
        ),
        active_player=PlayerId.PLAYER_1,
        next_decision_id=41,
        next_event_id=19,
        next_dogma_action_id=13,
    )


def test_turn_action_spec_and_sample_are_hidden_allocation_invariant() -> None:
    first = _turn_state()
    hidden = first.player(PlayerId.PLAYER_2).hand[0]
    replacement = next(card for card in first.supply.pile(1) if card != hidden)
    opponent = replace(first.player(PlayerId.PLAYER_2), hand=(replacement,))
    pile = tuple(hidden if card == replacement else card for card in first.supply.pile(1))
    second = replace(
        first.replace_player(opponent),
        supply=SupplyState((pile, *first.supply.piles[1:])),
    )
    first_decision = current_decision(first)
    second_decision = current_decision(second)
    assert first_decision is not None and second_decision is not None

    builder = InformationSetSpecBuilder()
    first_spec = builder.build(first, first_decision)
    second_spec = builder.build(second, second_decision)

    assert first_spec == second_spec
    assert first_spec.digest == second_spec.digest
    first_sample = InformationSetSampler().sample(first_spec, "same-seed")
    second_sample = InformationSetSampler().sample(second_spec, "same-seed")
    assert first_sample == second_sample
    assert first_sample is not None
    verify_sampled_state(first_spec, first_sample)
    assert observe(first_sample, first_spec.chooser) == first_spec.observation


def test_starting_meld_supports_both_pending_and_one_latent_commitment() -> None:
    state = build_setup_state(71)
    decisions = current_decisions(state)
    assert len(decisions) == 2
    both_spec = InformationSetSpecBuilder().build(state, decisions[0])
    both_sample = InformationSetSampler().sample(both_spec, 9)
    assert both_sample is not None
    assert len(current_decisions(both_sample)) == 2

    transition = apply_action(state, decisions[0].legal_actions[0])
    assert transition.decision is not None
    committed = transition.state.starting_meld_choices[0]
    assert committed is not None
    other = next(
        card for card in transition.state.player(PlayerId.PLAYER_1).hand if card != committed
    )
    hidden_equivalent = replace(
        transition.state,
        starting_meld_choices=(other, None),
    )
    equivalent_decision = current_decisions(hidden_equivalent)[0]

    builder = InformationSetSpecBuilder()
    first_spec = builder.build(transition.state, transition.decision)
    second_spec = builder.build(hidden_equivalent, equivalent_decision)
    assert first_spec == second_spec
    assert len(first_spec.hidden_card_tokens) == 1
    assert first_spec.hidden_card_tokens[0].domain.kind is HiddenCardDomainKind.OPPONENT_HAND

    first_sample = InformationSetSampler().sample(first_spec, "latent")
    second_sample = InformationSetSampler().sample(second_spec, "latent")
    assert first_sample == second_sample
    assert first_sample is not None
    assert first_sample.starting_meld_choices[0] in first_sample.player(PlayerId.PLAYER_1).hand
    verify_sampled_state(first_spec, first_sample)


def test_real_effect_choice_runtime_round_trips_and_preserves_hidden_aliases() -> None:
    state = build_explicit_state(
        positions=(
            (
                PlayerId.PLAYER_1,
                ExplicitPlayerPosition(
                    hand=(CardId("agriculture"), CardId("clothing"), CardId("writing")),
                    board=((Color.BLUE, (CardId("tools"),)),),
                ),
            ),
            (
                PlayerId.PLAYER_2,
                ExplicitPlayerPosition(board=((Color.RED, (CardId("archery"),)),)),
            ),
        ),
        next_decision_id=80,
    )
    turn = current_decision(state)
    assert turn is not None
    dogma = next(action for action in turn.legal_actions if isinstance(action, DogmaAction))
    started = apply_action(state, dogma)
    assert started.decision is not None

    # Exercise generic hidden runtime tokenization on a real, resumable production effect.  The
    # extra namespaced value is ignored by the program but is legitimate serializable runtime.
    hidden = started.state.supply.pile(2)[0]
    with_aliases = replace(
        started.state,
        effect_variables=(
            *started.state.effect_variables,
            EffectVariable("search-test:hidden-alias", (hidden.value, hidden.value)),
        ),
    )
    exact = current_decision(with_aliases)
    assert exact == started.decision

    spec = InformationSetSpecBuilder().build(with_aliases, exact)
    assert len(spec.hidden_card_tokens) == 1
    assert spec.hidden_card_tokens[0].domain.kind is HiddenCardDomainKind.SUPPLY
    sampled = InformationSetSampler(retry_limit=8).sample(spec, "effect-seed")
    assert sampled is not None
    value = next(
        item.value for item in sampled.effect_variables if item.name == "search-test:hidden-alias"
    )
    assert isinstance(value, tuple) and value[0] == value[1]
    assert CardId(value[0]) in sampled.supply.pile(2)
    verify_sampled_state(spec, sampled)


def test_builder_rejects_legacy_policy_and_stale_decision() -> None:
    state = _turn_state()
    decision = current_decision(state)
    assert decision is not None
    legacy = replace(state, information_policy_version="rulebook-private-covered-v1")
    with pytest.raises(UnsupportedInformationSet, match="public-covered"):
        InformationSetSpecBuilder().build(legacy, current_decisions(legacy)[0])
    stale = replace(decision, legal_actions=tuple(reversed(decision.legal_actions)))
    with pytest.raises(InformationSetSpecError, match="exact live decision"):
        InformationSetSpecBuilder().build(state, stale)
