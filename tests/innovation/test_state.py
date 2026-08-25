from __future__ import annotations

from dataclasses import replace

import pytest

from innovation_ai.innovation.catalog import load_card_registry
from innovation_ai.innovation.state import (
    STATE_SCHEMA_VERSION,
    Board,
    ColorStack,
    EffectFrameState,
    EffectVariable,
    GamePhase,
    PlayerState,
    SupplyState,
    build_setup_state,
    build_setup_state_from_piles,
    clone_state,
    state_hash,
    state_payload,
)
from innovation_ai.innovation.types import Color, PlayerId, SplayDirection
from innovation_ai.innovation.zones import assert_state_invariants


def test_setup_constructs_complete_deterministic_authoritative_state() -> None:
    registry = load_card_registry()
    state = build_setup_state(8675309, registry)

    assert state.phase is GamePhase.STARTING_MELDS
    assert state.active_player is None
    assert state.turn_number == 0
    assert state.paid_actions_remaining == 0
    assert tuple(len(player.hand) for player in state.players) == (2, 2)
    assert all(player.board == Board.empty() for player in state.players)
    assert len(state.normal_achievements.cards) == 9
    assert sum(len(pile) for pile in state.supply.piles) == 92
    assert state.setup.deal_sequence == (
        PlayerId.PLAYER_1,
        PlayerId.PLAYER_2,
        PlayerId.PLAYER_1,
        PlayerId.PLAYER_2,
    )
    assert state.setup.card_data_fingerprint == registry.data_fingerprint
    assert state.setup.rng_version == "python-mt19937-shuffle-v1"
    assert (
        build_setup_state_from_piles(
            state.setup.shuffled_piles,
            seed=state.setup.seed,
            registry=registry,
        )
        == state
    )
    assert build_setup_state(8675309, registry) == state
    assert build_setup_state(8675310, registry) != state
    with pytest.raises(ValueError, match="ten shuffled piles"):
        build_setup_state_from_piles((), seed=1, registry=registry)
    assert_state_invariants(state, registry)


def test_state_clone_payload_and_hash_are_deterministic_and_detached() -> None:
    state = build_setup_state(42)
    clone = clone_state(state)

    assert clone == state
    assert clone is not state
    assert clone.players is not state.players
    assert state_hash(clone) == state_hash(state)
    assert state_hash(build_setup_state(42)) == state_hash(state)
    payload = state_payload(state)
    assert payload["schema_version"] == STATE_SCHEMA_VERSION
    assert payload["phase"] == "starting-melds"
    assert payload["setup"]["seed"] == 42  # type: ignore[index]

    changed = replace(state, next_event_id=2)
    assert state_hash(changed) != state_hash(state)


def test_state_hash_supports_serializable_pending_effect_data() -> None:
    state = build_setup_state(7)
    frame = EffectFrameState(
        "repeat",
        step=2,
        source_card_id=state.players[0].hand[0],
        variables=(EffectVariable("choices", ("blue", 3, True, None)),),
    )
    pending = replace(
        state,
        pending_effects=(frame,),
        effect_variables=(EffectVariable("executor", PlayerId.PLAYER_2.value),),
    )

    assert state_hash(pending).startswith("sha256:")
    assert state_payload(pending)["pending_effects"] != []


def test_state_hash_canonicalizes_unordered_player_and_removed_zones() -> None:
    state = build_setup_state(8)
    first = state.players[0]
    reversed_player = PlayerState(
        first.player_id,
        tuple(reversed(first.hand)),
        first.board,
        tuple(reversed(first.score_pile)),
        tuple(reversed(first.normal_achievements)),
        tuple(reversed(first.special_achievements)),
    )
    reversed_state = replace(
        state.replace_player(reversed_player),
        removed_cards=tuple(reversed(state.removed_cards)),
    )
    assert reversed_state == state
    assert state_hash(reversed_state) == state_hash(state)


@pytest.mark.parametrize("size", [0, 1])
def test_small_stack_cannot_be_constructed_splayed(size: int) -> None:
    cards = () if size == 0 else (build_setup_state(1).players[0].hand[0],)
    with pytest.raises(ValueError, match="must be unsplayed"):
        ColorStack(Color.BLUE, cards, SplayDirection.LEFT)


def test_structural_state_validation_rejects_invalid_supply_and_board() -> None:
    with pytest.raises(ValueError, match="ten age piles"):
        SupplyState(())
    with pytest.raises(ValueError, match="canonical color order"):
        Board((ColorStack(Color.BLUE),))

    state = build_setup_state(9)
    duplicate = state.players[0].hand[0]
    with pytest.raises(ValueError, match="hand cannot contain"):
        PlayerState(
            PlayerId.PLAYER_1,
            (*state.players[0].hand, duplicate),
            state.players[0].board,
            (),
        )
