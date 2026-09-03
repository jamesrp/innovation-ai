from __future__ import annotations

from dataclasses import replace

from innovation_ai.innovation import CardId, GameState, PlayerId, build_explicit_state, state_hash
from innovation_ai.innovation.state import EffectFrameState, EffectVariable, SetupProvenance
from innovation_ai.innovation.strategic import (
    is_strategically_equal,
    strategic_state_digest,
    strategic_state_payload,
)


def _runtime_state(turn_id: int, dogma_id: int, *, dogma_alias: int | None = None) -> GameState:
    state = build_explicit_state(
        turn_number=turn_id,
        next_decision_id=100 + turn_id,
        next_event_id=200 + turn_id,
        next_dogma_action_id=dogma_id + 1,
    )
    frame = EffectFrameState(
        "test-frame",
        source_card_id=CardId("agriculture"),
        variables=(
            EffectVariable("dogma_action", dogma_id if dogma_alias is None else dogma_alias),
            EffectVariable("dogma_action_id", dogma_id),
            EffectVariable("program_id", "semantic-program-v1"),
            EffectVariable("turn_id", turn_id),
        ),
    )
    return replace(state, pending_effects=(frame,))


def test_digest_excludes_setup_turn_and_monotonic_top_level_ids() -> None:
    first = build_explicit_state(turn_number=3, next_decision_id=10, next_event_id=20)
    changed_setup = SetupProvenance(
        seed=999,
        card_data_fingerprint=first.setup.card_data_fingerprint,
        shuffled_piles=tuple(tuple(reversed(pile)) for pile in first.setup.shuffled_piles),
        deal_sequence=tuple(reversed(first.setup.deal_sequence)),
        rng_version="different-rng",
    )
    second = replace(
        first,
        turn_number=999,
        starting_meld_decision_ids=(501, 502),
        next_decision_id=700,
        next_event_id=800,
        next_dogma_action_id=900,
        setup=changed_setup,
    )

    assert state_hash(first) != state_hash(second)
    assert strategic_state_digest(first) == strategic_state_digest(second)
    payload = strategic_state_payload(first)
    assert "turn_number" not in payload
    assert "setup" not in payload
    assert all(not key.startswith("next_") for key in payload)


def test_digest_retains_gameplay_state_and_turn_boundary() -> None:
    state = build_explicit_state()
    other_player = replace(
        state.player(PlayerId.PLAYER_1),
        hand=(*state.player(PlayerId.PLAYER_1).hand, CardId("agriculture")),
    )
    changed_cards = state.replace_player(other_player)
    changed_boundary = replace(state, paid_actions_remaining=1)

    assert strategic_state_digest(state) != strategic_state_digest(changed_cards)
    assert strategic_state_digest(state) != strategic_state_digest(changed_boundary)


def test_effect_runtime_normalizes_only_known_monotonic_ids_and_preserves_aliasing() -> None:
    first = _runtime_state(3, 7)
    renumbered = _runtime_state(88, 900)
    different_alias_relationship = _runtime_state(88, 900, dogma_alias=901)

    assert is_strategically_equal(first, renumbered)
    assert strategic_state_digest(first) != strategic_state_digest(different_alias_relationship)
    variables = strategic_state_payload(first)["pending_effects"]
    assert "semantic-program-v1" in str(variables)
