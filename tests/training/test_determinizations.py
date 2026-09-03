from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from innovation_ai.harness.afterstates import TrustedCandidateExpander
from innovation_ai.innovation.actions import DrawAction
from innovation_ai.innovation.observations import observe
from innovation_ai.innovation.state import (
    LEGACY_INFORMATION_POLICY_VERSION,
    EffectFrameState,
    ExplicitPlayerPosition,
    GameState,
    SupplyState,
    build_explicit_state,
)
from innovation_ai.innovation.types import CardId, Color, PlayerId, SplayDirection
from innovation_ai.training.determinizations import (
    SYNTHETIC_SETUP_SEED,
    HiddenAllocationKind,
    InformationSetSampler,
    InformationSetSpec,
    InformationSetSpecBuilder,
    UnsupportedInformationSet,
    verify_sampled_state,
)
from innovation_ai.training.encoding import FlatObservationEncoder


class _CountingSampler(InformationSetSampler):
    def __init__(self) -> None:
        super().__init__(seed="count-validations")
        self.validation_calls = 0

    def _validate_spec(self, spec: InformationSetSpec) -> None:
        self.validation_calls += 1
        super()._validate_spec(spec)


def _stable_state(*, splayed: bool = False, paid_actions: int = 2) -> GameState:
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
                    board=(
                        (
                            Color.BLUE,
                            (CardId("pottery"), CardId("tools"), CardId("writing")),
                        ),
                    ),
                    splays=((Color.BLUE, SplayDirection.UP),) if splayed else (),
                ),
            ),
        ),
        active_player=PlayerId.PLAYER_1,
        paid_actions_remaining=paid_actions,
        next_decision_id=41,
        next_event_id=19,
        next_dogma_action_id=13,
    )


def _swap_hidden_next_supply_card(state: GameState) -> GameState:
    age_one = state.supply.pile(1)
    assert len(age_one) >= 2
    return replace(
        state,
        supply=SupplyState(((age_one[1], age_one[0], *age_one[2:]), *state.supply.piles[1:])),
    )


def test_hidden_equivalent_specs_samples_and_candidate_features_are_identical() -> None:
    first = _stable_state()
    second = _swap_hidden_next_supply_card(first)
    builder = InformationSetSpecBuilder()

    first_spec = builder.build(first)
    second_spec = builder.build(second)

    assert first_spec == second_spec
    assert first_spec.digest == second_spec.digest
    sampler = InformationSetSampler(seed="policy-rng-19")
    first_samples = sampler.sample_many(first_spec, 2)
    second_samples = sampler.sample_many(second_spec, 2)
    assert first_samples == second_samples
    assert all(sample is not None for sample in first_samples)

    sampled_first = tuple(sample for sample in first_samples if sample is not None)
    sampled_second = tuple(sample for sample in second_samples if sample is not None)
    first_expansion = TrustedCandidateExpander().expand(
        first_spec,
        sampled_first,
        game_id="game-1",
        evaluator_key="frozen-a",
    )
    second_expansion = TrustedCandidateExpander().expand(
        second_spec,
        sampled_second,
        game_id="game-1",
        evaluator_key="frozen-a",
    )
    assert first_expansion.routes == second_expansion.routes
    assert first_expansion.terminal_candidates == second_expansion.terminal_candidates
    encoder = FlatObservationEncoder()
    np.testing.assert_array_equal(
        encoder.encode_batch(first_expansion.positions),
        encoder.encode_batch(second_expansion.positions),
    )


def test_sampler_accepts_legacy_policy_for_historical_reproduction() -> None:
    legacy = replace(
        _stable_state(),
        information_policy_version=LEGACY_INFORMATION_POLICY_VERSION,
    )
    spec = InformationSetSpecBuilder().build(legacy)

    sampled = InformationSetSampler(seed="legacy-reproduction").sample(spec)

    assert sampled is not None
    assert sampled.information_policy_version == LEGACY_INFORMATION_POLICY_VERSION
    verify_sampled_state(spec, sampled)


def test_sample_many_validates_shared_spec_once() -> None:
    spec = InformationSetSpecBuilder().build(_stable_state())
    sampler = _CountingSampler()

    samples = sampler.sample_many(spec, 4)

    assert all(sample is not None for sample in samples)
    assert sampler.validation_calls == 1


def test_sample_preserves_observation_legal_actions_splays_and_synthetic_provenance() -> None:
    state = _stable_state(splayed=True)
    spec = InformationSetSpecBuilder().build(state)
    hidden_splayed = tuple(
        item
        for item in spec.hidden_allocation_constraints
        if item.kind is HiddenAllocationKind.OPPONENT_SPLAYED_COVERED
    )
    assert hidden_splayed == ()

    sampled = InformationSetSampler(seed=211).sample(spec)
    assert sampled is not None
    verify_sampled_state(spec, sampled)
    assert sampled.setup.seed == SYNTHETIC_SETUP_SEED
    assert sampled.setup != state.setup
    assert observe(sampled, spec.chooser) == spec.observation
    sampled_stack = sampled.player(PlayerId.PLAYER_2).board.stack(Color.BLUE)
    original_stack = state.player(PlayerId.PLAYER_2).board.stack(Color.BLUE)
    assert sampled_stack.cards == original_stack.cards
    assert sampled_stack.splay is SplayDirection.UP


def test_draw_afterstate_is_reobserved_for_original_chooser_after_turn_rotation() -> None:
    state = _stable_state(paid_actions=1)
    spec = InformationSetSpecBuilder().build(state)
    sampled = InformationSetSampler(seed=312).sample(spec)
    assert sampled is not None
    expansion = TrustedCandidateExpander().expand(
        spec,
        (sampled,),
        game_id="game-2",
        evaluator_key="frozen-b",
    )
    draw_position = next(
        position
        for route, position in zip(expansion.routes, expansion.positions, strict=True)
        if isinstance(route.action, DrawAction)
    )
    assert draw_position.viewer is spec.chooser
    assert draw_position.observation.viewer is spec.chooser
    assert draw_position.boundary.chooser_relation.value == "opponent"


def test_builder_rejects_unstable_effect_boundary() -> None:
    unstable = replace(_stable_state(), pending_effects=(EffectFrameState("test"),))
    with pytest.raises(UnsupportedInformationSet, match="no pending effects"):
        InformationSetSpecBuilder().build(unstable)
