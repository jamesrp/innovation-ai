"""Reusable invariant checks for authoritative states, observations, and transitions.

The checks in this module deliberately sit above the engine primitives.  They are useful from
focused tests, deterministic fuzzers, runners, and replay verification without becoming another
mutation path in the rules engine.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

from innovation_ai.innovation.actions import (
    AchieveAction,
    ChooseStartingMeldAction,
    Decision,
    DogmaAction,
    DrawAction,
    MeldAction,
    SemanticAction,
)
from innovation_ai.innovation.board import (
    covered_visible_slots,
    highest_top_value,
    score_value,
    top_cards,
    visible_icons_for_stack,
)
from innovation_ai.innovation.catalog import CardRegistry, load_card_registry
from innovation_ai.innovation.observations import InformationPolicy, observe
from innovation_ai.innovation.protocol import (
    EngineInvariantError,
    Transition,
    apply_action,
    current_decisions,
)
from innovation_ai.innovation.state import (
    GamePhase,
    GameState,
    clone_state,
    state_hash,
)
from innovation_ai.innovation.types import (
    CardId,
    Color,
    Icon,
    IconSlot,
    NormalAchievementId,
    PlayerId,
    SplayDirection,
)
from innovation_ai.innovation.zones import (
    CardLocation,
    StateInvariantError,
    ZoneOperationError,
    assert_state_invariants,
    cards_at,
    draw_card,
    exchange_cards,
    meld_card,
    move_card,
    rearrange_stack,
    remove_card,
    return_card,
    score_card,
    set_splay,
    tuck_card,
)


class InvariantViolation(AssertionError):
    """A reusable WP10 property check failed."""


_EXPECTED_COVERED_SLOTS: dict[SplayDirection, tuple[IconSlot, ...]] = {
    SplayDirection.NONE: (),
    SplayDirection.LEFT: (IconSlot.BOTTOM_RIGHT,),
    SplayDirection.RIGHT: (IconSlot.TOP_LEFT, IconSlot.BOTTOM_LEFT),
    SplayDirection.UP: (
        IconSlot.BOTTOM_LEFT,
        IconSlot.BOTTOM_CENTER,
        IconSlot.BOTTOM_RIGHT,
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvariantViolation(message)


def _card_occurrences(state: GameState) -> tuple[tuple[CardId, CardLocation], ...]:
    occurrences: list[tuple[CardId, CardLocation]] = []
    for age in range(1, 11):
        location = CardLocation.supply(age)
        occurrences.extend((card_id, location) for card_id in cards_at(state, location))
    for achievement_id in NormalAchievementId:
        location = CardLocation.normal_achievement(achievement_id)
        occurrences.extend((card_id, location) for card_id in cards_at(state, location))
    for player_id in PlayerId:
        locations = (
            CardLocation.hand(player_id),
            *(CardLocation.board(player_id, color) for color in Color),
            CardLocation.score(player_id),
        )
        for location in locations:
            occurrences.extend((card_id, location) for card_id in cards_at(state, location))
    removed = CardLocation.removed()
    occurrences.extend((card_id, removed) for card_id in cards_at(state, removed))
    return tuple(occurrences)


def assert_unique_card_locations(state: GameState) -> None:
    """Require every present card identity to occur in exactly one authoritative location."""

    occurrences = _card_occurrences(state)
    counts = Counter(card_id for card_id, _ in occurrences)
    duplicates = tuple(
        sorted((card_id for card_id, count in counts.items() if count != 1), key=str)
    )
    _require(not duplicates, f"cards do not have unique locations: {duplicates}")


def assert_card_conservation(state: GameState, registry: CardRegistry | None = None) -> None:
    """Require the authoritative zones to contain the complete catalog exactly once."""

    registry = registry or load_card_registry()
    occurrences = _card_occurrences(state)
    actual = Counter(card_id for card_id, _ in occurrences)
    expected = Counter(registry.by_id.keys())
    missing = tuple(sorted((expected - actual).elements(), key=str))
    extra = tuple(sorted((actual - expected).elements(), key=str))
    _require(
        not missing and not extra,
        f"card conservation failed; missing={missing}, extra={extra}",
    )


def assert_score_consistency(state: GameState, registry: CardRegistry | None = None) -> None:
    """Require score queries and both player observations to agree with score-pile cards."""

    registry = registry or load_card_registry()
    observations = {viewer: observe(state, viewer, registry) for viewer in PlayerId}
    for player in state.players:
        expected_values = tuple(sorted(registry.card(card_id).age for card_id in player.score_pile))
        _require(
            score_value(player, registry) == sum(expected_values),
            f"score total is inconsistent for {player.player_id}",
        )
        for viewer, observation in observations.items():
            score = observation.player(player.player_id).score_pile
            _require(
                score.values == expected_values,
                f"public score values are inconsistent for {player.player_id}",
            )
            expected_cards = player.score_pile if viewer is player.player_id else ()
            _require(
                score.known_cards == expected_cards,
                f"score identity visibility is inconsistent for viewer {viewer}",
            )


def _expected_stack_icons(
    state: GameState, player_id: PlayerId, color: Color, registry: CardRegistry
) -> Counter[Icon]:
    stack = state.player(player_id).board.stack(color)
    expected: Counter[Icon] = Counter()
    if stack.top is None:
        return expected
    expected.update(registry.card(stack.top).functional_icons)
    for card_id in stack.cards[:-1]:
        card = registry.card(card_id)
        expected.update(
            icon
            for slot in _EXPECTED_COVERED_SLOTS[stack.splay]
            if (icon := card.icon_at(slot)) is not None
        )
    return expected


def assert_icon_geometry(state: GameState, registry: CardRegistry | None = None) -> None:
    """Require catalog faces, stack splaying, icon queries, and owner observations to agree."""

    registry = registry or load_card_registry()
    for direction, expected_slots in _EXPECTED_COVERED_SLOTS.items():
        _require(
            covered_visible_slots(direction) == expected_slots,
            f"covered icon slots are wrong for {direction}",
        )
    for card in registry.cards:
        face = card.face_symbols
        _require(len(face) == 4, f"card {card.id} does not have four geometry slots")
        _require(face.count(None) == 1, f"card {card.id} does not have one image slot")
        _require(
            tuple(icon for icon in face if icon is not None) == card.functional_icons,
            f"card {card.id} functional icons disagree with geometry",
        )
        _require(
            card.featured_icon in card.functional_icons,
            f"card {card.id} featured icon is absent from its face",
        )

    owner_observations = {
        player_id: observe(state, player_id, registry).player(player_id) for player_id in PlayerId
    }
    for player_id in PlayerId:
        player = state.player(player_id)
        observed_player = owner_observations[player_id]
        for color, stack_observation in zip(Color, observed_player.board, strict=True):
            stack = player.board.stack(color)
            _require(
                (len(stack.cards) >= 2 or stack.splay is SplayDirection.NONE),
                f"small {player_id} {color} stack is splayed",
            )
            _require(
                visible_icons_for_stack(stack, registry)
                == _expected_stack_icons(state, player_id, color, registry),
                f"visible icon query is inconsistent for {player_id} {color}",
            )
            _require(stack_observation.top_card_id == stack.top, "owner top-card view is wrong")
            _require(
                stack_observation.covered_count == max(0, len(stack.cards) - 1),
                "owner covered-card count is wrong",
            )
            _require(
                tuple(card.card_id for card in stack_observation.covered_cards) == stack.cards[:-1],
                "owner covered-card order is wrong",
            )


def assert_turn_consistency(state: GameState) -> None:
    """Require phase, active-player, paid-action, and pending-effect fields to cohere."""

    _require(
        max(state.starting_meld_decision_ids) < state.next_decision_id,
        "next decision ID does not follow setup decision IDs",
    )
    if state.phase is GamePhase.STARTING_MELDS:
        _require(state.active_player is None, "setup cannot have an active player")
        _require(state.turn_number == 0, "setup must be turn zero")
        _require(state.paid_actions_remaining == 0, "setup cannot have paid actions")
        _require(not state.pending_effects, "setup cannot have pending effects")
        return
    if state.phase is GamePhase.PLAY:
        _require(state.active_player is not None, "play requires an active player")
        _require(state.turn_number >= 1, "play requires a positive turn number")
        _require(0 <= state.paid_actions_remaining <= 2, "paid actions must be in range 0-2")
        if state.turn_number == 1:
            _require(state.paid_actions_remaining <= 1, "the first turn has only one action")
        if not state.pending_effects:
            _require(state.paid_actions_remaining >= 1, "idle play state has no paid action")
        return
    _require(state.phase is GamePhase.TERMINAL, f"unknown phase: {state.phase}")
    _require(state.terminal_result is not None, "terminal phase has no result")
    _require(not state.pending_effects, "terminal state retains pending effects")


def _eligible_achievements(
    state: GameState, player_id: PlayerId, registry: CardRegistry
) -> tuple[NormalAchievementId, ...]:
    claimed = {
        achievement for player in state.players for achievement in player.normal_achievements
    }
    player = state.player(player_id)
    score = sum(registry.card(card_id).age for card_id in player.score_pile)
    top_value = highest_top_value(player.board, registry)
    return tuple(
        achievement_id
        for age, achievement_id in enumerate(NormalAchievementId, start=1)
        if achievement_id not in claimed and score >= age * 5 and top_value >= age
    )


def assert_legal_action_completeness(
    state: GameState,
    registry: CardRegistry | None = None,
    *,
    decisions: tuple[Decision, ...] | None = None,
) -> None:
    """Require current WP3 decisions to enumerate every and only legal semantic action.

    Pending effect frames intentionally have no decision API yet.  WP4 must extend this check when
    effect choices become available rather than weakening the paid-action checks here.
    """

    registry = registry or load_card_registry()
    actual = current_decisions(state, registry) if decisions is None else decisions
    if state.phase is GamePhase.TERMINAL or state.pending_effects:
        _require(not actual, "terminal or pending state exposes a decision")
        return
    if state.phase is GamePhase.STARTING_MELDS:
        expected_choosers = tuple(
            player_id
            for player_id, choice in zip(PlayerId, state.starting_meld_choices, strict=True)
            if choice is None
        )
        _require(
            tuple(decision.chooser for decision in actual) == expected_choosers,
            "starting-meld chooser set is incomplete",
        )
        for decision in actual:
            player_id = decision.chooser
            index = tuple(PlayerId).index(player_id)
            expected = tuple(
                ChooseStartingMeldAction(state.starting_meld_decision_ids[index], card_id)
                for card_id in state.player(player_id).hand
            )
            _require(
                decision.legal_actions == expected,
                f"starting actions are wrong for {player_id}",
            )
            _require(
                decision.observation == observe(state, player_id, registry),
                f"starting observation is stale for {player_id}",
            )
        return

    _require(len(actual) == 1, "play must expose exactly one paid-action decision")
    decision = actual[0]
    active_player = state.active_player
    if active_player is None:
        raise InvariantViolation("play decision has no active player")
    decision_id = state.next_decision_id
    player = state.player(active_player)
    expected_actions: tuple[SemanticAction, ...] = (
        DrawAction(decision_id),
        *(MeldAction(decision_id, card_id) for card_id in player.hand),
        *(DogmaAction(decision_id, card_id) for card_id in top_cards(player.board)),
        *(
            AchieveAction(decision_id, achievement_id)
            for achievement_id in _eligible_achievements(state, active_player, registry)
        ),
    )
    _require(decision.chooser is active_player, "paid-action chooser is not the active player")
    _require(decision.executor is active_player, "paid-action executor is not the active player")
    _require(decision.legal_actions == expected_actions, "paid legal-action set is incomplete")
    _require(
        decision.observation == observe(state, active_player, registry),
        "paid-action observation is stale",
    )


def assert_observation_leak_resistance(
    first: GameState,
    second: GameState,
    viewer: PlayerId,
    registry: CardRegistry | None = None,
    *,
    policy: InformationPolicy | None = None,
) -> None:
    """Require hidden-equivalent authoritative states to yield equal viewer observations."""

    registry = registry or load_card_registry()
    for candidate in (first, second):
        assert_unique_card_locations(candidate)
        assert_card_conservation(candidate, registry)
        try:
            assert_state_invariants(candidate, registry)
        except StateInvariantError as error:
            raise InvariantViolation(str(error)) from error
    _require(
        state_hash(first) != state_hash(second),
        "leak check needs distinct authoritative states",
    )
    first_observation = observe(first, viewer, registry, policy=policy)
    second_observation = observe(second, viewer, registry, policy=policy)
    _require(
        first_observation == second_observation,
        f"hidden authoritative information leaked to {viewer}",
    )


def assert_transition_purity(before: GameState, snapshot: GameState) -> None:
    """Require a transition call to have left its input state observably unchanged."""

    _require(before == snapshot, "transition mutated its input state")
    _require(state_hash(before) == state_hash(snapshot), "transition changed the input state hash")


def assert_turn_progression(
    before: GameState, action: SemanticAction, transition: Transition
) -> None:
    """Require monotonic IDs and paid-turn fields to progress exactly once."""

    after = transition.state
    if isinstance(action, ChooseStartingMeldAction):
        _require(
            after.next_decision_id == before.next_decision_id,
            "setup advanced decision counter",
        )
        _require(
            after.next_dogma_action_id == before.next_dogma_action_id,
            "setup advanced dogma ID",
        )
        if after.phase is GamePhase.PLAY:
            _require(after.turn_number == 1, "setup did not enter turn one")
            _require(after.paid_actions_remaining == 1, "first player did not receive one action")
        else:
            _require(after.turn_number == 0, "partial setup advanced the turn")
        return

    _require(after.next_decision_id == before.next_decision_id + 1, "decision ID did not advance")
    expected_dogma_id = before.next_dogma_action_id + int(isinstance(action, DogmaAction))
    _require(after.next_dogma_action_id == expected_dogma_id, "dogma action ID progressed wrongly")
    _require(after.next_event_id >= before.next_event_id, "event ID moved backwards")
    _require(after.turn_number >= before.turn_number, "turn number moved backwards")
    _require(after.turn_number <= before.turn_number + 1, "transition skipped a turn")

    if after.phase is GamePhase.TERMINAL or transition.effect_resolution_pending:
        _require(
            after.active_player is before.active_player,
            "unfinished action changed active player",
        )
        _require(
            after.paid_actions_remaining == before.paid_actions_remaining - 1,
            "unfinished or terminal action consumed the wrong number of paid actions",
        )
    elif after.active_player is before.active_player:
        _require(after.turn_number == before.turn_number, "same player advanced the turn")
        _require(
            after.paid_actions_remaining == before.paid_actions_remaining - 1,
            "paid action did not decrement exactly once",
        )
    else:
        _require(after.turn_number == before.turn_number + 1, "player switch did not advance turn")
        _require(after.paid_actions_remaining == 2, "new turn did not receive two paid actions")


def assert_transition_consistency(
    before: GameState,
    action: SemanticAction,
    transition: Transition,
    registry: CardRegistry | None = None,
) -> None:
    """Validate one completed public transition and its resulting state."""

    registry = registry or load_card_registry()
    _require(transition.state is not before, "a legal action returned its input state object")
    assert_turn_progression(before, action, transition)
    assert_state_properties(transition.state, registry)
    if transition.decision is not None:
        decisions = current_decisions(transition.state, registry)
        _require(transition.decision in decisions, "transition returned a stale next decision")
    if transition.terminal is not None:
        _require(
            transition.terminal == transition.state.terminal_result,
            "transition terminal result disagrees with state",
        )


def checked_apply_action(
    state: GameState,
    action: SemanticAction,
    registry: CardRegistry | None = None,
) -> Transition:
    """Apply one action while checking transition purity and all available WP10 properties."""

    registry = registry or load_card_registry()
    snapshot = clone_state(state)
    transition = apply_action(state, action, registry)
    assert_transition_purity(state, snapshot)
    assert_transition_consistency(state, action, transition, registry)
    return transition


def assert_terminal_immutability(state: GameState, registry: CardRegistry | None = None) -> None:
    """Require all current public mutation entry points to reject a terminal state purely."""

    registry = registry or load_card_registry()
    _require(state.phase is GamePhase.TERMINAL, "terminal immutability needs a terminal state")
    snapshot = clone_state(state)
    card_id = next(card_id for card_id, _ in _card_occurrences(state))
    color = registry.card(card_id).color
    operations: tuple[Callable[[], object], ...] = (
        lambda: apply_action(state, DrawAction(state.next_decision_id), registry),
        lambda: draw_card(state, 1, PlayerId.PLAYER_1, registry),
        lambda: move_card(state, card_id, CardLocation.hand(PlayerId.PLAYER_1), registry),
        lambda: meld_card(state, PlayerId.PLAYER_1, card_id, registry),
        lambda: tuck_card(state, PlayerId.PLAYER_1, card_id, registry),
        lambda: score_card(state, PlayerId.PLAYER_1, card_id, registry),
        lambda: return_card(state, card_id, registry),
        lambda: remove_card(state, card_id, registry),
        lambda: exchange_cards(
            state,
            CardLocation.hand(PlayerId.PLAYER_1),
            (),
            CardLocation.hand(PlayerId.PLAYER_2),
            (),
            registry,
        ),
        lambda: set_splay(state, PlayerId.PLAYER_1, color, SplayDirection.LEFT, registry),
        lambda: rearrange_stack(state, PlayerId.PLAYER_1, color, (), registry),
    )
    for operation in operations:
        try:
            operation()
        except (EngineInvariantError, ZoneOperationError):
            pass
        else:
            raise InvariantViolation("a terminal state accepted a mutation")
        assert_transition_purity(state, snapshot)


def assert_state_properties(state: GameState, registry: CardRegistry | None = None) -> None:
    """Run all state-local WP10 properties available through the WP1-WP3 APIs."""

    registry = registry or load_card_registry()
    assert_unique_card_locations(state)
    assert_card_conservation(state, registry)
    try:
        assert_state_invariants(state, registry)
    except StateInvariantError as error:
        raise InvariantViolation(str(error)) from error
    assert_score_consistency(state, registry)
    assert_icon_geometry(state, registry)
    assert_turn_consistency(state)
    assert_legal_action_completeness(state, registry)
