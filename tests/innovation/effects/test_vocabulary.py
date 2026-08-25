"""The expanded declarative vocabulary: selectors, quantities, predicates, and nodes.

These are contract tests for the shared primitives every card program is built from. They are
deliberately table-driven so a new selector field or value kind has one obvious place to go.
"""

from __future__ import annotations

import pytest

from innovation_ai.innovation.actions import ChooseColorAction
from innovation_ai.innovation.catalog import CardRegistry, load_card_registry
from innovation_ai.innovation.effects import (
    ACTIVATOR,
    ALL_OTHER_PLAYERS,
    ALL_PLAYERS,
    EXECUTOR,
    OPPONENT,
    AllOrNoneNode,
    BatchNode,
    CardSelector,
    CardSelectorKind,
    ChoiceColorSource,
    ChoiceKind,
    ChoiceNode,
    ClaimAchievementNode,
    Cmp,
    ConditionNode,
    DrawNode,
    EffectContext,
    EffectEventKind,
    EffectInvariantError,
    EffectNode,
    EffectProgram,
    EffectProgramRegistry,
    EffectResolution,
    EffectStatus,
    Extreme,
    ExtremeScope,
    ForEachCardNode,
    LetNode,
    MovementKind,
    MoveNode,
    NoOpNode,
    OrderGroup,
    PlayerRef,
    PlayerRefKind,
    Predicate,
    ProgramEffect,
    Rounding,
    SelectorRelation,
    SelectorRelationKind,
    SequenceNode,
    SplayNode,
    StackPosition,
    StopNode,
    TimesNode,
    ValueRef,
    ValueRefKind,
    WinMetric,
    WinMode,
    WinNode,
    evaluate_predicate,
    resolve_player,
    resolve_players,
    resolve_value,
    select_cards,
    set_effect_variable,
    start_effect,
    submit_effect_action,
)
from innovation_ai.innovation.state import (
    ExplicitPlayerPosition,
    GameState,
    TurnCounters,
    build_explicit_state,
)
from innovation_ai.innovation.types import (
    CardId,
    Color,
    DogmaEffectId,
    Icon,
    NormalAchievementId,
    PlayerId,
    SpecialAchievementId,
    SplayDirection,
)
from innovation_ai.innovation.zones import ZoneKind

P1 = PlayerId.PLAYER_1
P2 = PlayerId.PLAYER_2
REGISTRY = load_card_registry()


def _state() -> GameState:
    """A fixed position with a varied board so every selector filter has something to bite on."""

    return build_explicit_state(
        REGISTRY,
        positions=(
            (
                P1,
                ExplicitPlayerPosition(
                    hand=(CardId("tools"), CardId("canal-building"), CardId("construction")),
                    score_pile=(CardId("writing"), CardId("alchemy")),
                    board=(
                        (Color.RED, (CardId("archery"), CardId("metalworking"))),
                        (Color.BLUE, (CardId("pottery"),)),
                        (Color.GREEN, (CardId("the-wheel"), CardId("sailing"))),
                    ),
                    splays=((Color.RED, SplayDirection.RIGHT),),
                ),
            ),
            (
                P2,
                ExplicitPlayerPosition(
                    hand=(CardId("mysticism"),),
                    board=((Color.PURPLE, (CardId("code-of-laws"),)),),
                ),
            ),
        ),
    )


def _context(state: GameState, card: str = "pottery") -> EffectContext:
    return EffectContext(
        actor=P1,
        chooser=P1,
        executor=P1,
        dogma_activator=P1,
        source_card_id=CardId(card),
        source_effect_id=None,
        turn_id=state.turn_number,
        dogma_action_id=1,
    )


# --------------------------------------------------------------------------------------------
# Selectors
# --------------------------------------------------------------------------------------------


def test_a_value_filter_uses_its_comparator() -> None:
    state, context = _state(), _context(_state())
    cases: tuple[tuple[Cmp, int, set[CardId]], ...] = (
        (Cmp.EQ, 1, {CardId("tools")}),
        (Cmp.LE, 1, {CardId("tools")}),
        (Cmp.GE, 2, {CardId("canal-building"), CardId("construction")}),
        (Cmp.GT, 2, set()),
        (Cmp.LT, 2, {CardId("tools")}),
        (Cmp.NE, 1, {CardId("canal-building"), CardId("construction")}),
    )
    for comparator, value, expected in cases:
        selector = CardSelector.hand(EXECUTOR, value=value, value_cmp=comparator)
        assert set(select_cards(state, context, selector, REGISTRY)) == expected


def test_a_value_filter_can_read_a_computed_expression_with_an_offset() -> None:
    state, context = _state(), _context(_state())
    # "a card of value one higher than my top blue card" - pottery is value 1, so value 2.
    selector = CardSelector(
        CardSelectorKind.HAND,
        EXECUTOR,
        value_expr=ValueRef.selector_extreme(
            CardSelector.stack(EXECUTOR, color=Color.BLUE, position=StackPosition.TOP)
        ),
        value_offset=1,
    )
    assert set(select_cards(state, context, selector, REGISTRY)) == {
        CardId("canal-building"),
        CardId("construction"),
    }


def test_colour_include_and_exclude_filters_compose() -> None:
    state, context = _state(), _context(_state())
    tops = CardSelector.top_cards(EXECUTOR, colors=(Color.RED, Color.BLUE))
    assert set(select_cards(state, context, tops, REGISTRY)) == {
        CardId("metalworking"),
        CardId("pottery"),
    }
    non_green = CardSelector.top_cards(EXECUTOR, exclude_colors=(Color.GREEN,))
    assert set(select_cards(state, context, non_green, REGISTRY)) == {
        CardId("metalworking"),
        CardId("pottery"),
    }


def test_icon_and_without_icon_filters_are_complementary() -> None:
    state, context = _state(), _context(_state())
    with_leaf = CardSelector.top_cards(EXECUTOR, icon=Icon.LEAF)
    without_leaf = CardSelector.top_cards(EXECUTOR, without_icon=Icon.LEAF)
    all_tops = set(select_cards(state, context, CardSelector.top_cards(EXECUTOR), REGISTRY))
    assert (
        set(select_cards(state, context, with_leaf, REGISTRY))
        | set(select_cards(state, context, without_leaf, REGISTRY))
        == all_tops
    )


@pytest.mark.parametrize(
    ("position", "expected"),
    [
        (StackPosition.TOP, {CardId("metalworking")}),
        (StackPosition.BOTTOM, {CardId("archery")}),
        (StackPosition.BENEATH_TOP, {CardId("archery")}),
        (StackPosition.ANY, {CardId("archery"), CardId("metalworking")}),
    ],
)
def test_stack_positions_address_top_bottom_and_beneath(
    position: StackPosition, expected: set[CardId]
) -> None:
    state, context = _state(), _context(_state())
    selector = CardSelector.stack(EXECUTOR, color=Color.RED, position=position)
    assert set(select_cards(state, context, selector, REGISTRY)) == expected


def test_beneath_a_named_card_resolves_through_a_variable() -> None:
    """Card text like "score the card beneath it" needs a positional reference, not a literal."""

    state, context = _state(), _context(_state())
    state = set_effect_variable(state, context, "anchor", CardId("metalworking").value)
    selector = CardSelector.stack(
        EXECUTOR,
        color=Color.RED,
        position=StackPosition.BENEATH_VARIABLE,
        position_variable="anchor",
    )
    assert select_cards(state, context, selector, REGISTRY) == (CardId("archery"),)


def test_an_extreme_selector_returns_all_tied_cards_for_a_chooser() -> None:
    state, context = _state(), _context(_state())
    all_tied = CardSelector.hand(
        EXECUTOR, extreme=Extreme.HIGHEST, extreme_scope=ExtremeScope.ALL_TIED
    )
    assert set(select_cards(state, context, all_tied, REGISTRY, for_choice=True)) == {
        CardId("canal-building"),
        CardId("construction"),
    }


def test_one_tied_collapses_to_the_lowest_card_id_when_nobody_chooses() -> None:
    """Decision 13's stable fallback for a direct consumer with no chooser."""

    state, context = _state(), _context(_state())
    one_tied = CardSelector.hand(
        EXECUTOR, extreme=Extreme.HIGHEST, extreme_scope=ExtremeScope.ONE_TIED
    )
    assert select_cards(state, context, one_tied, REGISTRY) == (CardId("canal-building"),)
    # A choice node still sees both candidates so the owner can break the tie.
    assert len(select_cards(state, context, one_tied, REGISTRY, for_choice=True)) == 2


def test_the_lowest_extreme_is_available_too() -> None:
    state, context = _state(), _context(_state())
    lowest = CardSelector.hand(EXECUTOR, extreme=Extreme.LOWEST)
    assert select_cards(state, context, lowest, REGISTRY) == (CardId("tools"),)


def test_relational_selectors_compare_against_another_card_set() -> None:
    state, context = _state(), _context(_state())
    same_colour = CardSelector(
        CardSelectorKind.HAND,
        EXECUTOR,
        relation=SelectorRelation(
            SelectorRelationKind.SAME_COLOR_AS_ANY, CardSelector.board(EXECUTOR)
        ),
    )
    # tools is blue and the board has blue; canal-building is yellow and construction is red.
    assert set(select_cards(state, context, same_colour, REGISTRY)) == {
        CardId("tools"),
        CardId("construction"),
    }
    different_colour = CardSelector(
        CardSelectorKind.HAND,
        EXECUTOR,
        relation=SelectorRelation(
            SelectorRelationKind.DIFFERENT_COLOR_FROM_ALL, CardSelector.board(EXECUTOR)
        ),
    )
    assert set(select_cards(state, context, different_colour, REGISTRY)) == {
        CardId("canal-building")
    }


def test_same_value_relations_work_across_zones() -> None:
    state, context = _state(), _context(_state())
    selector = CardSelector(
        CardSelectorKind.HAND,
        EXECUTOR,
        relation=SelectorRelation(
            SelectorRelationKind.SAME_VALUE_AS_ANY, CardSelector.top_cards(EXECUTOR)
        ),
    )
    # Every top card here is value 1, so only the age 1 hand card qualifies.
    assert set(select_cards(state, context, selector, REGISTRY)) == {CardId("tools")}


def test_a_variable_can_exclude_cards_from_a_selection() -> None:
    state, context = _state(), _context(_state())
    state = set_effect_variable(state, context, "used", (CardId("tools").value,))
    selector = CardSelector(CardSelectorKind.HAND, EXECUTOR, exclude_variable="used")
    assert set(select_cards(state, context, selector, REGISTRY)) == {
        CardId("canal-building"),
        CardId("construction"),
    }


def test_a_scalar_variable_can_exclude_one_card() -> None:
    state, context = _state(), _context(_state())
    state = set_effect_variable(state, context, "used", CardId("tools").value)
    selector = CardSelector(CardSelectorKind.HAND, EXECUTOR, exclude_variable="used")
    assert set(select_cards(state, context, selector, REGISTRY)) == {
        CardId("canal-building"),
        CardId("construction"),
    }


def test_all_players_selectors_span_both_boards_in_canonical_order() -> None:
    state, context = _state(), _context(_state())
    tops = select_cards(state, context, CardSelector.top_cards(ALL_PLAYERS), REGISTRY)
    assert CardId("code-of-laws") in tops
    assert CardId("metalworking") in tops
    others = select_cards(state, context, CardSelector.top_cards(ALL_OTHER_PLAYERS), REGISTRY)
    assert others == (CardId("code-of-laws"),)


def test_a_multi_player_reference_cannot_be_a_single_destination() -> None:
    context = _context(_state())
    with pytest.raises(EffectInvariantError, match="several players"):
        resolve_player(ALL_PLAYERS, context)
    assert resolve_players(ALL_PLAYERS, context) == (P1, P2)
    assert resolve_players(EXECUTOR, context) == (P1,)


def test_a_named_selector_predicate_filters_one_candidate_at_a_time() -> None:
    """The bounded escape hatch stays pure and per-candidate."""

    state, context = _state(), _context(_state())

    def only_tools(state: GameState, context: EffectContext, registry: CardRegistry) -> bool:
        from innovation_ai.innovation.effects import get_effect_variable

        return get_effect_variable(state, context, "_candidate") == "tools"

    program = EffectProgram(
        "named-predicate-v1",
        CardId("pottery"),
        (
            ProgramEffect(DogmaEffectId(CardId("pottery"), 1), False, "noop"),
            ProgramEffect(DogmaEffectId(CardId("pottery"), 2), False, "noop"),
        ),
        (NoOpNode("noop"),),
    )
    programs = EffectProgramRegistry(
        (program,), predicates={CardId("pottery"): {"only-tools": only_tools}}
    )
    selector = CardSelector(CardSelectorKind.HAND, EXECUTOR, predicate="only-tools")
    assert select_cards(state, context, selector, REGISTRY, programs) == (CardId("tools"),)
    with pytest.raises(EffectInvariantError, match="requires the effect program registry"):
        select_cards(state, context, selector, REGISTRY)


# --------------------------------------------------------------------------------------------
# Quantities
# --------------------------------------------------------------------------------------------


def test_icon_counts_use_frozen_free_visible_geometry_and_divide_by_per() -> None:
    state, context = _state(), _context(_state())
    # The red stack is splayed right, so both cards contribute castles.
    total = resolve_value(state, context, ValueRef.icon_count(Icon.CASTLE, EXECUTOR), REGISTRY)
    assert total >= 4
    halved = resolve_value(
        state, context, ValueRef.icon_count(Icon.CASTLE, EXECUTOR, per=2), REGISTRY
    )
    assert halved == total // 2


def test_rounding_up_and_down_are_both_available() -> None:
    state, context = _state(), _context(_state())
    down = ValueRef(
        ValueRefKind.COUNT_SELECTOR,
        selector=CardSelector.hand(EXECUTOR),
        per=2,
        rounding=Rounding.FLOOR,
    )
    up = ValueRef(
        ValueRefKind.COUNT_SELECTOR,
        selector=CardSelector.hand(EXECUTOR),
        per=2,
        rounding=Rounding.CEIL,
    )
    assert resolve_value(state, context, down, REGISTRY) == 1
    assert resolve_value(state, context, up, REGISTRY) == 2


def test_literal_quantities_use_the_same_division_rounding_and_offset_pipeline() -> None:
    state, context = _state(), _context(_state())
    floor = ValueRef(ValueRefKind.LITERAL, value=5, per=2, offset=1)
    ceil = ValueRef(
        ValueRefKind.LITERAL,
        value=5,
        per=2,
        rounding=Rounding.CEIL,
    )
    assert resolve_value(state, context, floor, REGISTRY) == 3
    assert resolve_value(state, context, ceil, REGISTRY) == 3


def test_colours_with_icon_and_colours_splayed_count_stacks_not_cards() -> None:
    state, context = _state(), _context(_state())
    with_castle = ValueRef(ValueRefKind.COLORS_WITH_ICON, icon=Icon.CASTLE, player=EXECUTOR)
    # Only the red stack shows a castle: pottery is all leaves and sailing tops green.
    assert resolve_value(state, context, with_castle, REGISTRY) == 1
    with_leaf = ValueRef(ValueRefKind.COLORS_WITH_ICON, icon=Icon.LEAF, player=EXECUTOR)
    assert resolve_value(state, context, with_leaf, REGISTRY) >= 2
    splayed_right = ValueRef(
        ValueRefKind.COLORS_SPLAYED, direction=SplayDirection.RIGHT, player=EXECUTOR
    )
    assert resolve_value(state, context, splayed_right, REGISTRY) == 1


def test_colours_present_only_here_excludes_shared_colours() -> None:
    state, context = _state(), _context(_state())
    only_mine = ValueRef(ValueRefKind.COLORS_PRESENT_ONLY_HERE, player=EXECUTOR)
    # Red, blue, and green are only on player 1's board; purple is only on player 2's.
    assert resolve_value(state, context, only_mine, REGISTRY) == 3


def test_distinct_values_and_card_value_read_scoped_variables() -> None:
    state, context = _state(), _context(_state())
    state = set_effect_variable(
        state,
        context,
        "returned",
        (CardId("tools").value, CardId("canal-building").value, CardId("construction").value),
    )
    distinct = ValueRef(ValueRefKind.DISTINCT_VALUES, variable="returned")
    assert resolve_value(state, context, distinct, REGISTRY) == 2
    state = set_effect_variable(state, context, "one", CardId("alchemy").value)
    assert resolve_value(state, context, ValueRef.card_value("one"), REGISTRY) == 3
    assert resolve_value(state, context, ValueRef.card_value("one", offset=2), REGISTRY) == 5
    # "the value of something you do not have is zero".
    assert resolve_value(state, context, ValueRef.card_value("absent"), REGISTRY) == 0


def test_score_and_achievement_count_quantities_read_live_state() -> None:
    state, context = _state(), _context(_state())
    score = ValueRef(ValueRefKind.SCORE, player=EXECUTOR)
    # writing is value 1 and alchemy is value 3.
    assert resolve_value(state, context, score, REGISTRY) == 4
    achievements = ValueRef(ValueRefKind.ACHIEVEMENT_COUNT, player=EXECUTOR)
    assert resolve_value(state, context, achievements, REGISTRY) == 0


def test_a_selector_extreme_of_an_empty_set_is_zero() -> None:
    state, context = _state(), _context(_state())
    empty = ValueRef.selector_extreme(
        CardSelector.stack(EXECUTOR, color=Color.YELLOW, position=StackPosition.TOP)
    )
    assert resolve_value(state, context, empty, REGISTRY) == 0


def test_a_named_quantity_requires_the_registry() -> None:
    state, context = _state(), _context(_state())
    named = ValueRef.named("custom")
    with pytest.raises(EffectInvariantError, match="requires the effect program registry"):
        resolve_value(state, context, named, REGISTRY)


# --------------------------------------------------------------------------------------------
# Predicates
# --------------------------------------------------------------------------------------------


def test_a_count_comparison_compares_two_quantities() -> None:
    state, context = _state(), _context(_state())
    predicate = Predicate.count(
        ValueRef.icon_count(Icon.CASTLE, EXECUTOR), Cmp.GE, ValueRef.literal(3)
    )
    assert evaluate_predicate(state, context, predicate, REGISTRY)
    assert not evaluate_predicate(
        state,
        context,
        Predicate.count(ValueRef.icon_count(Icon.CASTLE, EXECUTOR), Cmp.GE, ValueRef.literal(99)),
        REGISTRY,
    )


def test_a_universal_card_test_is_true_for_an_empty_candidate_set() -> None:
    """Decision 10, implemented exactly once for every card that needs it."""

    state, context = _state(), _context(_state())
    empty_candidates = CardSelector.stack(EXECUTOR, color=Color.YELLOW, position=StackPosition.TOP)
    predicate = Predicate.all_match(empty_candidates, CardSelector.top_cards(EXECUTOR))
    assert evaluate_predicate(state, context, predicate, REGISTRY)
    # "any" over an empty set is false, which is the correct dual.
    assert not evaluate_predicate(
        state,
        context,
        Predicate.any_match(empty_candidates, CardSelector.top_cards(EXECUTOR)),
        REGISTRY,
    )


def test_a_universal_card_test_detects_one_failing_card() -> None:
    state, context = _state(), _context(_state())
    every_top_has_leaf = Predicate.all_match(
        CardSelector.top_cards(EXECUTOR), CardSelector.top_cards(EXECUTOR, icon=Icon.LEAF)
    )
    assert not evaluate_predicate(state, context, every_top_has_leaf, REGISTRY)
    only_blue = Predicate.all_match(
        CardSelector.stack(EXECUTOR, color=Color.BLUE, position=StackPosition.TOP),
        CardSelector.top_cards(EXECUTOR, icon=Icon.LEAF),
    )
    assert evaluate_predicate(state, context, only_blue, REGISTRY)


def test_selector_non_empty_expresses_a_generic_if_you_do() -> None:
    state, context = _state(), _context(_state())
    assert evaluate_predicate(
        state, context, Predicate.non_empty(CardSelector.hand(EXECUTOR)), REGISTRY
    )
    assert not evaluate_predicate(
        state,
        context,
        Predicate.non_empty(
            CardSelector.stack(EXECUTOR, color=Color.YELLOW, position=StackPosition.TOP)
        ),
        REGISTRY,
    )


def test_a_colour_set_predicate_can_read_a_variable_colour_pair() -> None:
    """Empiricism-style text compares against two *chosen* colours, not literals."""

    state, context = _state(), _context(_state())
    state = set_effect_variable(state, context, "card", CardId("pottery").value)
    literal = Predicate.card_color_in("card", (Color.BLUE, Color.GREEN))
    assert evaluate_predicate(state, context, literal, REGISTRY)

    state = set_effect_variable(state, context, "chosen", (Color.RED.value, Color.GREEN.value))
    from_variable = Predicate.card_color_in("card", colors_variable="chosen")
    assert not evaluate_predicate(state, context, from_variable, REGISTRY)
    state = set_effect_variable(state, context, "chosen", Color.BLUE.value)
    assert evaluate_predicate(state, context, from_variable, REGISTRY)


def test_a_card_predicate_on_a_non_card_variable_raises_instead_of_silently_failing() -> None:
    state, context = _state(), _context(_state())
    state = set_effect_variable(state, context, "number", 4)
    with pytest.raises(EffectInvariantError, match="not a card ID"):
        evaluate_predicate(state, context, Predicate.card_has_icon("number", Icon.LEAF), REGISTRY)


def test_negation_wraps_any_predicate() -> None:
    state, context = _state(), _context(_state())
    inner = Predicate.non_empty(CardSelector.hand(EXECUTOR))
    assert not evaluate_predicate(state, context, Predicate.negate(inner), REGISTRY)


def test_a_named_predicate_requires_the_registry() -> None:
    state, context = _state(), _context(_state())
    with pytest.raises(EffectInvariantError, match="requires the effect program registry"):
        evaluate_predicate(state, context, Predicate.named("custom"), REGISTRY)


# --------------------------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------------------------


def _run(
    program: EffectProgram,
    state: GameState,
    *,
    card: str = "pottery",
) -> EffectResolution:
    programs = EffectProgramRegistry((program,))
    return start_effect(state, program.program_id, _context(state, card), programs, REGISTRY)


def _single_effect_program(
    program_id: str, root: str, nodes: tuple[EffectNode, ...]
) -> EffectProgram:
    card = CardId("the-wheel")
    return EffectProgram(
        program_id,
        card,
        (ProgramEffect(DogmaEffectId(card, 1), False, root),),
        nodes,
    )


def test_a_times_node_repeats_a_body_a_computed_number_of_times() -> None:
    program = _single_effect_program(
        "times-v1",
        "times",
        (
            TimesNode("times", ValueRef.literal(3), "draw", index_variable="index"),
            DrawNode("draw", ValueRef.literal(1), "drawn", player=EXECUTOR),
        ),
    )
    state = _state()
    before = len(state.player(P1).hand)
    result = _run(program, state, card="the-wheel")
    assert result.status is EffectStatus.COMPLETE
    assert len(result.state.player(P1).hand) == before + 3


def test_a_times_node_with_a_zero_count_does_nothing() -> None:
    program = _single_effect_program(
        "times-zero-v1",
        "times",
        (
            TimesNode("times", ValueRef.literal(0), "draw"),
            DrawNode("draw", ValueRef.literal(1), "drawn", player=EXECUTOR),
        ),
    )
    state = _state()
    result = _run(program, state, card="the-wheel")
    assert result.state.player(P1).hand == state.player(P1).hand


def test_a_times_node_fails_loudly_above_its_ceiling() -> None:
    program = _single_effect_program(
        "times-ceiling-v1",
        "times",
        (
            TimesNode("times", ValueRef.literal(5), "draw", maximum_iterations=2),
            DrawNode("draw", ValueRef.literal(1), "drawn", player=EXECUTOR),
        ),
    )
    with pytest.raises(EffectInvariantError, match="exceeding"):
        _run(program, _state(), card="the-wheel")


def test_a_for_each_card_node_iterates_a_variable_in_stored_order() -> None:
    program = _single_effect_program(
        "foreach-v1",
        "sequence",
        (
            SequenceNode("sequence", ("bind", "foreach")),
            LetNode("bind", "cards", cards=CardSelector.hand(EXECUTOR)),
            ForEachCardNode("foreach", "cards", "one", "score"),
            MoveNode(
                "score",
                MovementKind.SCORE,
                CardSelector.from_variable("one"),
                destination_player=EXECUTOR,
            ),
        ),
    )
    state = _state()
    result = _run(program, state, card="the-wheel")
    assert result.status is EffectStatus.COMPLETE
    assert not result.state.player(P1).hand
    assert set(state.player(P1).hand) <= set(result.state.player(P1).score_pile)


def test_an_all_or_none_node_never_performs_a_partial_prefix() -> None:
    program = _single_effect_program(
        "all-or-none-v1",
        "guarded",
        (
            AllOrNoneNode(
                "guarded",
                Predicate.count(
                    ValueRef(ValueRefKind.COUNT_SELECTOR, selector=CardSelector.hand(EXECUTOR)),
                    Cmp.GE,
                    ValueRef.literal(99),
                ),
                "draw",
            ),
            DrawNode("draw", ValueRef.literal(1), "drawn", player=EXECUTOR),
        ),
    )
    state = _state()
    result = _run(program, state, card="the-wheel")
    assert result.state.player(P1).hand == state.player(P1).hand


def test_a_stop_node_ends_only_the_current_printed_effect() -> None:
    card = CardId("pottery")
    program = EffectProgram(
        "stop-v1",
        card,
        (
            ProgramEffect(DogmaEffectId(card, 1), False, "first"),
            ProgramEffect(DogmaEffectId(card, 2), False, "second"),
        ),
        (
            SequenceNode("first", ("stop", "never")),
            StopNode("stop"),
            DrawNode("never", ValueRef.literal(10), "unreachable", player=EXECUTOR),
            DrawNode("second", ValueRef.literal(1), "drawn", player=EXECUTOR),
        ),
    )
    state = _state()
    result = _run(program, state, card="pottery")
    assert result.status is EffectStatus.COMPLETE
    # Effect 1 stopped, but effect 2 still ran, and no age 10 card was drawn.
    hand = result.state.player(P1).hand
    assert len(hand) == len(state.player(P1).hand) + 1
    assert all(REGISTRY.card(card_id).age < 10 for card_id in hand)


def test_a_transfer_can_target_another_players_board_and_adopts_its_splay() -> None:
    """Decision 14: the card lands atop the matching destination stack."""

    program = _single_effect_program(
        "board-transfer-v1",
        "transfer",
        (
            MoveNode(
                "transfer",
                MovementKind.TRANSFER,
                CardSelector.stack(EXECUTOR, color=Color.RED, position=StackPosition.TOP),
                destination_player=OPPONENT,
                destination_zone=ZoneKind.BOARD,
            ),
        ),
    )
    state = _state()
    result = _run(program, state, card="the-wheel")
    assert result.status is EffectStatus.COMPLETE
    destination = result.state.player(P2).board.stack(Color.RED)
    assert destination.top == CardId("metalworking")
    # A new one-card stack is unsplayed.
    assert destination.splay is SplayDirection.NONE


def test_a_splay_node_can_take_its_colour_from_a_variable() -> None:
    program = _single_effect_program(
        "variable-splay-v1",
        "sequence",
        (
            SequenceNode("sequence", ("choose", "splay")),
            ChoiceNode(
                "choose",
                ChoiceKind.COLOR,
                "color",
                color_source=ChoiceColorSource.PRESENT_ON_BOARD,
            ),
            SplayNode(
                "splay",
                EXECUTOR,
                color_variable="color",
                direction=SplayDirection.UP,
                result_variable="splayed",
            ),
        ),
    )
    state = _state()
    programs = EffectProgramRegistry((program,))
    started = start_effect(
        state, program.program_id, _context(state, "the-wheel"), programs, REGISTRY
    )
    assert started.status is EffectStatus.AWAIT_DECISION
    assert started.decision is not None
    # Decision 15: every colour the chooser has is legal, including singleton stacks.
    offered = {
        action.color
        for action in started.decision.legal_actions
        if isinstance(action, ChooseColorAction)
    }
    assert offered == {Color.RED, Color.BLUE, Color.GREEN}
    chosen = next(
        action
        for action in started.decision.legal_actions
        if getattr(action, "color", None) is Color.GREEN
    )
    result = submit_effect_action(started.state, chosen, programs, REGISTRY)
    assert result.state.player(P1).board.stack(Color.GREEN).splay is SplayDirection.UP


def test_a_no_op_splay_of_a_singleton_stack_emits_no_change() -> None:
    """Decision 15: the choice is legal, but the no-op earns no sharing credit."""

    program = _single_effect_program(
        "noop-splay-v1",
        "splay",
        (
            SplayNode(
                "splay",
                EXECUTOR,
                Color.BLUE,
                SplayDirection.LEFT,
                result_variable="splayed",
            ),
        ),
    )
    state = _state()
    result = _run(program, state, card="the-wheel")
    assert result.status is EffectStatus.COMPLETE
    assert result.qualifying_changes == 0
    assert result.state.player(P1).board.stack(Color.BLUE).splay is (SplayDirection.NONE)


def test_a_win_node_awards_the_executor_the_game() -> None:
    program = _single_effect_program(
        "executor-win-v1", "win", (WinNode("win", mode=WinMode.EXECUTOR, player=EXECUTOR),)
    )
    result = _run(program, _state(), card="the-wheel")
    assert result.status is EffectStatus.TERMINAL
    terminal = result.state.terminal_result
    assert terminal is not None and terminal.winners == (P1,)


def test_a_unique_extreme_win_is_ignored_entirely_on_a_tie() -> None:
    """Rules section 11: a tie ignores the whole win effect and play continues."""

    tie_program = _single_effect_program(
        "tie-win-v1",
        "win",
        (
            WinNode(
                "win",
                mode=WinMode.UNIQUE_EXTREME,
                metric=WinMetric.ACHIEVEMENTS,
                extreme=Extreme.HIGHEST,
            ),
        ),
    )
    result = _run(tie_program, _state(), card="the-wheel")
    assert result.status is EffectStatus.COMPLETE
    assert result.state.terminal_result is None


def test_a_unique_extreme_win_fires_for_a_strict_leader() -> None:
    program = _single_effect_program(
        "score-win-v1",
        "win",
        (
            WinNode(
                "win",
                mode=WinMode.UNIQUE_EXTREME,
                metric=WinMetric.SCORE,
                extreme=Extreme.HIGHEST,
            ),
        ),
    )
    result = _run(program, _state(), card="the-wheel")
    terminal = result.state.terminal_result
    assert terminal is not None and terminal.winners == (P1,)


def test_a_unique_lowest_win_and_a_visible_icon_metric_both_resolve() -> None:
    lowest = _single_effect_program(
        "lowest-win-v1",
        "win",
        (
            WinNode(
                "win",
                mode=WinMode.UNIQUE_EXTREME,
                metric=WinMetric.SCORE,
                extreme=Extreme.LOWEST,
            ),
        ),
    )
    result = _run(lowest, _state(), card="the-wheel")
    terminal = result.state.terminal_result
    assert terminal is not None and terminal.winners == (P2,)

    icons = _single_effect_program(
        "icon-win-v1",
        "win",
        (
            WinNode(
                "win",
                mode=WinMode.UNIQUE_EXTREME,
                metric=WinMetric.VISIBLE_ICON,
                icon=Icon.CASTLE,
                extreme=Extreme.HIGHEST,
            ),
        ),
    )
    icon_result = _run(icons, _state(), card="the-wheel")
    icon_terminal = icon_result.state.terminal_result
    assert icon_terminal is not None and icon_terminal.winners == (P1,)


def test_batch_achievement_claims_share_the_atom_group_and_wait_for_all_children() -> None:
    card = CardId("the-wheel")
    program = EffectProgram(
        "atomic-achievement-v1",
        card,
        (ProgramEffect(DogmaEffectId(card, 1), False, "batch"),),
        (
            BatchNode("batch", ("score-tools", "score-writing")),
            MoveNode(
                "score-tools",
                MovementKind.SCORE,
                CardSelector.constant((CardId("tools"),)),
                destination_player=EXECUTOR,
            ),
            MoveNode(
                "score-writing",
                MovementKind.SCORE,
                CardSelector.constant((CardId("writing"),)),
                destination_player=EXECUTOR,
            ),
        ),
    )
    counters = TurnCounters.empty().increment(P1, scored=5)
    state = build_explicit_state(
        REGISTRY,
        positions=(
            (
                P1,
                ExplicitPlayerPosition(
                    hand=(CardId("tools"), CardId("writing")),
                    board=((Color.GREEN, (card,)),),
                ),
            ),
            (P2, ExplicitPlayerPosition(board=((Color.BLUE, (CardId("pottery"),)),))),
        ),
        turn_counters=counters,
    )

    result = _run(program, state, card="the-wheel")
    assert result.status is EffectStatus.COMPLETE
    assert set(result.state.player(P1).score_pile) == {CardId("tools"), CardId("writing")}
    assert SpecialAchievementId.MONUMENT in result.state.player(P1).special_achievements
    changed = tuple(event for event in result.events if event.changed)
    assert tuple(event.kind for event in changed) == (
        EffectEventKind.CHANGE,
        EffectEventKind.CHANGE,
        EffectEventKind.ACHIEVEMENT,
    )
    achievement = changed[-1]
    assert achievement.achievement_player is P1
    assert achievement.achievement_id is SpecialAchievementId.MONUMENT
    assert len({event.atomic_group_id for event in changed}) == 1


def test_a_sixth_achievement_at_a_batch_boundary_stops_following_nodes_and_unwinds() -> None:
    card = CardId("the-wheel")
    program = EffectProgram(
        "atomic-terminal-v1",
        card,
        (ProgramEffect(DogmaEffectId(card, 1), False, "sequence"),),
        (
            SequenceNode("sequence", ("batch", "never-draw")),
            BatchNode("batch", ("score-tools", "score-writing")),
            MoveNode(
                "score-tools",
                MovementKind.SCORE,
                CardSelector.constant((CardId("tools"),)),
                destination_player=EXECUTOR,
            ),
            MoveNode(
                "score-writing",
                MovementKind.SCORE,
                CardSelector.constant((CardId("writing"),)),
                destination_player=EXECUTOR,
            ),
            DrawNode("never-draw", ValueRef.literal(1), "drawn", player=EXECUTOR),
        ),
    )
    counters = TurnCounters.empty().increment(P1, scored=5)
    state = build_explicit_state(
        REGISTRY,
        positions=(
            (
                P1,
                ExplicitPlayerPosition(
                    hand=(CardId("tools"), CardId("writing")),
                    board=((Color.GREEN, (card,)),),
                    normal_achievements=tuple(NormalAchievementId)[:5],
                ),
            ),
            (P2, ExplicitPlayerPosition(board=((Color.BLUE, (CardId("pottery"),)),))),
        ),
        supply_tops=((1, (CardId("agriculture"),)),),
        turn_counters=counters,
    )

    result = _run(program, state, card="the-wheel")
    assert result.status is EffectStatus.TERMINAL
    assert result.state.player(P1).hand == ()
    assert CardId("agriculture") not in result.state.player(P1).hand
    assert result.state.pending_effects == ()
    assert result.state.effect_variables == ()
    assert result.state.revealed == ()
    assert result.state.terminal_result is not None
    assert result.state.terminal_result.winners == (P1,)


def test_a_claim_achievement_node_uses_its_cards_linked_route() -> None:
    """Translation's World route: every top card has a crown."""

    card = CardId("translation")
    program = EffectProgram(
        "translation-route-v1",
        card,
        (
            ProgramEffect(DogmaEffectId(card, 1), False, "noop"),
            ProgramEffect(DogmaEffectId(card, 2), False, "claim"),
        ),
        (
            NoOpNode("noop"),
            ClaimAchievementNode("claim", SpecialAchievementId.WORLD, result_variable="claimed"),
        ),
    )
    state = build_explicit_state(
        REGISTRY,
        positions=(
            (
                P1,
                ExplicitPlayerPosition(
                    board=(
                        (Color.PURPLE, (CardId("code-of-laws"),)),
                        (Color.GREEN, (CardId("sailing"),)),
                    )
                ),
            ),
            (P2, ExplicitPlayerPosition(board=((Color.BLUE, (CardId("pottery"),)),))),
        ),
    )
    result = _run(program, state, card="translation")
    assert result.status is EffectStatus.COMPLETE
    assert SpecialAchievementId.WORLD in (result.state.player(P1).special_achievements)
    # Decision 2: an achievement claim is a player-facing change.
    assert result.qualifying_changes >= 1


def test_a_claim_node_does_nothing_when_its_route_condition_fails() -> None:
    card = CardId("translation")
    program = EffectProgram(
        "translation-route-fail-v1",
        card,
        (
            ProgramEffect(DogmaEffectId(card, 1), False, "noop"),
            ProgramEffect(DogmaEffectId(card, 2), False, "claim"),
        ),
        (
            NoOpNode("noop"),
            ClaimAchievementNode("claim", SpecialAchievementId.WORLD),
        ),
    )
    result = _run(program, _state(), card="translation")
    assert result.status is EffectStatus.COMPLETE
    assert not result.state.player(P1).special_achievements


def test_a_let_node_binds_a_quantity_a_card_set_and_a_colour() -> None:
    program = _single_effect_program(
        "let-v1",
        "sequence",
        (
            SequenceNode("sequence", ("count", "cards", "colour", "branch")),
            LetNode(
                "count",
                "hand-size",
                value=ValueRef(ValueRefKind.COUNT_SELECTOR, selector=CardSelector.hand(EXECUTOR)),
            ),
            LetNode("cards", "tops", cards=CardSelector.top_cards(EXECUTOR)),
            LetNode("colour", "blue-colour", color_of="absent"),
            ConditionNode(
                "branch",
                Predicate.count(ValueRef.from_variable("hand-size"), Cmp.EQ, ValueRef.literal(3)),
                "draw",
            ),
            DrawNode("draw", ValueRef.literal(1), "drawn", player=EXECUTOR),
        ),
    )
    state = _state()
    result = _run(program, state, card="the-wheel")
    assert len(result.state.player(P1).hand) == 4


def test_an_explicit_movement_order_variable_must_cover_the_moved_cards() -> None:
    program = _single_effect_program(
        "bad-order-v1",
        "sequence",
        (
            SequenceNode("sequence", ("bind", "move")),
            LetNode("bind", "order", cards=CardSelector.top_cards(EXECUTOR)),
            MoveNode(
                "move",
                MovementKind.RETURN,
                CardSelector.hand(EXECUTOR),
                order_variable="order",
            ),
        ),
    )
    with pytest.raises(EffectInvariantError, match="does not cover"):
        _run(program, _state(), card="the-wheel")


def test_a_movement_records_which_cards_actually_moved() -> None:
    program = _single_effect_program(
        "moved-variable-v1",
        "sequence",
        (
            SequenceNode("sequence", ("move", "branch")),
            MoveNode(
                "move",
                MovementKind.SCORE,
                CardSelector.hand(EXECUTOR, value=1),
                destination_player=EXECUTOR,
                moved_variable="moved",
            ),
            ConditionNode("branch", Predicate.truthy("moved"), "draw"),
            DrawNode("draw", ValueRef.literal(1), "drawn", player=EXECUTOR),
        ),
    )
    state = _state()
    result = _run(program, state, card="the-wheel")
    assert CardId("tools") in result.state.player(P1).score_pile
    # The "if you do" branch fired because the move really happened.
    assert len(result.state.player(P1).hand) == 3


def test_an_order_group_of_all_asks_whenever_two_or_more_cards_move() -> None:
    program = _single_effect_program(
        "order-all-v1",
        "sequence",
        (
            SequenceNode("sequence", ("order", "move")),
            ChoiceNode(
                "order",
                ChoiceKind.ORDER_CARDS,
                "order",
                cards=CardSelector.hand(EXECUTOR),
                order_group=OrderGroup.ALL,
            ),
            MoveNode(
                "move",
                MovementKind.SCORE,
                CardSelector.hand(EXECUTOR),
                destination_player=EXECUTOR,
                order_variable="order",
            ),
        ),
    )
    state = _state()
    programs = EffectProgramRegistry((program,))
    started = start_effect(
        state, program.program_id, _context(state, "the-wheel"), programs, REGISTRY
    )
    assert started.status is EffectStatus.AWAIT_DECISION
    assert started.decision is not None
    assert len(started.decision.legal_actions) == 3
    assert started.decision.context is not None
    assert started.decision.context.maximum_count == 3


def test_declarative_contract_validation_rejects_impossible_programs() -> None:
    with pytest.raises(ValueError, match="unreachable nodes"):
        _single_effect_program(
            "unreachable-v1",
            "noop",
            (NoOpNode("noop"), NoOpNode("orphan")),
        )
    with pytest.raises(ValueError, match="all-or-none bodies"):
        _single_effect_program(
            "non-atomic-all-or-none-v1",
            "guarded",
            (
                AllOrNoneNode(
                    "guarded",
                    Predicate.non_empty(CardSelector.hand(EXECUTOR)),
                    "sequence",
                ),
                SequenceNode("sequence", ("noop",)),
                NoOpNode("noop"),
            ),
        )
    with pytest.raises(ValueError, match="one value, card, or colour source"):
        LetNode("bad", "result")
    with pytest.raises(ValueError, match="hand, score, or board destination"):
        MoveNode(
            "bad",
            MovementKind.TRANSFER,
            CardSelector.hand(EXECUTOR),
            destination_player=EXECUTOR,
            destination_zone=ZoneKind.SUPPLY,
        )
    with pytest.raises(ValueError, match="unique-extreme win carries a metric"):
        WinNode("bad", mode=WinMode.EXECUTOR, metric=WinMetric.SCORE)
    with pytest.raises(ValueError, match="one colour source"):
        Predicate.card_color_in("card")
    with pytest.raises(ValueError, match="both required and excluded"):
        CardSelector.top_cards(EXECUTOR, colors=(Color.RED,), exclude_colors=(Color.RED,))
    with pytest.raises(ValueError, match="one literal or expression source"):
        CardSelector(CardSelectorKind.HAND, EXECUTOR, value=1, value_expr=ValueRef.literal(2))
    with pytest.raises(ValueError, match="only to board selectors"):
        CardSelector(CardSelectorKind.HAND, EXECUTOR, position=StackPosition.TOP)
    with pytest.raises(ValueError, match="one option source"):
        ChoiceNode(
            "bad",
            ChoiceKind.COLOR,
            "result",
            colors=(Color.RED,),
            color_source=ChoiceColorSource.PRESENT_ON_BOARD,
        )
    with pytest.raises(ValueError, match="only a hidden-card choice"):
        ChoiceNode(
            "bad",
            ChoiceKind.CARD,
            "result",
            cards=CardSelector.hand(EXECUTOR),
            owner=EXECUTOR,
        )
    with pytest.raises(ValueError, match="only to ordering choices"):
        ChoiceNode(
            "bad",
            ChoiceKind.CARD,
            "result",
            cards=CardSelector.hand(EXECUTOR),
            order_group=OrderGroup.AGE,
        )


def test_player_reference_kinds_all_resolve() -> None:
    state = _state()
    context = _context(state)
    state = set_effect_variable(state, context, "target", P2.value)
    cases = (
        (PlayerRef(PlayerRefKind.ACTOR), P1),
        (PlayerRef(PlayerRefKind.CHOOSER), P1),
        (EXECUTOR, P1),
        (ACTIVATOR, P1),
        (OPPONENT, P2),
        (PlayerRef.literal(P2), P2),
        (PlayerRef.from_variable("target"), P2),
    )
    for reference, expected in cases:
        assert resolve_player(reference, context, state) is expected
