# WP6 Achievement and Terminal Contract

WP6 owns `innovation/achievements.py`, `innovation/terminal.py`, and their tests. It adds no new
state fields; the WP2 `TurnCounters` and `PlayerState` achievement tuples are sufficient.

## Modules and ownership

| Module | Owns |
|---|---|
| `innovation/achievements.py` | normal legality/claiming, five automatic predicates, five linked-card routes, Monument counters, atomic boundary checks |
| `innovation/terminal.py` | sixth-achievement, draw-above-10, direct card-effect wins, unique-most/lowest tie rules, `apply_terminal` |

Integration edits to `innovation/protocol.py` are limited to delegating achievement legality,
claiming, the boundary check, and terminal construction. No shared contract changed.

## Normal achievements

`normal_achievement_is_eligible` requires both printed conditions: score at least `5 × age`, and
at least one top card of value `age` or higher. Claiming never spends score. An achievement owned
by either player is unavailable to both. `claim_normal_achievement` raises
`AchievementClaimError` for an ineligible claim; `protocol.apply_action` never offers one, so an
agent instead receives `IllegalAction`.

## Special achievements

Each special achievement has two independent routes.

| Achievement | Automatic predicate | Linked route (card effect) |
|---|---|---|
| Monument | tucked ≥ 6 **or** scored ≥ 6 in one turn, counted separately | Masonry effect 1: melded ≥ 4 castle cards in that effect |
| Empire | ≥ 3 visible icons of all six types | Construction effect 2: only player with five top cards |
| World | ≥ 12 visible clock icons | Translation effect 2: every top card has a crown |
| Wonder | five colors present, each splayed right or up | Invention effect 2: five colors splayed in any direction |
| Universe | five top cards, one per color, each value ≥ 8 | Astronomy effect 2: all non-purple top cards value ≥ 6 |

The linked routes are strictly weaker or differently shaped than the automatic predicates, so
they are separate functions and separate `ClaimRoute` values. `AchievementClaim` records the
`DogmaEffectId` for a linked claim, which keeps the achievement log provenance-friendly.

Two rules-decision interpretations are exercised by tests:

- decision 10 — Translation and Astronomy universal predicates hold vacuously for an empty
  relevant top-card set;
- decision 3 — the deterministic check order below.

## Atomic-boundary check order

`check_order(state, active_player=None)` returns the total order used at every boundary: the
active player is fully checked before the opponent, and each player's achievements follow
`SPECIAL_CHECK_ORDER` = Monument, Empire, World, Wonder, Universe. Sixth-achievement victory is
tested after every single claim, so an immediate win stops all remaining checks and all remaining
dogma work, including the sharing bonus.

Predicates read live state, never frozen dogma icon counts.

## Monument counters and provenance exclusions

`qualifying_monument_movements(change)` classifies a WP2 `ChangeRecord`. Only `ChangeKind.TUCK`
into a board and `ChangeKind.SCORE` into a score pile qualify; `TRANSFER` and `EXCHANGE` never
do, which is exactly the printed exclusion for cards transferred into or exchanged into a score
pile. Because the filter is keyed on change kind, no card-specific code can bypass it.

WP2's `tuck_card` and `score_card` already increment the counters for their own single-card
movement, so `record_qualifying_movements` (and `check_after_change(..., count_monument_movements=True)`)
is only for bulk atoms that write zones without those primitives. Counters advance only in
`GamePhase.PLAY` and reset when the turn rotates.

## Integration API for WP4/WP5

WP4 provenance types are being designed concurrently, so WP6 exposes *generic* entry points that
take authoritative state plus plain-data hints. Nothing here imports effect frames, effect
contexts, or provenance records.

```python
check_atomic_boundary(state, registry, *, active_player=None, previous_claims=()) -> AchievementCheckResult
check_after_change(state, change, registry, *, active_player=None, count_monument_movements=False) -> AchievementCheckResult
claim_linked_route(state, player_id, achievement_id, registry, *, context=LinkedRouteContext(), check_boundary=True) -> AchievementCheckResult
record_qualifying_movements(state, movements) -> GameState
apply_terminal(state, result) -> GameState
```

`AchievementCheckResult` carries the new state, the ordered claims, and an optional
`TerminalResult`. `result.game_over` is the single signal the effect executor must honour:

1. call `check_atomic_boundary` (or `check_after_change`) after every atomic operation, after
   each completed dogma effect, after each paid action, and at turn completion;
2. replace the working state with `result.state`;
3. if `result.game_over`, unwind the frame stack and return the terminal result without running
   any further effect, sharing bonus, or paid action.

When WP4's provenance record exists, the expected integration is a thin adapter that maps one
provenance batch to `tuple[QualifyingMovement, ...]` plus the executing player, then calls the
functions above. That adapter belongs to WP4; no WP6 signature needs to change.

## Terminal results

`terminal.py` never mutates state except through `apply_terminal`, which refuses to finalize an
already terminal state and produces a state that rejects further zone mutation.

- `achievement_victory_result` — sixth achievement, normal plus special.
- `draw_beyond_age_ten_result` — score, then achievement count, then draw.
- `direct_card_effect_win(player)` / `card_effect_draw()` — unconditional card text.
- `unique_most_result` / `unique_lowest_result` — return `None` on a tie, meaning the whole win
  effect is ignored and play continues (Empiricism-style unconditional wins use
  `direct_card_effect_win` instead).
- Convenience wrappers for the printed comparisons that exist in the base game:
  `unique_most_points_result`, `unique_lowest_score_result`, `unique_most_visible_icon_result`,
  plus the raw count helpers `score_counts`, `achievement_counts`, `visible_icon_counts`,
  `board_color_counts`, `card_is_top_anywhere`, and `has_strict_maximum`.

`protocol.terminal_transition(state, result)` is the shared protocol-level handoff for card
effects that end the game.

## Sharing bonus

Per rules-decision 2, claiming an achievement is not by itself a qualifying change for the
sharing bonus. `AchievementCheckResult.changed` reports claims only so WP5 can explicitly
exclude them; WP6 never grants a free Draw.
