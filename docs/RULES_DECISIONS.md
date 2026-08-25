# Rules Decisions

This record prevents card implementations from silently choosing between plausible readings.
The supplied two-player rules and card text remain authoritative. Each decision must gain focused
engine tests in the work package that first makes it executable.

## Status

- **Accepted decisions:** 1–16 and 18.
- **Awaiting owner confirmation:** 17 (quantity snapshot timing).
- Decisions 2, 3, 8, 10, and 13 were confirmed by the project owner on August 25, 2026.
- Decisions 15, 16, and 18 record the corresponding effect/protocol contracts confirmed in the same review.

## Decisions

### 1. Draws requested at value zero — accepted

A requested value of zero starts at age 1, then uses the normal upward-only empty-pile rule. A
request above age 10 ends the game immediately. This gives operational meaning to the rule that
an absent value is zero without inventing an age-0 pile.

**Tests:** empty-board Draw; Machine Tools with an empty score pile; upward fallback; value 11.

### 2. Changes qualifying for the sharing bonus — accepted

Use the rulebook's broad “if anything happened” criterion: any player-facing gameplay-state
change caused by a shared opponent execution qualifies. This includes card draws, reveals,
movements, exchanges, removals, effective splay changes, and achievement claims. Internal frame
progress, monotonic IDs, bookkeeping variables, and choices that leave gameplay state unchanged
do not qualify. A no-op re-splay does not qualify. Nested changes retain attribution to the outer
shared execution.

**Tests:** declined optional effect; no-op splay; achievement-only shared effect; nested movement.

### 3. Different special achievements becoming eligible simultaneously — accepted

At an atomic boundary, check the active player and then the opponent. For each player, check
Monument, Empire, World, Wonder, then Universe. Check sixth-achievement victory after each claim
and stop immediately. The supplied rule explicitly gives the active player priority only when
both players become eligible for the same special achievement; this broader ordering is an
engine determinism rule.

**Tests:** both players become eligible for different sixth achievements in one atomic operation.

### 4. Atomic bulk operations — accepted

One card-effect instruction is the normal atomic boundary. Exchanges, explicitly grouped
transfers/returns/melds, draw-and-X operations, and Fission's mass removal do not expose
intermediate decisions or achievement checks. Per-card provenance is still recorded inside the
atom. A draw above age 10 terminates immediately inside an atom.

**Tests:** six-card batch score; empty-side exchange; no observable intermediate exchange state.

### 5. Ordering multiple cards — accepted

The returning player orders cards returned to the same age pile. The executing player orders
multiple melds or tucks into one stack. Ask for an order only when it can affect authoritative
state; otherwise use stable card-ID order.

**Tests:** same-age return order; mixed-age returns; multi-meld resulting top card.

### 6. Nested non-demand execution — accepted

Computers, Satellites, Robotics, Self Service, and Software execute only the selected card's
non-demand effects for the current executor. They do not start a share pass, freeze new icon
counts, apply demands, or award a separate sharing bonus. Their changes preserve causal
attribution to the outer execution. A deterministic step ceiling must fail loudly rather than
silently truncating a legal chain.

**Tests:** demand skipped; no nested sharing; outer sharing credit; serialize during a nested choice.

### 7. Fission abort — accepted

When Fission's red 10 condition occurs, remove the specified cards, preserve achievements, and
abort all remaining dogma work. Do not evaluate a sharing bonus. If no win condition fired, the
turn continues with any paid action still remaining.

**Tests:** red and non-red branches; remaining paid action; serialize immediately before removal.

### 8. Societies target scope — accepted

Interpret “a card ... from your board” as a **top card**. The authoritative CSV wording is amended
to say “a top card” explicitly. This is consistent with public board selection under the default
information policy and the project owner's intended behavior.

**Tests:** qualifying top card; covered qualifying card excluded; absent matching color has value 0.

### 9. A.I. top-card locations — accepted

Robotics and Software may be top cards on different players' boards. If both are top cards
somewhere, the unique player with the lower score wins; a score tie means the win effect does
nothing.

**Tests:** cards split across boards; same board; tied score.

### 10. Empty universal predicates — accepted

Treat “all” predicates as true for an empty relevant set. Astronomy can therefore satisfy its
non-purple-top-card condition when no non-purple top cards exist; Translation's corresponding
empty-top-card case is handled the same way.

**Tests:** purple-only Astronomy board; one failing non-purple top card; synthetic empty board.

### 11. Democracy with zero returns — accepted

Compare each executor's return count with the greatest prior count, defaulting to zero. The
comparison is strict, so returning zero cards never qualifies. Counters are scoped to one dogma
action and resolution remains sequential.

**Tests:** 0/0, 2/2, 2/3, and reset on the next dogma action.

### 12. Automatic-achievement timing — accepted

Evaluate automatic predicates at each atomic operation boundary, using live state rather than
frozen dogma icon counts. Also check at effect, action, and turn completion as defensive invariant
boundaries. Monument counts individual qualifying tucks/scores with provenance exclusions for
transfers and exchanges.

**Tests:** eligibility during demand compliance; live Empire icons; Monument provenance exclusions.

### 13. Hidden equivalent choices — accepted

The player who can legally inspect and owns or controls the affected hidden zone chooses the exact
card identity among cards satisfying the public criterion. Thus a demand victim chooses exact
cards from their own hand or score pile, even though the demander may know only their values. If
an effect explicitly assigns the choice to a player who cannot inspect the identities, first
expose the public semantic choice (normally a value), then let the zone owner disambiguate among
matching identities without leaking the alternatives. If no player can choose, select the lowest
stable card ID. Decisions distinguish the effect executor from the chooser.

**Tests:** Medicine/Sanitation hidden ties; equal observations after swapping hidden identities;
stable fallback across hash seeds.

### 14. Splay after unusual movement — accepted

Splay direction belongs to a color stack. Preserve it through additions, removals, transfers, and
rearrangement, except that a stack at zero or one cards becomes unsplayed and does not remember
its prior direction. A card transferred to another board goes atop the matching destination stack
and adopts that stack's splay; a new one-card stack is unsplayed.

**Tests:** bottom-card removal; reduction to one; Publications rearrangement; board transfer.

### 15. Legal no-op splay choices — accepted

An instruction to splay a color may choose any color the chooser currently has on their board,
even when its stack has one card or is already splayed in the requested direction. A nonexistent
color is not legal unless card text explicitly permits it. The resulting no-op is still a legal
choice but does not itself qualify as “anything happened” for a sharing bonus.

**Tests:** singleton stack; already-matching direction; absent color excluded; Measurement case.

### 16. Subset selection versus movement order — accepted

Selecting an unordered subset and ordering the resulting movements are separate decisions.
Unordered subset construction may use canonical card-ID ordering so the same subset has one
semantic selection path; this does **not** determine movement order. When selected cards are
returned, melded, or tucked and order can affect authoritative state, the owner/executor of those
cards chooses the movement order unless card text explicitly names another chooser. Otherwise use
stable card-ID order without presenting a meaningless order decision.

**Tests:** canonical subset selection; same-age return order; same-color meld/tuck order; demand
victim remains the order chooser.

### 17. Quantity snapshot timing — proposed; awaiting owner confirmation

A quantity in one printed instruction is evaluated once when that instruction begins. Operations
performed by the instruction do not increase or decrease its remaining count. An explicit repeat
of a dogma effect begins a new execution and therefore reevaluates quantities and predicates from
the then-current state.

**Tests:** Industrialization factory count does not grow during its own tucks; The Internet clock
count is fixed for one execution; explicit repeat reevaluates.

### 18. Revealed information is physical, not scope-sticky — accepted

Track cards or card attributes as publicly revealed only while they are literally face-up under
the currently resolving instruction. At every decision, the observation reflects the current
physical public information. Moving, returning, scoring, or keeping a revealed card clears the
transient reveal marker unless its destination is independently public. Do not keep information
public merely because the enclosing effect scope has not completed.

Longer-lived deductions from previously observed identities are outside Milestone 1 and are
recorded in `docs/ROADMAP.md`.

**Tests:** decision while cards remain face-up; reveal marker cleared on keep/return/score;
subsequent decision does not leak a now-hidden identity.

## Contract implications discovered during review

- Observations expose public hand/score value multisets, not only counts.
- `Decision` must separately represent the effect executor and the player making the choice.
- Linked special-achievement routes identify exact card effects and remain distinct from automatic
  predicates.
- Nested execution needs a deterministic, serialized step counter or equivalent cycle protection.
