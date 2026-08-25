# WP5 Freeze-B Contract and Completed Card Integration

**Status:** Freeze B completed on August 25, 2026, and the downstream WP7 integration is now
complete. The shared dogma, effect, decision, achievement, serialization, and registry contracts
support **105/105 cards** and **158/158 printed effects**. The original six-card vertical slice
below is retained as historical validation context, not as current coverage.

## Public transition boundary

`current_decisions()` and `apply_action()` cover setup, paid actions, and every player choice made
inside dogma resolution. Selecting Dogma spends exactly one paid action, increments the dogma ID,
and resumes deterministic work until the next player decision, terminal result, abort, or normal
completion. There is no public `effect-resolution-pending` outcome and no
`finish_effect_resolution()` handoff.

Normal protocol calls always return a decision or terminal result. Low-level `step_effect()` and
`pause_before_first_step=True` intentionally expose deterministic diagnostic checkpoints; after
loading one of those checkpoints, call `resume_pending_effects()` before asking the public
protocol for a decision.

All catalog top cards are offered as Dogma actions because all 105 programs are registered.
`implemented_card_ids()` remains the authoritative completeness gate, and directly requesting an
absent program still raises `UnimplementedCardError`; missing behavior can never become a silent
no-op.

## Dogma orchestration

A serialized `dogma-action` frame freezes the activated card's featured icon and both players'
visible counts once. The immutable schedule then applies these rules:

1. an opponent with fewer icons executes demands and does not share;
2. an opponent with equal or more icons ignores demands and executes each non-demand effect first;
3. the activator executes that same non-demand effect before the next printed ordinal begins;
4. shared, nested, and demand provenance remains attached to every emitted event;
5. at most one free Draw is awarded when the opponent's shared execution caused a qualifying
   player-facing change;
6. demand changes never qualify for that bonus;
7. `AbortDogmaNode` clears all runtime, skips later effects and the bonus, and leaves any second
   paid action available.

Qualifying shared changes include effective reveals and achievement claims. Declines, bookkeeping,
and no-op splay/reveal operations do not qualify.

## Reveals and hidden choices

`GameState.revealed` is authoritative, hashed, observed, and serialized. Reveal ownership is
multi-scope, so clearing a nested re-reveal does not erase an outer instruction's still-live
reveal. Moving, returning, scoring, keeping, exchanging, or removing a card clears all of its
markers. Terminal and abort finalization clear every marker.

An exact `CARD` choice is rejected if its chooser cannot see every offered identity. Private hand
or score choices use `HIDDEN_CARD`: the assigned chooser first chooses the public value, then the
zone owner disambiguates tied identities. If the original chooser can already inspect every card
(for example after a reveal), the choice collapses to one exact-card decision. Decisions keep the
chooser distinct from the effect executor.

## Atomic achievements and terminal unwind

The VM checks automatic achievements after every non-batch atomic leaf, once after a complete
batch, at printed-effect completion, and again at action/turn protocol boundaries. Batch mutation
and the resulting achievement events share one atomic-group ID. Claim events record the claimant
and achievement ID, and active-player priority comes from authoritative `state.active_player`.

`apply_terminal()` is the single finalizer for sixth achievements, draws beyond age 10, and card
wins. It atomically sets the result and clears pending frames, effect variables, and reveal
markers. `GameState` and deserialization reject terminal states retaining transient runtime.

`AllOrNoneNode` requires its feasibility guard to describe the complete instruction and restricts
its body to one atomic leaf or `BatchNode`; card programs may not put a multi-step sequence behind
an all-or-none guard.

## Ordering, quantities, and nesting

- Unordered bounded subsets are selected incrementally in increasing card-ID order. The effective
  minimum respects mandatory partial execution when too few cards exist, while legal choices
  cannot strand a minimum that was otherwise reachable.
- Movement ordering is a separate incremental decision. It is asked only when at least two cards
  enter the same age pile or color stack; all other cross-group ordering is canonical.
- Quantities are evaluated once when their owning instruction begins. Literal and computed values
  share the same division, rounding, and offset pipeline.
- Nested execution runs only non-demand effects for the current executor, starts no new sharing
  pass, inherits outer shared attribution, and uses a serialized depth limit of 16.
- Root effect and dogma entry points reject pre-existing runtime; nesting goes only through
  `NestedNode`.

## Registry and replay compatibility

Card modules are discovered lazily from `cards/ageNN`, validated against printed effect ordinals
and demand flags, and prohibited by tests from importing mutation paths. The effect registry
publishes `implemented_card_ids()`, raises typed errors for absent programs, rejects unreachable
nodes, and hashes canonical programs plus inspectable named-helper implementations.

Game-log schema version 2 records `effects_fingerprint` beside the card-data fingerprint. Replay
rejects either fingerprint when incompatible.

## Historical six-card validation slice

- `the-wheel`: times/draw, opponent-first sharing, one free Draw;
- `code-of-laws`: relational optional tuck, `if you do`, dynamic-color optional splay;
- `archery`: demand immunity, victim-owned hidden tie choice, transfer pronouns;
- `pottery`: canonical bounded subset, effective return ordering, quantity snapshot, two ordinals;
- `metalworking`: draw/reveal/branch/repeat, physical reveal cleanup;
- `fission`: demand branch, atomic mass removal, source self-removal, complete dogma abort.

Focused suites exercise serialize/restore checkpoints, public observations, demand/share matrices,
atomic claim and terminal interruption, ordering, hidden selection, nested attribution/depth, and
registry discovery/fingerprints.

## Current WP7 breadth status

All ten age packages are integrated. The production registry contains all **105/105** catalog
cards, all **158/158** printed effects have declarative implementations, and there are no
production `NoOpNode` placeholders. Focused card suites, the all-card minimum-state smoke test,
full-Dogma log/replay, real batch-versus-sequential runner coverage, and deterministic protocol
fuzzing exercise the completed registry.

The release-scale deterministic fuzz gate is:

```bash
INNOVATION_LARGE_FUZZ_SEEDS=100 uv run pytest -q -m fuzz tests/innovation/test_fuzz.py
```
