# WP4 effect contract

WP4 implements card behavior as explicit declarative programs interpreted by
`innovation_ai.innovation.effects`. It does not register the 105 production card programs; the
named Pottery, Metalworking, Machinery, Publications, Fission, and Self Service programs are
synthetic specification fixtures for the shared VM only.

## Resumption boundary

- `start_effect(...)` installs an `effect-program` root frame.
- `start_program_effect(...)` starts one printed effect ordinal so WP5 can perform its required
  per-effect opponent/activator ordering without reaching into private frame helpers.
- `resume_effect(...)` runs deterministic frames until it returns `await-decision`, `complete`,
  `abort-dogma`, or `terminal`.
- `step_effect(...)` executes exactly one frame operation, allowing a checkpoint at every control
  or atomic-mutation boundary.
- `submit_effect_action(...)` validates one WP3 semantic action against the current player-safe
  `Decision`, updates scoped serializable variables, and resumes.
- `effect_runtime_payload(...)` and `restore_effect_runtime(...)` are the versioned WP4-only
  frame/variable round trip. Full `GameState` loading remains WP9 ownership.

The tuple in `GameState.pending_effects` is bottom-to-top. A WP4 root may be pushed above an
orchestrator-owned frame; completing that root returns an effect boundary while preserving lower
frames and variables outside its context scope. Frames contain only primitive values, semantic
IDs, progress counters, and `EffectContext`; no generators, closures, bound methods, or agent
callbacks are retained. Programs are looked up by stable program ID in an explicit
`EffectProgramRegistry`. The serialized step count and per-context limit provide deterministic
cycle protection, including across nested execution and save/restore.

## Scope and causality

`EffectContext` records actor, chooser, executor, dogma activator, source card/effect, turn and
dogma-action IDs, and demand/shared/nested flags. Variables use `scope:name` keys. Each printed
effect and nested program receives a child scope which is cleared on return, preventing stale
choice values from bleeding into repeats or later effects. Nested execution filters demand
entries, does not start a sharing pass, and retains the outer `shared` flag so WP5 can justify one
outer sharing bonus.

Every physical mutation or geometry change returns an `EffectEvent` with a monotonic event ID,
full context, moved card IDs, and the WP2 `ChangeRecord` containing source/destination addresses.
Reveal, keep, and abort are also explicit events. Batch children share an atomic group ID and do
not expose a decision between children. No-op movement or re-splay emits no change event. A
serialized root-scope qualifying-change count survives pauses; `EffectResolution` returns the
final count even after runtime variables are cleared. Mutation nodes may also write a scoped
boolean result, allowing explicit `if you do` conditions without inspecting transient events.

## Primitive surface

The declarative node set freezes shared sequence, card movement, draw/reveal/keep, exchange,
splay, rearrangement, choice, condition, repeat, atomic batch, nested non-demand execution,
mass-removal, no-op, and dogma-abort behavior. Choice nodes map to WP3 semantic action types and
separate chooser from executor. Bounded card selection is incremental and uses
`FinishSelectionAction`; arbitrary rearrangement uses `OrderCardsAction` as required by Freeze A.

## Integration boundary

WP3's current `dogma-action` handoff frame remains intentionally generic: WP5 owns icon freezing
and demand/share executor ordering. A WP4 root can be pushed above that frame with
`start_program_effect`; completion preserves the lower orchestration frame, while Fission abort
and terminal results deliberately unwind everything. The synthetic registry is not imported by
the paid-turn protocol. WP9 should embed the WP4 runtime payload in full state/log decoding rather
than defining a second frame schema; the runtime loader round-trips unowned frame kinds while
strictly validating WP4 program/node references and serialized counters.
