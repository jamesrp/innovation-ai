# WP10 invariant and fuzz foundation

WP10 adds reusable checks in `innovation.invariants` and deterministic protocol fuzzing in
`innovation.fuzz`. The foundation intentionally uses only the current WP1-WP3 public contracts
and does not treat unimplemented card effects as no-ops.

## Available checks

`assert_state_properties` composes the current state-local checks:

- card conservation and unique authoritative location;
- the existing structural zone/fingerprint/achievement checks;
- score totals, public score values, and private score identities;
- frozen icon-slot/splay geometry and owner board observations;
- phase, active-player, paid-action, and turn consistency;
- complete deterministic setup and paid-action legal-action sets.

Transition helpers are `assert_transition_purity`, `assert_turn_progression`,
`assert_transition_consistency`, and `checked_apply_action`. Hidden-equivalent state pairs can be
checked with `assert_observation_leak_resistance`. `assert_terminal_immutability` probes every
current public zone mutation entry point plus `apply_action` and requires typed rejection without
a state/hash change.

`run_protocol_fuzz(seed)` deterministically chooses legal setup and Draw/Meld/Achieve actions,
checks every transition, records before/after hashes, enforces a step ceiling, and must terminate.
Dogma actions remain in legal-action completeness checks but are not selected because WP3 only
creates a placeholder frame. Small golden trace digests pin deterministic behavior.

The default fuzz coverage is fast. Run a larger deterministic batch explicitly with:

```bash
INNOVATION_LARGE_FUZZ_SEEDS=100 uv run pytest -m fuzz tests/innovation/test_fuzz.py
```

## Integration requirements

- **WP4:** extend legal-action completeness to effect decisions and teach the fuzzer how to resume
  pending frames. Effect-choice transitions must get progression rules distinct from paid actions;
  do not route them through the current paid-action decrement assertion unchanged.
- **WP5:** enable Dogma selection in fuzzing only after orchestration can run to the next decision
  or terminal boundary. Preserve input purity and validate frozen icon eligibility separately from
  live board icon geometry.
- **WP6:** add automatic/special achievement consistency checks and run terminal immutability after
  every terminal route. If intentional achievement timing changes non-Dogma golden traces, review
  and update their digests rather than deleting them.
- **WP7:** card effects must use shared mutation APIs or add their new public mutation entry points
  to terminal-immutability probing. Full Dogma fuzzing must fail loudly for missing registrations.
- **WP8:** runners may use `checked_apply_action` in debug/test mode. Agent randomness must remain
  separate from setup and protocol-fuzzer RNG seeds.
- **WP9:** replay should call state-local checks at restored boundaries and compare recorded hashes.
  Serialization must preserve fuzz steps' semantic actions and terminal result; add round-trip and
  replay checks without coupling the invariant module to log schemas.

When downstream contracts add legitimate phases, decisions, terminal reasons, or mutation APIs,
extend these exhaustive checks in the same change. Do not weaken them to accept an unknown case.
