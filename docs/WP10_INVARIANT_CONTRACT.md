# WP10 invariant and fuzz contract

WP10 provides reusable checks in `innovation.invariants` and deterministic full-protocol fuzzing
in `innovation.fuzz`. It is integrated with the completed WP1–WP9 engine and the full **105/105**
card registry; unimplemented card behavior is never treated as a no-op.

## Available checks

`assert_state_properties` composes state-local checks for:

- card conservation and unique authoritative location;
- structural zone, registry-fingerprint, and achievement ownership consistency;
- score totals, public score values, and private score identities;
- icon geometry, splay visibility, and owner board observations;
- phase, active player, paid actions, turn fields, and reveal markers;
- automatic special-achievement completion at stable play boundaries;
- complete deterministic setup, paid-action, and effect-choice legal-action sets;
- terminal immutability across every public mutation entry point.

Transition helpers are `assert_transition_purity`, `assert_turn_progression`,
`assert_transition_consistency`, and `checked_apply_action`. Effect choices have exact progression
rules: they never consume another paid action, preserve the current turn while work remains, and
may rotate only when the already-paid Dogma action completes. Hidden-equivalent state pairs use
`assert_observation_leak_resistance`; targeted tests cover supply order, achievements, hands,
scores, and covered boards.

Replay invokes state-local checks at the restored initial boundary and after every replayed action,
in addition to verifying recorded decisions and hashes. A seeded setup-to-terminal log test
contains real Dogma and nested effect choices. Runner integration compares real multi-game batch
records and final states with independent sequential runs using independently seeded agents.

## Deterministic fuzzing

`run_protocol_fuzz(seed)` chooses among every legal setup, Draw, Meld, Dogma, Achieve, and effect
choice action. It checks the initial state and every transition, records before/after hashes,
enforces a step ceiling, and must terminate. Small golden trace digests pin deliberate semantic
behavior changes.

The default suite keeps a fast deterministic fuzz sample. The release-scale gate is:

```bash
INNOVATION_LARGE_FUZZ_SEEDS=100 uv run pytest -q -m fuzz tests/innovation/test_fuzz.py
```

The latest integration run on August 25, 2026 completed all 100 seeds: `4 passed, 1 deselected`
in 143.68 seconds for the command above.

The fuzzer checks player-safe observations on every generated state. Strong noninterference claims
still belong to the focused hidden-equivalent-pair tests; random play does not synthesize a second
paired authoritative state at each transition.

## Maintenance requirements

- New effect choices or protocol phases must extend legal-action completeness and exact turn
  progression in the same change.
- New public mutation entry points must be added to terminal-immutability probing.
- Card programs must use shared declarative mutation APIs and remain covered by the all-card
  minimum-state smoke gate.
- Intentional semantic changes may update golden traces only after focused rules regressions prove
  the new behavior.
- Replay schema/fingerprint/version changes must remain explicit and fail incompatibly rather than
  diverging silently.
- Agent randomness stays separate from setup and protocol-fuzzer RNG seeds.

Do not weaken an invariant to accept an unknown state. Either extend the exhaustive contract for a
legitimate new case or fix the production transition that violated it.
