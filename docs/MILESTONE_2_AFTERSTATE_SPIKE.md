# Milestone 2 afterstate-boundary feasibility spike

**Decision date:** August 29, 2026  
**Selected semantics:** `immediate-one-transition-v1`

## Question

For a paid `DogmaAction`, should the first value policy stop after the single public
`apply_action()` transition, even when that transition exposes an `EFFECT_CHOICE`, or should the
heuristic fallback roll the whole Dogma activation to the next paid-turn boundary before encoding?

## Result

The immediate boundary is representable without using the next chooser's private observation. The
trusted expander applies exactly one semantic paid action to an information-set sample, then builds
a fresh `ValuePosition` by observing the sampled afterstate as the **original action chooser**.
`PublicBoundary` separately encodes the next decision's relative chooser/executor/activator,
source card/effect, demand/shared/nested flags, frozen icon counts, sanitized selection state, and
public Monument counters.

Rolling the fallback through a complete Dogma activation was rejected for encoder v1 because it:

1. changes “one ply” from one semantic action into a variable number of nested decisions;
2. couples candidate values to heuristic effect-choice behavior that is not part of the network;
3. requires sampling and resuming arbitrary paused effect-VM states, outside the initial sampler's
   stable paid-turn contract; and
4. makes replay extraction less direct because one training example would span several recorded
   semantic actions.

The selected boundary leaves effect choices to the existing heuristic on the next scheduler pull.
Compact replay still records those choices, and dataset extraction may encode their nonterminal
post-action positions.

## Acceptance corpus

The following committed fixtures cover the risks that drove the spike:

| Risk | Evidence |
|---|---|
| Draw must not reveal the real next supply card before commitment | `tests/training/test_determinizations.py::test_hidden_equivalent_specs_samples_and_candidate_features_are_identical` |
| Next decision belongs to the opponent | `tests/training/test_determinizations.py::test_draw_afterstate_is_reobserved_for_original_chooser_after_turn_rotation` |
| Immediate terminal candidate | `tests/harness/test_afterstates.py::test_terminal_candidate_bypasses_positions_and_uses_exact_original_viewer_utility` |
| Choice/effect context can be represented without hidden selected identities | `tests/harness/test_policy.py::test_context_sanitization_replaces_hidden_selected_identity_with_count` |
| Second-action Monument progress changes the value input | `tests/training/test_encoding.py::test_public_monument_progress_and_current_afterstate_marker_encode_differently` |
| Private unsplayed covered count remains unknown | `tests/training/test_encoding.py::test_unknown_covered_count_differs_from_known_zero` |
| Splayed hidden cards preserve visible-icon constraints | `tests/training/test_determinizations.py::test_sample_preserves_observation_legal_actions_splays_and_synthetic_provenance` |
| Unstable pending effects/reveals cannot enter the sampler | `tests/training/test_determinizations.py::test_builder_rejects_unstable_effect_boundary` |
| Exact semantic candidate grouping and terminal/model mixing | `tests/training/test_selection.py` |
| Batched learned routing, simultaneous setup fallback, and failure isolation | `tests/harness/test_policy_scheduler.py` |
| Fission reveal/removal/terminal branches remain engine-authoritative | `tests/innovation/cards/age09/test_fission.py` |

The information-set sampler supports only stable `PLAY` / `TURN_ACTION` states with no pending
effect stack, transient effect variables, or physical reveal. It has no API accepting a live state;
only the audited builder does. A sampler failure is strict or heuristic-fallback according to the
resolved policy and never falls back to evaluating true-state candidates.
