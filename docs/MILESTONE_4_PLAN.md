# Milestone 4 Plan — Player-Safe Search Heuristic and Retraining

**Planning status (September 2, 2026):** approved for implementation; no implementation work has
started. Milestone 3 remains an immutable historical pilot. Its repetition-aware selector is
retained as an experimental variant, while the primary learned-policy line returns to the original
`temperature-softmax-v1` selector architecture.

## 1. Milestone outcome

Milestone 4 should replace the deliberately weak printed-card heuristic with a substantially
stronger, information-safe search policy, improve diagnostics for pathological games, and then
regenerate and evaluate the learned pipeline under the changed continuation policy.

Completion requires:

- a deterministic sampled minimax heuristic that searches complete player turns rather than raw
  semantic-action plies;
- use of that heuristic as the heuristic baseline and as the learned policy's setup/effect-choice
  fallback;
- adoption of `public-covered-v1`, exposing every ordered board-card identity even when its stack is
  unsplayed;
- a versioned hand-engineered minimax leaf evaluator;
- an exact diagnostic reproduction of the original seed-50000 action-ceiling failure before its
  policy configuration is superseded;
- failure-focused compressed traces, strategic-cycle metrics, and learned/search selection audits;
- a fresh training run whose replay, checkpoint, and policy identities record the new heuristic and
  information policy; and
- fixed-seed evaluation of the new learned candidate and the new heuristic, with completion and
  throughput treated as gates before playing-strength claims.

This is a search-policy milestone, not a claim that bounded determinization search solves
Innovation's imperfect-information game exactly.

## 2. Project decisions

### 2.1 Primary learned selector

The primary learned-policy line uses `temperature-softmax-v1`, as it did before the Milestone 3
repetition experiment. `recent-paid-action-penalty-v1` remains implemented and its artifacts remain
valid, but it is an ablation/comparator rather than the default candidate or implicit incumbent.

This is a selector-policy rollback, not deletion of the repetition-aware implementation and not a
rollback of the Milestone 2 value-network architecture. The stronger fallback, public-board
information policy, and fresh checkpoint will produce a new complete policy identity.

### 2.2 Public board information

Milestone 4 adopts the already-versioned `public-covered-v1` information policy as the default for
new games, training runs, and arenas:

- every card identity, age, and order on both players' boards is public, including cards covered in
  unsplayed stacks;
- splay geometry still determines which printed icon positions are functionally visible and
  counted;
- each player continues to see the exact identities in their own hand and score pile;
- opponent hand and score-pile values remain public while identities remain private except during an
  explicit reveal;
- supply order, normal-achievement identities, and secret simultaneous setup choices remain hidden;
- old logs, checkpoints, policies, and observations retain their recorded information-policy
  version and must not be silently reinterpreted.

This is a deliberate digital-play convention chosen by the project owner rather than a claim about
physical Third Edition information rules. It is a project-level observation-policy change, so it
requires focused observation, replay-compatibility, sampler, encoder, and non-leak tests when
implemented.

The encoder-v1 layout can remain structurally unchanged if its existing covered-card identity and
known-count fields represent the new observations without ambiguity. Existing checkpoints trained
under `rulebook-private-covered-v1` are not the Milestone 4 training candidate; compatibility must
continue to reject mismatched policy/checkpoint contracts.

### 2.3 Search model

The new heuristic uses deterministic **root-sampled minimax**:

1. Build a player-safe information-set specification from the root chooser's `Decision`,
   observation, and public decision context.
2. Reconstruct a fixed number of synthetic authoritative states consistent with that information.
   The root chooser's observed private identities must be preserved exactly. Sampling must not
   condition on opponent identities, supply order, or achievement identities absent from the root
   observation.
3. Search each synthetic state through the real engine. Unknown cards therefore use sampled real
   identities rather than virtual cards or a sixth board color.
4. At every player decision, maximize when `decision.chooser` is the fixed root player and minimize
   otherwise. Deterministic engine work does not itself change max/min ownership.
5. Average each root semantic action's utility across determinizations. Hidden allocation and future
   draws are chance/belief samples, not adversarial choices.
6. Choose by stable legal-action order when final values tie exactly.

All legal actions exposed to a synthetic chooser are eligible, including opponent Meld actions.
Because an opponent knows their own sampled hand in that determinization, suppressing their Meld
choices would be both less realistic and more complicated.

The root action submitted to the real game must be selected from values aggregated across the full
sample set. Version 1 may allow deeper rollout choices to differ between determinizations; this is a
known determinization/strategy-fusion approximation, not permission to inspect the real hidden
state. Record it in the search version and revisit information-set action grouping only after the
first measured implementation.

The current sampler accepts only stable paid-turn boundaries. Milestone 4 must add a separately
reviewed player-safe search specification for starting-meld and pending-effect boundaries before
search can replace those fallback decisions. The live authoritative state must never be offered to
an agent or used directly to score hypothetical actions.

### 2.4 Search horizon

Depth is measured by completed player turns, never by semantic-action count. Nested dogma choices,
shared executions, demands, and automatic engine work are resolved inside the current turn.

For a root decision during the root player's active turn, search through:

```text
remainder of root turn 0
opponent turn 0
root turn 1
opponent turn 1
MINIMAX ENDS
```

This is four completed player-turns, approximately two full rounds. For a root-player choice during
an opponent turn, search through:

```text
remainder of opponent turn 0
root turn 1
opponent turn 1
MINIMAX ENDS
```

This is three completed player-turns. The current partial turn counts. Terminal states stop
immediately regardless of horizon.

Starting melds remain simultaneous secret submissions. When both choices are pending, one root
starting-meld decision uses this sampled maximin aggregation:

```text
max over root starting cards of
    mean over hidden-state determinizations of
        min over that sampled opponent hand's legal starting cards of
            continuation value after both choices commit
```

The hidden opponent hand is chance-sampled rather than selected adversarially. The opponent's
starting card is minimized only among cards actually legal in that sample, modeling a strong
opponent who knows their own hand but never sees the root's real pending submission.

If one starting choice has already been committed and remains secret while the other chooser acts,
the committed card is sampled as latent fixed history from the root observation; it is neither
copied from live authoritative state nor reopened as a minimizer branch. After both choices resolve,
starting-meld search covers the first four completed player turns, regardless of which player the
starting-card ordering makes first player.

### 2.5 Search budget and determinism

Two rounds are the target, not an assumption that every position is cheap. Before enabling the new
heuristic for generation:

- benchmark representative early-, middle-, late-, demand-, sharing-, and high-branching positions;
- measure legal branches, effect steps, nodes, transposition hits, determinizations, wall time, and
  peak memory;
- use deterministic iterative deepening by completed turn boundary;
- allocate the same frozen node/engine-transition budget independently to every `(root action,
  determinization)` route, so legal-action order cannot starve later alternatives;
- retain only fully completed depth iterations for a route; if no full iteration completes, use the
  deterministic immediate leaf value and record the cutoff;
- freeze the move ordering, transposition-key/version, cache scope, repeated-position rule, budget
  accounting, and incomplete-iteration behavior in the search descriptor;
- permit alpha-beta pruning and route-local strategic-state transposition caching only under those
  frozen semantics; and
- detect a repeated strategic position inside one search and cut it off to the leaf evaluator rather
  than inventing an Innovation draw.

The default determinization count and per-route node budget must be selected from measurements and
then frozen in the heuristic descriptor before training. Measure one, two, and four
determinizations; do not silently lower search quality only to finish a run. Work Package 0 must
publish a dated feasibility addendum with numeric maximum cutoff rate and minimum generation/arena
throughput gates; subsequent work may not start until those thresholds are fixed before seeing
strength results.

## 3. Hand-engineered leaf value

Every search value is centered and evaluated from the fixed root player's perspective.

Terminal ordering dominates every nonterminal value:

- root player is the sole winner: positive terminal sentinel;
- opponent is the sole winner: negative terminal sentinel;
- both or neither player is a winner in a terminal result: `0`.

Use explicit terminal ordering internally rather than serializing non-finite JSON numbers.

For a nonterminal state, define:

```text
achievement = 0.10 * clamp(root achievements - opponent achievements, -5, 5)
score       = 0.01 * clamp(root score points - opponent score points, -15, 15)

icons = 0.15 * mean over the six icon types of:
                  clamp(root visible count - opponent visible count, -3, 3) / 3

board = 0.10 * mean over the five board colors of:
                  clamp(root stack cards - opponent stack cards, -2, 2) / 2

hand = 0.02 * clamp(root hand count - opponent hand count, -5, 5)

value = achievement + score + icons + board + hand
```

The components contribute at most `0.50`, `0.15`, `0.15`, `0.10`, and `0.10` in either direction,
so every nonterminal value lies in `[-1, 1]`. “Visible icons” means icons functionally exposed by
the current top cards and splay geometry, not every icon printed on publicly identified covered
cards. The board-card term uses exact ordered stack sizes because `public-covered-v1` makes them
public.

These weights are an intentionally simple version-1 baseline chosen before experiments. Do not
tune them after looking at one arena result; any later formula change receives a new evaluator
version and a predeclared comparison.

## 4. Agent and learned-policy behavior

The stronger heuristic replaces `SimpleHeuristicAgent` as the primary heuristic baseline. It uses
sampled minimax for paid turn actions, starting melds, and nested effect choices.

The learned policy continues to use the frozen learned afterstate path only for actual
`DecisionKind.TURN_ACTION` choices. Starting melds and nested effect choices use the new heuristic.
Thus the learned policy differs from the heuristic policy at its paid turn-action choices, while
sharing the same fallback behavior elsewhere.

For version 1, minimax rollouts use the hand-engineered search policy for both players' future
choices, even when the real root policy is learned. The actual learned policy may choose a different
future paid action when that future decision arrives and reroots. This continuation mismatch is
accepted for the first version and must be explicit in the policy descriptor.

Changing fallback semantics changes the complete learned policy and the terminal distribution that
trains its value network. Do not make strength claims by attaching the new fallback to an old
checkpoint. Use old checkpoints only in explicitly labeled forensic or compatibility diagnostics.

## 5. Work package 0 — search contract and feasibility spike

Before broad implementation:

1. Freeze the search descriptor fields: search/evaluator and information-set-spec versions, sampling
   and hidden-allocation algorithm, sampler and selector seed derivation, horizon, determinization
   count, sample aggregation, per-route node budget and accounting, iterative-deepening completion
   rule, move ordering, transposition key/cache scope, tie-break, cycle cutoff, and fallback policy.
2. Require every complete learned/heuristic policy descriptor to reference the immutable search
   descriptor digest.
3. Design a scheduler-owned safe-search boundary for `TURN_ACTION`, `EFFECT_CHOICE`, and
   simultaneous `STARTING_MELD` decisions.
4. Prove that equivalent observations with different true hidden allocations produce identical
   search specifications and deterministic sampling distributions; root-known private identities
   remain fixed.
5. Build a committed representative decision corpus and measure the target two-round horizon.
6. Publish the numeric cutoff-rate and throughput gates, then freeze the production per-route budget
   and determinization count before inspecting playing-strength results.

Stop if implementing search would require agents to receive authoritative state or if sampler errors
would fall back to true-state expansion.

## 6. Work package 1 — observability and original reproduction

Ship diagnostics before changing the historical failing policy's behavior. Reproduce the original
candidate-as-player-1 heuristic game with:

- setup seed `50000`;
- checkpoint `sha256:652e12baed1bde3cac92aa474b53647891e7826fdfb19a4b95b73dd39949a04c`;
- arena policy `sha256:d66eb5cdd8dcfd2a29eeaaedf52e96b36e2f5c9045f7ffb44d26ea7f6258a1ad`;
- `temperature-softmax-v1`, temperature zero, and four determinizations;
- the original simple heuristic and `rulebook-private-covered-v1`; and
- the original 10,000-action ceiling.

A diagnostic trace must retain:

- source revision, game/setup IDs, manifest/config digests, and policy/checkpoint/fallback/search/
  sampler versions and RNG seed digests;
- complete compressed semantic-action prefix and decision/state hash chain;
- chooser, executor, activator, active player, decision kind, and paid actions remaining;
- legal actions and policy handling (`learned`, simple fallback, search fallback, or baseline);
- learned per-determinization values, means, selector scores, selected action, and tie/margin data;
- search root values, sample values, nodes, cutoffs, principal-variation summaries, and selected
  action;
- periodic strategic-position digests that exclude monotonic IDs but retain gameplay state and turn
  boundary; and
- no-progress telemetry: repeated paid-action windows, no-op dogmas, card movements, score/meld/
  tuck/return counts, splay changes, achievements, and supply changes.

All complete diagnostic traces are trusted private artifacts: actions, legal-action sets, sampled
principal variations, and chooser observations may expose identities legitimately known to one
player and may permit reconstruction of hidden authoritative state. They must never be published as
player-safe logs or supplied to a policy, encoder, or trainer. A separately generated redacted
summary may contain only public fields, digests, aggregate values, and explicitly approved excerpts.
Full authoritative snapshots require an additional explicit private-debug marker.

Normal terminal training games continue to use compact replay. Compressed full private traces are
required for explicit reproductions, action-ceiling failures, invariant/replay divergences, and
configured unusually long or strategically repetitive games. A failure is never converted to a
draw.

The original reproduction should determine which player chose each repeated action, whether hands
or other zones oscillated, why Agriculture had no choices, whether Machinery and `splay-left` were
no-ops, and what model values made Machinery win selection.

## 7. Work package 2 — stronger heuristic

Implement the frozen sampled-search contract and then validate it before using it for data
generation:

- exact terminal and leaf-evaluator unit tests;
- turn-horizon tests beginning at first/second paid actions and both players' effect choices;
- max/min ownership tests for demands, sharing, and consecutive decisions by one chooser;
- opponent Meld and private-hand tests across hidden-equivalent roots;
- public-covered observation and board-count tests;
- deterministic sampling, tie-break, node-budget, transposition, and cycle-cutoff tests;
- full replay and no-authoritative-state-leak tests; and
- representative tactical fixtures including achievement races, score races, Gunpowder/castle
  defense, no-op dogmas, optional effect choices, and unknown future draws.

Validation proceeds in two predeclared stages:

1. A 25-pair seat-swapped completion preflight against each of the original simple heuristic and
   random must finish every game below the 10,000-action ceiling with zero sampler, search,
   invariant, replay, or trace failures.
2. Before calling the policy stronger, a separately predeclared 100-pair seat-swapped comparison
   against the original simple heuristic must have a paired 95% utility lower bound above `0.5`.

Report W/D/L, paired utility/interval, seats, reasons, lengths, search nodes, budget cutoffs,
actions/s, RSS, and degenerate-game metrics. The frozen Work Package 0 cutoff-rate and throughput
gates must also pass before training; they cannot be relaxed after arena results are visible.

## 8. Work package 3 — restored primary candidate contract

Before training, freeze the primary policy template:

- selector `temperature-softmax-v1`;
- the new fallback/search descriptor digest;
- `public-covered-v1`;
- the existing learned afterstate architecture; and
- a checkpoint reference to be filled only by Work Package 4's compatible fresh training output.

This work package does not publish a runnable candidate before the checkpoint exists. Do not attach
the new fallback or information policy to a Milestone 3 checkpoint except in an explicitly labeled
compatibility experiment, and do not overwrite any Milestone 3 policy descriptor. A
repetition-aware policy may be derived as a matched ablation with the same new checkpoint and
fallback only after the primary runnable descriptor is frozen.

## 9. Work package 4 — fresh training run

The information policy and fallback policy change the observations, continuation behavior, replay
provenance, and target distribution. Start a new immutable run; do not resume `pilot-001` or reuse
its dataset/checkpoint as the Milestone 4 candidate.

The run sequence is:

1. generate heuristic/random bootstrap games using the stronger heuristic;
2. materialize a `public-covered-v1` dataset and verify exact replay/online encoding equality;
3. train a fresh value checkpoint with the existing encoder-v1/model architecture unless the
   feasibility work proves an incompatibility;
4. generate learned self-play using `temperature-softmax-v1` for paid actions and the stronger
   heuristic fallback;
5. train the next candidate checkpoint;
6. publish the runnable primary policy descriptor from the frozen Work Package 3 template; and
7. publish the iteration report.

Choose game counts only after measured search throughput and dataset-memory estimates. Retain the
Milestone 3 lesson that materialized arrays, not compressed logs, dominate disk use. Zero sampler,
replay, integrity, illegal-action, or action-ceiling failures remain hard gates. The selected
checkpoint must also beat the train-mean constant reference on held-out Brier score, show finite
predictions/losses, and improve beyond its first epoch or record a separate project decision before
proceeding to candidate evaluation.

## 10. Work package 5 — evaluation

Predeclare fixed seed sets before play. At minimum evaluate:

- new heuristic versus original simple heuristic;
- new heuristic versus random;
- primary learned candidate versus random;
- primary learned candidate versus original simple heuristic; and
- primary learned candidate versus the new heuristic.

Run seed `50000` first as an explicit two-seat diagnostic under the complete new runnable policy,
with private tracing enabled. Record whether any strategic cycle or action ceiling remains. Absence
of the old cycle is evidence about the complete new policy, not proof that the original selector
was independently repaired. Then run the fixed 25-pair seat-swapped preflight without adaptive
extension. Report each opponent separately and make completion rate the first metric. Include
search throughput/cutoffs and strategic-cycle telemetry alongside standard arena results.

The repetition-aware selector may be evaluated as a predeclared matched ablation, but it does not
replace the primary temperature-softmax candidate merely by completing an arena. A 200-pair
promotion comparison remains a later step requiring a compatible predeclared incumbent and the
normal paired lower-bound criterion.

## 11. Stop conditions

Stop and diagnose rather than weakening safety or silently reducing search if any of these occur:

- equivalent player observations produce search specifications conditioned on true hidden state;
- pending-effect sampling cannot reproduce the root decision and legal actions;
- search uses an action unavailable in the chooser's observation or synthetic state;
- target two-round searches exceed the frozen numeric budget-cutoff gate;
- heuristic arenas cycle, exceed action ceilings, or fall below the frozen throughput gate;
- public-covered observations expose opponent hand/score identities, supply order, achievement
  identities, or secret setup choices beyond the configured reveal rules;
- old replay/policy artifacts are silently interpreted under the new information policy;
- generated replay, dataset, checkpoint, or policy compatibility checks fail; or
- training/evaluation results cannot be tied to immutable search and fallback descriptors.

## 12. Non-goals

Milestone 4 does not initially include:

- MCTS, CFR, a policy network, recurrence, or learned belief models;
- exact game-theoretic solution of imperfect information;
- a second abstract Innovation engine or virtual sixth board color;
- arbitrary information-memory inference across prior hidden-zone movements;
- changing Innovation termination rules or declaring repeated positions draws;
- changing the encoder layout or value-network architecture unless a separately recorded
  feasibility result makes it necessary; or
- promotion-scale claims before the stronger heuristic and primary candidate pass fixed preflights.

## 13. Completion checklist

- [ ] Search descriptor, evaluator formula, horizon, sampling contract, and numeric feasibility
      gates frozen.
- [ ] `public-covered-v1` adopted with compatibility and non-leak tests.
- [ ] Original seed-50000 failure reproduced with a complete diagnostic trace.
- [ ] Stronger heuristic implemented for paid actions, setup, and effect choices.
- [ ] Fixed heuristic-versus-simple/random validation completed.
- [ ] Primary learned policy template restored to `temperature-softmax-v1` and instantiated with the
      fresh compatible checkpoint.
- [ ] New seed-50000 two-seat diagnostic completed without unpublished/incomplete outcomes.
- [ ] Fresh bootstrap, learned generation, training, and report completed.
- [ ] Fixed candidate preflight against random, simple heuristic, and new heuristic completed.
- [ ] Repetition-aware selector retained only as an explicit comparator.
- [ ] `make check` and relevant fuzz/determinization/search soaks pass.
- [ ] Focused commits pushed to `origin`.
