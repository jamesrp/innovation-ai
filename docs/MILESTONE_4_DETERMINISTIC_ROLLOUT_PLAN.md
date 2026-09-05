# Milestone 4 Deterministic Turn-Rollout Recovery Plan

**Date:** September 5, 2026  
**Planning status:** proposed for owner review; not yet approved for implementation.  
**Purpose:** replace infeasible exhaustive continuation after each root action with one versioned,
player-safe deterministic continuation rollout, while still evaluating every legal root action over
common hidden-state determinizations.

This plan is a recovery proposal for the feasibility stop recorded in
`docs/MILESTONE_4_FEASIBILITY_ADDENDUM.md`. It supersedes neither the historical measurements nor
the current binding search contract until the owner explicitly approves implementation.

## 1. Decision requested

Adopt a new heuristic line named **sampled deterministic turn rollout**:

1. Build the same player-safe root information-set specification used by the Milestone 4 search.
2. Sample the same fixed set of observation-consistent synthetic authoritative states for every
   legal root action.
3. Evaluate **every legal root action** in every determinization.
4. After the root action, choose exactly one legal continuation action at each later decision with
   a dedicated internal adapter for the existing deterministic
   `simple-heuristic/printed-card-observation-v1` policy.
5. Continue through the real engine until the current active player's turn completes or the game
   terminates.
6. Score the resulting state with `hand-engineered-leaf-v1` from the fixed root chooser's
   perspective.
7. Average each root action's values across determinizations and choose the first maximum in stable
   root legal-action order.

This deliberately replaces minimax continuation with a policy rollout. It is not described as
minimax, adversarial search, or an exact solution to imperfect information.

## 2. Rationale

The one-turn exhaustive benchmark showed that the problem is concentrated in continuation fanout:
seven ordinary corpus roots completed quickly, while one 14-action late-game root consumed almost
all wall time and still left routes incomplete at 1,600 transitions. The root actions themselves
must remain exhaustive so expensive or unusual legal actions are not silently suppressed. The
combinatorial multiplication occurs after those root actions, when search branches over every
second paid action and every nested effect choice.

A single deterministic continuation path changes route cost from a branching tree to approximately
linear work in the number of decisions remaining in the turn. It retains:

- root-action comparison under a common information set;
- real card identities and real engine effects in every synthetic state;
- complete resolution through a strategically meaningful turn boundary;
- strict hidden-information safety; and
- deterministic, replayable policy identity and telemetry.

It gives up adversarial continuation and may misvalue a root action when the simple continuation
policy would choose a weak later action. That approximation is explicit, versioned, and subject to
fixed tactical and arena validation.

## 3. Frozen proposed algorithm

### 3.1 Root semantics

The root chooser remains the fixed evaluation viewpoint. The implementation receives only an
`InformationSetSpec` and synthetic states reconstructed from it; the live authoritative state is
not accepted by the rollout selector.

For an ordinary paid-turn or pending-effect root:

```text
for each legal root action in stable order:
    for each common determinization:
        apply the root action through the real engine
        while nonterminal and the current active player's turn has not completed:
            obtain the synthetic chooser's real Decision
            choose one action with simple-heuristic/printed-card-observation-v1
            apply it through the real engine
        evaluate the terminal or boundary state from the root chooser's perspective
    root value = arithmetic mean of the determinization values
select the first root action with the maximum root value
```

The root action is never selected by the continuation heuristic. Every root legal action receives
one independently recorded route per determinization.

### 3.2 Continuation policy

The continuation policy is exactly the existing
`simple-heuristic/printed-card-observation-v1` behavior, wrapped by a dedicated internal rollout
adapter rather than the scheduler's caller-overridable fallback:

- the adapter receives the synthetic chooser's `Decision`, including only that chooser's
  observation;
- it cannot receive or retain the synthetic authoritative state;
- it validates that the returned action is legal before transition;
- it chooses one deterministic legal action;
- its existing stable card/action tie behavior is preserved; and
- the rollout descriptor references its complete canonical agent descriptor, legal-action schema,
  card-data fingerprint, and effect/rules compatibility fields.

The continuation action is chosen from each decision's actual chooser perspective, not from the
fixed root player's perspective. Opponents therefore follow the same known rollout policy rather
than minimizing root utility. Different determinizations may produce different continuation
choices because synthetic choosers know their own sampled private cards. This is an explicit
root-determinization policy-rollout approximation.

No learned checkpoint, random choice, temperature, recent-action history, or true hidden state is
used inside version 1 rollouts.

### 3.3 Horizon and exact stop boundary

The horizon remains **one completed active-player turn**, anchored to the synthetic state's turn
number rather than inferred from chooser changes:

```text
play root:
    target_turn = state.turn_number
    apply the root action
    continue only while nonterminal and state.turn_number == target_turn

setup root:
    commit both sampled starting choices
    target_turn = resulting play state's turn_number
    continue only while nonterminal and state.turn_number == target_turn
```

Reaching the next turn's decision boundary completes the route; the rollout must not choose an
action there. Thus:

- a root at the first paid action includes the root action, all nested resolution, the second paid
  action selected by the continuation policy, and all of that action's resolution;
- a root at the second paid action includes the root action and its complete resolution;
- the game's first turn, which has only one paid action, stops after that action resolves;
- a pending-effect root completes the turn currently in progress, regardless of whether the root
  chooser is the active player; and
- terminal states stop immediately.

The current partial turn counts. Any transition that skips more than one turn or exposes no single
continuation decision before the boundary is an invariant failure.

### 3.4 Starting melds

Starting melds use one deterministic sampled opponent response, not minimax. For each
determinization where both choices are pending:

1. Obtain the sampled opponent's own precommit starting-meld `Decision`.
2. Select one semantic starting-card choice with the frozen simple-heuristic rollout adapter.
3. Preserve that selected card choice across every root starting-card candidate in the same
   determinization, so the opponent response cannot condition on the root's secret candidate.
4. After applying a root candidate, rebind the preserved semantic card choice to the actual pending
   opponent decision ID and submit it through the real engine.
5. Follow one deterministic continuation path through the first completed turn.

The resulting root formula is:

```text
max over root starting cards of
    mean over hidden-state determinizations of
        deterministic rollout after the sample's preselected opponent starting card
```

When one starting card is already committed and secret, it remains sampled latent history. It is
not copied from live state and is not reopened or replaced by the continuation policy.

This intentionally supersedes the provisional sampled-maximin setup rule only if the rollout
experiment is approved. It models a fixed known opponent policy rather than a clairvoyant opponent
that may counter each secret root candidate separately.

### 3.5 Leaf and terminal values

Version 1 keeps `hand-engineered-leaf-v1` unchanged. Sole root wins score the positive terminal
sentinel, sole root losses score the negative terminal sentinel, and joint/neither terminal winners
score zero. Nonterminal boundary states use the frozen achievement, score, visible-icon, board, and
hand formula.

The evaluator is always centered on the root chooser, including roots where that chooser is making
a decision during the opponent's turn.

### 3.6 Ties and aggregation

- Determinization values use `math.fsum` in stable sample order, divided by the fixed sample count,
  for each root action.
- Exact root-value ties resolve by stable original legal-action order.
- Continuation ties are whatever the immutable simple heuristic already specifies.
- No random tie-breaking or selector seed is needed after determinizations are sampled.

### 3.7 Safety ceiling and failures

Rollout version 1 has no quality budget, iterative deepening, alpha-beta pruning, or immediate-leaf
fallback. A provisional **1,024 total semantic engine-transition safety ceiling per
`(root action, determinization)` route** guards nontermination across repeated Decisions. The count
includes the root transition, a starting-response transition when present, and every continuation
transition. The route fails before attempting transition 1,025.

This ceiling cannot interrupt a single hung `apply_action`; process supervision and external run
timeouts remain responsible until a separately specified internal effect-step watchdog exists.

Hitting the ceiling is a hard rollout failure:

- do not evaluate the partial state;
- do not fall back to the live state or simple root choice;
- do not publish an incomplete arena/training result; and
- emit a trusted-private failure trace under the existing diagnostic rules.

The ceiling is identity-bearing and may be changed before production freeze only by creating a new
descriptor. Feasibility must show at least 2x headroom between the maximum measured route and the
ceiling.

## 4. Versioning and compatibility

Do not reinterpret the existing `root-sampled-minimax-*` descriptors. Add a distinct immutable
contract, provisionally:

```text
algorithm: root-sampled-deterministic-turn-rollout-v1
descriptor kind/schema: rollout-search / v1
evaluator: hand-engineered-leaf-v1
information policy: public-covered-v1
information-set spec: player-safe-search-spec-v1
sampler/RNG/hidden allocation: existing frozen Milestone 4 versions
continuation policy: simple-heuristic/printed-card-observation-v1
continuation branching: single-action-per-decision-v1
hypothetical/live mismatch: simple-rollout-then-reroot-v1
turn boundary: anchored-one-completed-active-player-turn-v1
sampling: fixed-count-root-determinization-v1
sample ordering/aggregation: stable-sample-order-math-fsum-mean-v1
starting meld: preselect-sampled-opponent-simple-choice-then-rollout-v1
root ordering/action keys/tie break: existing stable semantic versions
route ceiling: 1024 all-in semantic engine transitions, strict failure
engine/rules/actions/decisions/cards/effects: exact installed compatibility fingerprints
```

Implement a new `RolloutSearchDescriptor` rather than overloading minimax-only fields such as
alpha-beta and transposition semantics. It must include the descriptor kind and schema, algorithm
and determinization count, all information/sampling/seed versions, evaluator and continuation-agent
identity, exact boundary and ordering semantics, strict ceiling accounting, and installed engine and
card/effect compatibility fingerprints. Add a tagged decoder that dispatches by explicit
descriptor kind; never infer rollout semantics from old minimax fields.

Both descriptor types expose a content-derived `descriptor_id` and a small common protocol consumed
by policy scheduling and diagnostics. Existing schema-v1 minimax descriptors and historical policy
artifacts remain readable and immutable. Do not create a production rollout constant or silently
replace `PRODUCTION_SEARCH_DESCRIPTOR` until feasibility and validation pass; register only an
explicit experimental descriptor during R1–R4.

The primary heuristic agent becomes `sampled-turn-rollout-heuristic-v1`. Learned policies require a
new policy-descriptor schema v3 with explicit fallback algorithm kind/descriptor and hypothetical
continuation-policy fields; schema v2's minimax-named continuation semantics must not be reused.
Compact replay and generation manifests likewise receive a generic per-seat decision-algorithm
reference rather than storing rollout identity under a misleading legacy minimax field.

A rollout policy is incompatible by training lineage with every Milestone 3 checkpoint/policy and
every provisional minimax policy, even where the encoder/model tensors are technically loadable.
`public-covered-v1` and `temperature-softmax-v1` remain the intended Milestone 4 information policy
and primary learned selector.

Hypothetical continuations use the simple heuristic, but an actual rollout-controlled player
reroots at its next real decision and again evaluates every legal root action. This receding-horizon
mismatch is identity-bearing and must be recorded in policy descriptors and reports.

## 5. Telemetry contract

For every trusted-private root decision, record:

- rollout descriptor and continuation-agent descriptor IDs;
- information-set digest and sampler seed digest;
- stable root action keys and determinization count;
- per-route final value and leaf components;
- root, continuation, setup-response, and total engine transitions;
- ordered continuation action keys with chooser and decision kind;
- completed-turn count and terminal reason, if any;
- safety-ceiling status;
- per-action sample values, arithmetic means, ties, and selected root action; and
- wall time and process RSS in benchmark/arena reports, outside deterministic content identity.

Use “rollout path” rather than “principal variation.” Minimax nodes, alpha-beta cutoffs,
transposition hits, and minimizer/maximizer labels are not emitted for this algorithm. Root and
continuation action keys, sampled observations, per-sample values, and full rollout paths are
trusted-private by construction. The public projection contains only approved aggregate counts,
histograms, timing, and digests and cannot reconstruct a synthetic chooser's private cards.

A synthetic opponent may legitimately see its own sampled hand and score identities through its
own Decision. Tests must prove those identities never reach the root policy, another chooser,
public output, trainer, or live-state bridge.

## 6. Work packages

### Work Package R0 — approve and freeze the experiment

Before implementation:

1. Approve this policy-rollout approximation and its deterministic starting-meld response.
2. Freeze the provisional algorithm, continuation policy, horizon, safety ceiling, evaluator,
   sampling, aggregation, and tie semantics.
3. Freeze the feasibility gates in Section 7 before measuring playing strength.
4. Record that the September 3 rejection of selective continuation is superseded only for this new
   versioned experiment; historical descriptors and evidence remain unchanged.

### Work Package R1 — contracts and reference implementation

- Add `RolloutSearchDescriptor`, the tagged descriptor decoder, and a shared scheduler-facing
  descriptor protocol.
- Add route/root rollout telemetry with deterministic serialization.
- Implement the selector in `src/innovation_ai/search/rollout.py` using only information-set specs,
  synthetic states, real engine transitions, the simple heuristic, and the leaf evaluator.
- Keep `src/innovation_ai/search/minimax.py` behavior unchanged.
- Add exact reference tests before optimizing execution.

### Work Package R2 — scheduler, agents, and provenance

- Route the new algorithm through the trusted scheduler boundary without giving an agent live
  authoritative state.
- Add `sampled-turn-rollout-heuristic-v1` as a distinct baseline descriptor.
- Generalize self-play and arena descriptor maps to accept explicitly tagged minimax or rollout
  contracts and require policy/algorithm/implementation-family agreement.
- Add policy-descriptor schema v3 and generic per-seat algorithm provenance to replay/generation
  schemas; preserve exact schema-v1/v2 decoding.
- Reject missing, mismatched, minimax/rollout-confused, or legacy-incompatible descriptors.
- Preserve strict failure behavior; never fall back from rollout failure to true-state expansion.

### Work Package R3 — focused correctness validation

Add tests for:

- every root legal action receiving every common determinization;
- continuation choosing exactly one legal action per decision;
- the rollout stopping at the next completed turn and never entering the following turn;
- first- versus second-paid-action roots;
- own and opponent pending-effect roots;
- consecutive decisions by the same chooser and alternating shared/demand choosers;
- terminal outcomes during the root action and during continuation;
- deterministic starting-card preselection shared across root candidates and one-latent-choice
  handling;
- stable ties and arithmetic aggregation;
- hidden-equivalent live roots producing identical specs, samples, and selected actions;
- root-known private identities remaining fixed;
- continuation agents receiving only synthetic chooser Decisions;
- no opponent hand/score, supply-order, achievement, or secret-setup leakage;
- strict safety-ceiling failure and trusted-private tracing;
- serialization and historical minimax/replay compatibility; and
- deterministic replay/online encoding equality under `public-covered-v1`.

Use tactical fixtures for achievement and score races, Gunpowder/castle defense, optional choices,
no-op dogmas, sharing, demands, and unknown future draws. Freeze expected root-action selections or
rankings before running them when the fixture is intended as a tactical gate; otherwise label the
fixture semantic correctness only. Tests should demonstrate behavior, not tune evaluator weights or
continuation rules.

### Work Package R4 — feasibility rerun

Extend the committed eight-root corpus with at least:

- the late 14-action root that stopped exhaustive search;
- a high-choice dogma on the first paid action;
- a turn with multiple nested effect decisions;
- a route where continuation reaches an immediate terminal result; and
- a no-progress/no-op continuation route.

Measure one, two, and four determinizations on the two-core CPU host. Freeze corpus/configuration
digests, seeds, concurrency, validation level, tracing mode, warmup, repetition count, and timing
method before execution. Use repeated warmed runs and report median and p95 per-root timing, plus
raw maxima diagnostically. Report root actions, rollout steps, engine transitions, maximum route
length, wall time, roots/s, RSS, failures, action-agreement/value-variance diagnostics across sample
counts, and action-selection determinism. Publish JSON and Markdown artifacts with both
deterministic counter digests and complete-file digests that include timing.

Before freezing the 1,024-transition ceiling, add all-card/high-choice fixtures and seeded
route-length soaks beyond the eight-root corpus. Corpus success alone does not establish headroom.

The target production setting is **two determinizations**, proposed as an explicit cost/quality
choice: one sample is too brittle for hidden allocation, while four doubles the target cost. The
one/two/four agreement and value-variance report informs review but must not be tuned from playing
strength. If two determinizations fail the frozen gate, stop and discuss rather than silently
reducing production to one. One and four remain scaling diagnostics.

### Work Package R5 — heuristic validation

Only after R4 passes:

1. Run the existing fixed 25-pair seat-swapped completion preflight against both random and the
   original simple heuristic.
2. Require every game to finish below 10,000 actions with zero sampler, rollout, invariant, replay,
   trace, or action-ceiling failures.
3. Run the separately fixed 100-pair seat-swapped comparison against the original simple heuristic.
4. Call the rollout heuristic stronger only if the paired 95% utility lower bound is above 0.5.

Report W/D/L, paired intervals, seats, terminal reasons, game lengths, rollout route lengths,
transitions, throughput, RSS, repeated strategic positions, and no-progress metrics. Seed sets must
be committed before play and must not be extended adaptively.

### Work Package R6 — resume Milestone 4 training

Only after heuristic validation passes:

- freeze the primary `temperature-softmax-v1` learned-policy template with the production rollout
  descriptor and `public-covered-v1`;
- choose bootstrap and learned game counts from measured end-to-end throughput and memory, recording
  the counts before generation;
- generate fresh heuristic/random bootstrap replay;
- materialize and verify the new dataset;
- train a fresh bootstrap value checkpoint;
- distinguish technical encoder/model loadability from approved rollout training lineage in every
  checkpoint and policy compatibility check;
- generate learned self-play with rollout fallback;
- train the next candidate checkpoint; and
- publish a runnable primary policy and immutable iteration report.

Do not resume `pilot-001`, attach the rollout fallback to a Milestone 3 checkpoint, or use an old
policy as the new candidate. Existing held-out Brier, finite-loss, improvement-beyond-epoch-one,
integrity, replay, sampler, and action-ceiling gates remain binding.

### Work Package R7 — candidate evaluation

Run seed 50000 first in both seats with private tracing, then the fixed 25-pair candidate preflight
against random, original simple heuristic, and the new rollout heuristic. Completion and throughput
remain gates before strength claims. A matched repetition-aware selector is optional and cannot
replace the primary temperature-softmax candidate merely by finishing.

Promotion-scale evaluation remains outside this recovery experiment unless a compatible incumbent
and the normal predeclared 200-pair comparison are available.

## 7. Frozen proposed feasibility gates

These thresholds must be approved before implementation measurements are used for a go/no-go
choice:

- zero sampler, rollout, invariant, replay, trace, or serialization failures;
- **100%** of routes reach the terminal or requested completed-turn boundary;
- zero safety-ceiling hits and maximum measured route length at most 512 transitions, preserving 2x
  headroom under the proposed 1,024-transition ceiling;
- at the target two determinizations, warmed median aggregate throughput of at least **2.5 searched
  roots/second** over the full corpus;
- per-root warmed p95 wall time no greater than **1.0 second** at two determinizations, including the
  frozen late/high-choice root;
- deterministic counters, action values, and selections identical on an immediate rerun;
- generation smoke throughput of at least **2.0 committed actions/second**; and
- arena smoke throughput of at least **1.0 committed action/second**.

Throughput cannot compensate for incomplete routes or failures. Strength cannot compensate for
missing throughput. The gates may be made stricter before seeing strength results, but must not be
relaxed after feasibility or arena outcomes are known.

## 8. Stop conditions

Stop rather than weakening safety or silently changing policy if:

- continuation requires live authoritative state or observes information absent from its synthetic
  chooser Decision;
- equivalent player observations produce hidden-state-conditioned specs or sampler distributions;
- a continuation action is illegal, nondeterministic, or selected by the wrong chooser;
- a route crosses into the next player's turn;
- a route hits the safety ceiling;
- the target two-determinization feasibility gates fail;
- heuristic arenas cycle, exceed the action ceiling, or miss throughput;
- rollout identity is absent from any replay, checkpoint, policy, or report;
- old artifacts are silently reinterpreted under rollout semantics; or
- training/evaluation cannot be tied to immutable descriptor digests and fixed seeds.

## 9. Non-goals

This recovery plan does not include:

- claiming minimax or optimal opponent modeling;
- pruning the root legal-action set;
- learned, random, temperature-based, or repetition-aware continuation;
- tuning the simple heuristic or hand-engineered leaf weights from the feasibility corpus;
- MCTS, CFR, policy networks, or learned belief models;
- a second Innovation rules engine;
- converting failures or strategic repetitions into draws; or
- strength claims before fixed completion, throughput, and arena gates pass.

## 10. Completion checklist

- [ ] Owner approves the deterministic rollout approximation and superseding experiment decision.
- [ ] Rollout descriptor, continuation policy, horizon, evaluator, sampling, and gates frozen.
- [ ] Reference rollout and telemetry implemented with minimax compatibility preserved.
- [ ] Scheduler, policy identity, replay, training, arena, and diagnostic routing implemented.
- [ ] Focused safety, determinism, horizon, setup, effect, and compatibility tests pass.
- [ ] Expanded feasibility corpus passes at two determinizations.
- [ ] Fixed heuristic completion and strength arenas pass.
- [ ] Fresh Milestone 4 training and runnable primary policy complete.
- [ ] Fixed candidate diagnostics and preflights complete.
- [ ] `make check` and relevant determinization/search soaks pass.
- [ ] Focused commits pushed to `origin`.
