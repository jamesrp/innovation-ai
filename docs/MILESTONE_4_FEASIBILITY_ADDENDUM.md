# Milestone 4 feasibility and diagnostic addendum

**Date:** September 3, 2026  
**Status:** both exhaustive search horizons tested for Milestone 4 are infeasible on the two-core
CPU host. The original 4/3/4-turn gate failed, and the owner-approved one-completed-turn rerun also
failed the unchanged cutoff, fallback, category-completion, and throughput gates. No production
search descriptor, route budget, or determinization count is frozen. Heuristic arenas, training,
and strength evaluation remain stopped pending a project decision.

## One-completed-turn feasibility rerun

The revised exhaustive horizon was measured from source revision
`0f698ce090479f3d2737e9f60239af13c75dc3c0` on September 3, 2026. The committed eight-root corpus
was swept at 400, 800, and 1,600 recursive engine transitions per independently budgeted
`(root action, determinization)` route, with one, two, and four determinizations. Sampling remained
strict and player-safe. All 72 root measurements completed with zero sampler, search, invariant, or
benchmark failures.

The existing pre-strength gates were applied unchanged; no threshold was relaxed after seeing the
one-turn result. In particular, a production choice still required at most 5% budget-cutoff routes,
at most 1% immediate-leaf routes, at least 95% full-depth completion in every corpus category, at
least 2.0 roots/s at one determinization, and at least 0.75 roots/s at four determinizations.

| route budget | determinizations | routes | cutoff rate | immediate-leaf rate | full-depth rate | late-root full depth | wall time | roots/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 400 | 1 | 31 | 22.6% | 22.6% | 77.4% | 50.0% | 16.03 s | 0.499 |
| 400 | 2 | 62 | 22.6% | 22.6% | 77.4% | 50.0% | 29.89 s | 0.268 |
| 400 | 4 | 124 | 22.6% | 22.6% | 77.4% | 50.0% | 53.06 s | 0.151 |
| 800 | 1 | 31 | 12.9% | 12.9% | 87.1% | 71.4% | 20.24 s | 0.395 |
| 800 | 2 | 62 | 12.9% | 12.9% | 87.1% | 71.4% | 88.36 s | 0.091 |
| 800 | 4 | 124 | 12.9% | 12.9% | 87.1% | 71.4% | 131.29 s | 0.061 |
| 1,600 | 1 | 31 | 9.7% | 9.7% | 90.3% | 78.6% | 57.17 s | 0.140 |
| 1,600 | 2 | 62 | 9.7% | 9.7% | 90.3% | 78.6% | 111.96 s | 0.071 |
| 1,600 | 4 | 124 | 9.7% | 9.7% | 90.3% | 78.6% | 260.05 s | 0.031 |

Every setup, early-game, second-action, own-effect, opponent-demand, and opponent-shared root
completed the one-turn horizon at every tested budget. The late high-branching root, which exposes
14 legal root actions, was the binding failure: 7, 4, and 3 of its 14 one-determinization routes
fell back to the immediate leaf at budgets 400, 800, and 1,600 respectively. Raising the budget
improved completion but made already-failing throughput worse. Even the cheapest tested descriptor
missed the one-determinization throughput gate by about 4x and the four-determinization gate by
about 5x.

The complete local evidence is retained at:

```text
artifacts/runs/milestone-4/search-feasibility-one-turn/search-feasibility.json
artifacts/runs/milestone-4/search-feasibility-one-turn/search-feasibility.md
```

The deterministic counter/content digest is:

```text
sha256:d9d133b581fec7fa073f797d21556439f9db79b1eaa79e271cccd89b7d124a29
```

Because that digest deliberately excludes timing and RSS, the complete JSON file digest is also
recorded:

```text
sha256:421c6cb597ba61543b15306abfbaa6d2a4599573a938cc373f492cee9faa7b8d
```

The host exposed two AMD EPYC 9554P vCPUs, 7.7 GiB RAM, and no swap. Peak measured process RSS was
36.0 MiB, so CPU search cost rather than memory was the binding resource.

**Decision:** stop Milestone 4 before heuristic validation, fresh training, or strength evaluation.
No tested budget/determinization pair qualifies as the production heuristic, and increasing only
the route budget cannot plausibly satisfy both completion and throughput. The next step requires an
explicit project decision about search optimization, search approximation, evaluator/policy design,
or additional compute.

## Superseding one-turn decision

**Owner decision, September 3, 2026:** replace the exhaustive 4/3/4 completed-turn horizon with an
exhaustive **one completed-turn horizon for all paid-turn, pending-effect, and starting-meld roots**.
Do not add choose-two, beam, or other selective continuation in Milestone 4. The rationale is that
selecting future branches with the simple version-1 leaf evaluator would add another arbitrary
approximation whose benefit has not been established.

The search version is `root-sampled-minimax-one-completed-turn-v1`; changing the horizon created
a new descriptor and policy identity. All real root actions, opponent Meld actions, nested effect
choices, stable legal-order ties, common determinizations, route-local transpositions, and cycle
cutoffs remain as specified. Only the completed-turn horizon changed.

The current one-turn implementation default still uses the provisional 400-transition,
one-determinization descriptor:

```text
sha256:a3de2116646c297b30b75e25fc4ab065ef1a0674f89b932eecd9ef5eea2b067e
```

Its budget and determinization count are not production-frozen. The one-turn rerun above found no
qualifying replacement identity, so this provisional descriptor must not be used for heuristic
validation, training, or strength evaluation.

The earlier 4/3/4 gates and measurements below are retained unchanged as historical evidence.

## Frozen pre-strength gates for the original 4/3/4 experiment

These thresholds were declared before any Milestone 4 playing-strength arena:

- no sampler, search, invariant, replay, or trace failures;
- at most **5%** of independently budgeted routes may end at the node budget before completing the
  requested horizon;
- at most **1%** of routes may fall back to the immediate leaf because no completed-turn iteration
  finished;
- at least **95%** of routes in every representative corpus category must complete the full frozen
  horizon (four completed turns for root-active/setup decisions, three for choices during the
  opponent's turn);
- at least **2.0 searched root decisions/second** at one determinization and **0.75 searched root
  decisions/second** at four determinizations on the two-core CPU host; and
- subsequent end-to-end generation and arena smoke runs must sustain at least **2.0 committed
  actions/second** and **1.0 committed action/second**, respectively.

The route cutoff thresholds are the binding Work Package 0 gates. Throughput cannot compensate for
failing the requested complete-turn horizon.

## Historical 4/3/4 search contract

The content-addressed historical provisional descriptor was:

```text
sha256:96a44eb380c292b8fd8aae7cf005331b9cb6d37665366d0705ec6e4b7360bd7b
```

That descriptor recorded:

- deterministic root-sampled minimax;
- the exact `hand-engineered-leaf-v1` formula;
- `player-safe-search-spec-v1` and without-replacement hidden allocation;
- root/opponent/setup horizons of 4 **4/3/4 completed player turns**;
- one determinization and 400 recursive engine transitions per root-action/sample route;
- cumulative iterative deepening, with only fully completed depth iterations retained;
- root and mandatory setup-response transitions outside the route budget;
- stable legal-action order, stable first-action tie breaks, and arithmetic sample means;
- stable-order alpha-beta;
- route-local, exact-only transposition entries keyed by `strategic-state-v1`; and
- path-local repeated strategic positions cut to the leaf evaluator, never converted to draws.

That descriptor was never training-approved because it failed the frozen feasibility gates below.

## Representative feasibility corpus

The committed corpus covers:

1. both starting choices pending;
2. one latent committed starting choice;
3. an early first paid turn;
4. a position with one paid action remaining;
5. an own-turn effect choice;
6. an opponent demand choice;
7. an opponent shared-effect choice; and
8. a late, high-branching position.

The complete 16/32/128-budget sweep, including one, two, and four determinizations, is retained at:

```text
artifacts/runs/milestone-4/search-feasibility/search-feasibility.json
artifacts/runs/milestone-4/search-feasibility/search-feasibility.md
```

Its deterministic content digest is:

```text
sha256:bc6fc54fa999bacb5f2bf17763aaaebdf5964ad19c0ed2edc75c0e62883898c9
```

Aggregate results across all eight roots and all three determinization counts:

| route budget | measurements | routes | budget cutoff rate | immediate-leaf rate | wall time | engine transitions/s | roots/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 24 | 217 | 100.0% | 48.4% | 5.42 s | 685.7 | 4.43 |
| 32 | 24 | 217 | 100.0% | 45.2% | 13.47 s | 533.6 | 1.78 |
| 128 | 24 | 217 | 100.0% | 39.6% | 60.58 s | 462.5 | 0.40 |

The provisional 400-transition, one-determinization descriptor was then measured on all eight
roots:

| corpus root | routes | retained completed-turn depths | cutoff routes | immediate leaves | wall time |
|---|---:|---|---:|---:|---:|
| both-pending setup | 2 | 3:2 | 2 | 0 | 0.94 s |
| one-latent setup | 2 | 2:2 | 2 | 0 | 1.57 s |
| early first paid turn | 3 | 3:3 | 3 | 0 | 1.85 s |
| one paid action remaining | 4 | 2:1, 3:3 | 4 | 0 | 2.56 s |
| own-turn effect choice | 2 | 3:2 | 2 | 0 | 0.97 s |
| opponent demand choice | 2 | 2:2 | 2 | 0 | 1.35 s |
| opponent shared choice | 2 | 2:2 | 2 | 0 | 1.02 s |
| late high branching | 14 | 0:7, 1:6, 2:1 | 14 | 7 | 14.93 s |

Thus the provisional descriptor had a **100% route cutoff rate**, a **22.6% immediate-leaf rate**,
and only **0.32 searched roots/second** overall. No route in the corpus completed the requested full
horizon. The late root alone took almost 15 seconds and left half its routes at the immediate leaf.
This fails all three search-quality gates and the one-determinization root-throughput gate.

## Exact historical seed-50000 reproduction

The original pathological game was reproduced before changing its historical policy behavior:

- setup seed: `50000`;
- exact game ID: `001-000000:candidate-player-1`;
- checkpoint: `sha256:652e12baed1bde3cac92aa474b53647891e7826fdfb19a4b95b73dd39949a04c`;
- policy: `sha256:d66eb5cdd8dcfd2a29eeaaedf52e96b36e2f5c9045f7ffb44d26ea7f6258a1ad`;
- selector: `temperature-softmax-v1`, temperature zero;
- four learned determinizations;
- fallback/opponent: original `simple-heuristic`;
- information policy: `rulebook-private-covered-v1`; and
- action ceiling: 10,000.

The reproduction reached exactly **10,000 actions** in **269.50 seconds**. The full trusted-private
trace and public redaction are retained under:

```text
artifacts/runs/milestone-4/original-seed-50000/traces/
```

The candidate-as-player-1 private trace is 1.5 MiB compressed and contains the complete action,
decision, state-hash, strategic-digest, learned-value, and seed-digest chain.

The trace answers the original diagnostic questions:

- **Player 1 (learned)** repeatedly chose `Machinery`; **Player 1's simple fallback** then chose
  `splay-left`; **Player 2 (simple baseline)** spent both paid actions on `Agriculture`.
- In the last 500 actions the exact counts were 167 `Machinery`, 167 `splay-left`, and 166
  `Agriculture` actions.
- Both hands were empty. Agriculture therefore exposed no card choice and changed no card zone;
  3,257 of 3,261 Agriculture actions were directly classified as no-op dogmas.
- Player 1's red stack was already splayed left. Of 3,254 `splay-left` choices, 3,253 changed no
  card and no splay geometry.
- The late loop did not oscillate hidden zones: the last 500 actions contained zero card-zone
  transitions. Scores remained tied at 32 points; Player 1 had two achievements and Player 2 had
  three.
- The strategic digest, which excludes monotonic IDs, repeated 9,861 times; one digest occurred
  1,446 times. This confirms a strategic policy cycle rather than merely advancing state IDs.
- `Machinery` remained the unique learned argmax. In representative final selections it scored
  about 0.376–0.378 versus the next action at about 0.361–0.363, a margin of about 0.0149. Its four
  determinization values were identical in those states, while the Draw route varied by sample.

No failure was converted to a draw.

## Historical 4/3/4 stop decision

The original approved plan required a stop rather than weakening search when the two-round horizon
exceeded the frozen cutoff gate. That historical stop condition was met. The owner then changed the
horizon to one completed turn, and the fresh rerun documented above also failed. Milestone 4
therefore remains stopped. This implementation does **not**:

- freeze the provisional descriptor as the production heuristic;
- run heuristic strength arenas;
- generate Milestone 4 bootstrap or learned replay;
- publish a fresh checkpoint or primary runnable learned policy; or
- make playing-strength claims.

The revised one-turn horizon remains the binding Milestone 4 direction, but it is not feasible under
the tested exhaustive implementation and existing gates. Production budget and determinization
count remain intentionally unfrozen pending an owner decision.
