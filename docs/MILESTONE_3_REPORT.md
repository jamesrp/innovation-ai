# Milestone 3 implementation and pilot report

**Status:** pilot and repetition-aware policy preflight completed August 30, 2026; the original
policy cycle is resolved by a versioned selector experiment, while promotion-scale work remains
pending.

## Implementation delivered

Milestone 3 adds:

- canonical `iteration-summary.json` and `iteration-summary.md` reports derived from immutable
  replay, dataset, checkpoint, policy, and optional arena artifacts;
- resolved iteration configuration and digest recording;
- generation, split, target-distribution, constant-reference, optimization, calibration,
  throughput, RSS, policy-lineage, and failure-counter reporting;
- resumable iteration stage state and immutable configuration checks;
- deterministic checkpoint identities that exclude wall-clock throughput while retaining timing
  in separate run telemetry;
- strict replay verification while building reports;
- self-play and arena action-ceiling diagnostics with setup, policy, action-prefix, and state-hash
  evidence; and
- immutable arena result/report publication and deterministic same-command reruns.

The implementation was validated before the pilot with `make check`: **780 passed, 1 skipped**,
with **85.07%** branch coverage.

## Run 1: `pilot-001`

The prescribed medium pilot completed successfully:

```text
bootstrap: 512 episodes, 130,174 transitions, 128,638 examples
learned:   256 episodes,  53,748 transitions,  52,980 examples
sampler/replay/integrity/action-ceiling failures: 0/0/0/0
```

Generation throughput was 120.7 actions/s for heuristic/random bootstrap and 26.4 actions/s for
learned self-play. Optimizer throughput was about 32k examples/s. Peak process RSS reached about
5.73 GiB during the learned stage on the 7.7 GiB host; this completed without swapping or OOM, but
it leaves too little memory margin to justify a larger data run without changing materialization
memory behavior.

### Held-out results

| generation | train examples | validation examples | best epoch | train Brier | validation Brier | constant validation Brier |
|---|---:|---:|---:|---:|---:|---:|
| bootstrap | 109,337 | 19,301 | 1 | 0.1861 | 0.1871 | 0.2453 |
| learned | 41,471 | 11,509 | 4 | 0.1664 | 0.2257 | 0.2500 |

Both selected checkpoints beat the train-mean constant reference. The learned candidate improved
validation Brier from 0.2461 at epoch 1 to 0.2257 at epoch 4, then overfit through epoch 9 before
patience stopped training and restored epoch 4. The bootstrap fit selected epoch 1 and worsened on
all later epochs despite continued train improvement. Thus the candidate optimization is useful,
but the pilot does not satisfy the strongest reading of the gate that every training phase improve
beyond its first epoch.

Candidate identifiers:

- checkpoint: `sha256:652e12baed1bde3cac92aa474b53647891e7826fdfb19a4b95b73dd39949a04c`
- self-play policy: `sha256:6ffa5335cd971ecfe422ac82049040dfdce848c4cdbf9bc3235d128d04884cf9`

## Arena preflight outcome

The fixed 25-pair combined random/heuristic preflight was attempted on August 29–30, 2026 without
extending its seed set. It stopped without producing an `ArenaResult` when the candidate-as-player-1
game against the heuristic at setup seed **50000** reached exactly **10,000 actions**.

The retained diagnostic shows a stable strategic loop dominated by repeated `Machinery` dogma,
`splay-left`, and `Agriculture` dogma actions. State hashes continue changing because turn/decision
counters advance, so this is a semantic policy cycle rather than a byte-identical engine-state
repetition. The failure was not converted to a draw and no incomplete arena result was published.

A separate 25-pair random-only preflight completed all **50 games** on August 30, 2026:

- W/D/L: **33/0/17**;
- paired mean utility: **0.660**;
- paired 95% interval: **[0.500, 0.820]**;
- candidate as player 1: **17/0/8**;
- candidate as player 2: **16/0/9**;
- game length mean/min/max: **218.9/89/376** actions;
- terminal reasons: 37 achievement, 8 draw-beyond-age-10, 5 card-effect; and
- sampler/action-ceiling failures: **0/0**.

The near-balanced seat split is reassuring, and the candidate is competitive with random on this
small diagnostic set, but the interval includes 0.5 and this is not a promotion claim. The original
run took 11,441 seconds for only 10,945 actions (**0.96 actions/s**), making four-determinization
arena execution an apparent second scaling problem independent of the policy cycle. The failed
combined run and the successful random-only run together count as two arena attempts; no controlled
training-variable comparison was started after the stop condition.

## Arena throughput profiling follow-up

**Profiled August 30, 2026.** The original random preflight's 0.96 actions/s was not reproduced by
smaller fixed-seed reruns on the same checkout. Before optimization, the first four seed pairs
completed at 44.15 actions/s and the first eight at 26.84 actions/s. The decline with the larger
all-in-flight arena is real, but the smaller measurements do not explain the earlier
11,441-second wall time. It should not be treated as the current steady-state cost without a fresh
full rerun.

A four-pair `cProfile` run attributed the largest cumulative costs to sampled candidate expansion
through the real effect engine (62.1 seconds), flat encoding (22.5 seconds), and information-set
sampling (15.3 seconds). These totals overlap, but they show that PyTorch's tiny value network is
not the primary bottleneck. Four determinizations generated 11,840 model positions in that run,
and many were equal public afterstates repeated across hidden-state samples.

Two semantics-preserving changes were made:

- `CpuBatchValueEvaluator` now encodes and scores equal `ValuePosition` objects once per evaluator
  call, then restores values to the original candidate-route order. The profiled four-pair run
  encoded 4,779 unique positions instead of 11,840, a 59.6% reduction.
- `InformationSetSampler.sample_many` now validates the shared immutable specification and computes
  its digest once, while retaining independent per-sample reconstruction and verification. The
  trusted candidate expander still independently verifies every sampled state before applying any
  hypothetical action.

Fixed-seed before/after results were identical and throughput improved as follows:

| fixed random arena | actions | before | after | improvement | after peak RSS |
|---|---:|---:|---:|---:|---:|
| 4 seed pairs / 8 games | 1,802 | 44.15 actions/s | 52.16 actions/s | 18.1% | 242 MiB |
| 8 seed pairs / 16 games | 3,272 | 26.84 actions/s | 30.65 actions/s | 14.2% | 246 MiB |

The four-pair profiled wall time fell from 98.6 to 85.1 seconds (13.7%), despite profiler overhead.
The remaining dominant cost is exact one-ply expansion, especially dogma actions that execute many
effect steps before reaching the immediate afterstate boundary. Further work should measure
candidate counts and effect-step counts per game/decision, then target repeated observation and
engine-boundary construction. It should not weaken sample verification or change the frozen
encoder/model/policy semantics merely to improve the benchmark.

## Decision and next experiment

**No-go for a larger data run or promotion arena.** The current blockers are deterministic policy
behavior and arena throughput, not data generation or numerical training stability. Scaling
512/256 to 1000/500 would consume substantially more CPU and memory without addressing either
problem.

Before another training-scale comparison, make a project decision on one controlled policy-layer
change with a new policy/version identity. Recommended first experiment: add bounded repetition
awareness to the temperature-zero action selector, penalizing recently repeated paid-action
patterns while leaving engine rules, encoder, model, sampler, and checkpoint unchanged. Re-run the
same seed-50000 heuristic reproduction first, then the unchanged 25-pair preflight. An alternative
is a versioned deterministic tie-break/diversification rule, but merely raising the action ceiling
or declaring the game a draw is not acceptable. In parallel, profile why the arena achieves only
about one action/s at four determinizations before scheduling another 100-game preflight.

## Repetition-aware selector experiment

**Completed August 30, 2026.** The controlled policy-layer change is
`recent-paid-action-penalty-v1`. It leaves the checkpoint, encoder, value aggregation,
determinizations, engine, and legal-action set unchanged. For each learned physical seat, the
selector retains only the last four committed paid actions. It compares semantic action patterns
without decision IDs and subtracts **0.05 per matching recent action** from the value used for
selection, for a maximum penalty of **0.20**. The original model mean remains the reported
selection value. The selector version is already part of the content-derived policy descriptor,
so this produces a new policy identity without changing checkpoint identity.

History advances only after the runner successfully commits a schedule. Repeated scheduling before
submission is therefore stable, and committed-decision recording is idempotent. Setup and nested
effect choices remain heuristic fallbacks and do not enter the paid-action history. The arena CLI
can derive the versioned policy with:

```text
--selector-version recent-paid-action-penalty-v1
```

The exact failed reproduction—heuristic opponent, setup seed **50000**, four determinizations,
temperature zero, and the original candidate checkpoint—completed both seat-swapped games in
**59** and **135** actions. The derived policy ID is
`sha256:e1dcca55aaeb4fec502019c7b590ef49883ff4f375172414813bf930758bc1e9`.

The unchanged 25-pair random/heuristic preflight then completed all **100 games** with zero sampler
or action-ceiling failures:

| opponent | W/D/L | utility | paired 95% interval | P1 W/D/L | P2 W/D/L | length mean/min/max |
|---|---:|---:|---:|---:|---:|---:|
| random | 46/0/4 | 0.920 | [0.840, 0.980] | 23/0/2 | 23/0/2 | 185.9/120/353 |
| heuristic | 43/0/7 | 0.860 | [0.760, 0.940] | 22/0/3 | 21/0/4 | 197.1/93/606 |
| combined | 89/0/11 | 0.890 | [0.830, 0.950] | 45/0/5 | 44/0/6 | 191.5/93/606 |

The run processed 19,147 actions in 478.8 seconds (**39.99 actions/s**) with peak RSS of about
274 MiB. This clears the Milestone 3 completion blocker and strongly supports retaining the
selector as a policy variant, but the preflight remains diagnostic rather than a promotion claim.
A promotion arena still requires a predeclared incumbent comparison under the normal 200-pair
criterion.

**Recommendation:** retain this selector variant as the evaluation candidate and, if no compatible
champion exists, use the documented non-statistical bootstrap-champion path. A predeclared
200-pair candidate/incumbent arena is now technically unblocked. Continue the no-go on a larger
training-data run until the pilot's materialization-memory margin is improved; more data is not
needed to validate this policy-layer result.
