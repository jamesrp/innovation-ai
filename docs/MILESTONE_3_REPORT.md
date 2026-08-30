# Milestone 3 implementation and pilot report

**Status:** pilot completed August 29, 2026; promotion-scale work is blocked pending a
policy-level cycle decision.

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
small diagnostic set, but the interval includes 0.5 and this is not a promotion claim. The run took
11,441 seconds for only 10,945 actions (**0.96 actions/s**), making four-determinization arena
execution a second scaling problem independent of the policy cycle. The failed combined run and the
successful random-only run together count as two arena attempts; no controlled training-variable
comparison was started after the stop condition.

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
