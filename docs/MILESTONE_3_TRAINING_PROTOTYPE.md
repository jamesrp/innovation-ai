# Milestone 3 Plan — Training Prototype

**Planning status (September 2, 2026):** complete. Milestones 1 and 2 supplied the frozen
starting point; this milestone executed the first serious measured training baseline and its
controlled selector follow-up. Historical commands and gates remain below for reproducibility.

## 1. Milestone outcome

Milestone 3 should answer a practical question:

> Can the existing value-learning pipeline generate stable training data, improve held-out value
> metrics, complete deterministic evaluation games, and produce evidence-guided follow-up
> experiments at useful CPU scale?

The target is a **training prototype**, not a playing-strength guarantee. Completion requires:

- one reproducible medium-scale bootstrap/learned iteration;
- a compact experiment summary containing data, optimization, throughput, and arena metrics;
- strict zero-failure replay and information-set sampling on generated training episodes;
- deterministic 25-pair preflight arenas against random and heuristic baselines;
- diagnosis of any action-ceiling/cycle failures without inventing game outcomes;
- at least one controlled follow-up experiment changing one variable at a time; and
- a clear go/no-go recommendation for a larger run and eventual 200-pair promotion arena.

A candidate beating the heuristic is welcome, but not required. Scaling an unhealthy run is a
failure even if a tiny arena score happens to look favorable.

## 2. Frozen starting point

Do not change these in the first pilot:

- encoder: `flat-observation-v1`, dimension **4690**;
- encoder layout fingerprint:
  `sha256:b472a8911f444bcf7920ff89fab2ff55aa23f054747242b538c32a7851c5b2b5`;
- model: `4690 -> 128 tanh -> 1 sigmoid`;
- target: terminal viewer utility `1 / 0.5 / 0`;
- afterstate semantics: one submitted paid action, immediate returned boundary;
- learned control: `TURN_ACTION` only, with heuristic setup/effect fallback;
- self-play determinizations: 1;
- arena determinizations: 4;
- learned self-play temperature: 0.15;
- arena temperature: 0;
- default information policy and information-set sampler versions.

The first experiments should determine whether this fixed system learns anything useful. Do not
confound the result with simultaneous encoder, architecture, sampler, or rules changes.

## 3. Handoff validation already completed

The repository was clean and synchronized with `origin/main` before this plan was written.
Validation completed on August 29, 2026:

- `make check`: **776 passed, 1 skipped**, branch coverage **85.12%**;
- opt-in engine protocol fuzz: **100 seeds**, all passed;
- additional determinization soak:
  - **50 complete games**;
  - **11,428 semantic actions**;
  - **7,092 sampled paid-turn boundaries**;
  - **0 sampler or sample-verification failures**.

No more engine fuzzing is required before the first training pilot. Repeat the 100-seed engine fuzz
and determinization soak if engine rules, observations, public-boundary construction, or sampling
behavior change. The determinization soak was an ad hoc release validation, not yet a committed
CLI command; making it an opt-in reproducible test is a useful small follow-up but not a blocker to
training.

## 4. Work package 0 — Experiment reporting

Before or alongside the first run, add a small report builder if the raw artifacts are awkward to
compare. It must consume existing immutable artifacts rather than creating a second source of
truth.

At minimum, record:

- resolved configuration and digest;
- episode, transition, and encoded-example counts by generation and split;
- target counts/mean for train and validation;
- constant-mean validation Brier score as a trivial reference;
- best epoch, epochs completed, train/validation BCE and Brier;
- train/validation prediction means and calibration bins;
- examples/s, actions/s, sampler failures, and action-ceiling failures;
- checkpoint and policy IDs, parent checkpoint IDs, and dataset IDs;
- arena completion count, W/D/L, paired utility/interval, seat split, reasons, and game lengths.

Prefer a canonical JSON summary plus a concise Markdown table beneath the run directory. Do not
copy model weights or rewrite existing manifests.

## 5. Work package 1 — Medium pilot

Run this first:

```bash
uv run innovation-ai iterate \
  --run-dir artifacts/runs/pilot-001 \
  --bootstrap-games 512 \
  --learned-games 256 \
  --games-in-flight 16 \
  --shard-size 64 \
  --epochs 30 \
  --patience 5 \
  --batch-size 1024 \
  --validation-fraction 0.2 \
  --temperature 0.15 \
  --determinizations 1 \
  --max-actions 10000 \
  --seed 41000
```

`iterate` now alternates heuristic/random physical seats in bootstrap generation. Learned
self-play uses the same frozen learned policy in both seats. Keep strict sampler behavior: any
sampler failure must stop the run.

Expected rough scale, based on the release smoke, is tens of thousands of examples rather than a
tiny fixture. Do not treat the estimate as an acceptance criterion; the actual episode-length and
example distributions belong in the report.

### Pilot health gates

All of these must hold before scaling:

1. Every planned compact replay shard seals and verifies.
2. Sampler failures and replay divergences are zero.
3. Train and validation splits are both nonempty and grouped by full setup provenance.
4. Losses and predictions remain finite.
5. Held-out Brier improves from the first epoch and is competitive with the constant-mean
   reference.
6. The train/validation gap is explainable; obvious memorization or collapsed predictions stop the
   experiment.
7. Interruption/resume, if exercised, creates no missing or duplicate episode IDs.
8. Throughput and peak RSS remain consistent with the CPU box rather than degrading unexpectedly.

## 6. Work package 2 — Arena preflight

Read the candidate policy ID from:

```text
artifacts/runs/pilot-001/iteration-summary.json
```

Then run a non-promotion preflight:

```bash
uv run innovation-ai arena \
  --candidate-policy artifacts/runs/pilot-001/policies/<candidate-policy-id>.json \
  --checkpoint-root artifacts/runs/pilot-001/checkpoints \
  --opponents random,heuristic \
  --seed-start 50000 \
  --seed-pairs 25 \
  --determinizations 4 \
  --max-actions 10000 \
  --output artifacts/runs/pilot-001/arena-preflight
```

This is a completion and diagnostic gate, not a promotion test. Inspect opponents separately and
report seat bias rather than relying only on aggregate utility.

### Arena gates

- All 50 games per opponent suite must complete below the declared action ceiling.
- Re-running the manifest at temperature zero must reproduce the same semantic results.
- Candidate actions must remain legal and sampler failures must remain zero.
- Record paired utility and interval, W/D/L, seat split, terminal reasons, and game lengths.
- Do not extend the seed set after seeing results.

A one-pair release smoke completed against random. Deliberately tiny two-epoch learned checkpoints
cycled in learned-versus-learned temperature-zero play and reached the action ceiling. Therefore,
**arena completion rate is the first evaluation metric** for this milestone.

An action-ceiling failure is not a draw and must not be inserted into `ArenaResult`. Capture the
exact setup seed, game ID, policy IDs, action count, and enough state/action hashes to reproduce the
cycle. If repetition handling is later added, it is policy behavior with a new policy/version
identity—not an Innovation rules change.

## 7. Work package 3 — Controlled experiments

If the pilot is healthy, run one-variable comparisons. Keep setup and arena seed sets fixed within
each comparison.

Recommended order:

1. **Data scale:** 512/256 games versus 1000/500 or the full configured 1000/2000 run.
2. **Exploration temperature:** 0.10, 0.15, and 0.25, with all other settings fixed.
3. **Optimization duration:** rely on held-out Brier early stopping; compare patience/epoch ceilings
   only if the pilot stops for a clear reason.
4. **Policy pool:** once at least two learned generations exist, use the implemented fixed mix:
   50% latest, 25% previous, 25% uniform older/baseline pool.
5. **Actor scale:** profile 8, 16, and 32 games in flight before implementing parallel processes.

Keep self-play determinizations at 1 for these CPU experiments. Arena determinizations remain 4.
Do not compare architecture changes until a fixed encoder/model baseline has at least one complete,
repeatable experiment report.

For every experiment, change one principal variable, use a new run directory, preserve immutable
artifacts, and write the hypothesis before acting.

## 8. Work package 4 — Scale and promotion decision

Only after a candidate completes the 25-pair preflight:

1. run a larger fixed evaluation against random and heuristic baselines;
2. establish the first compatible learned checkpoint as bootstrap champion without a statistical
   claim if no champion exists;
3. predeclare exactly 200 seed pairs against the incumbent;
4. run the candidate/incumbent arena once without adaptive extension; and
5. promote only when the paired 95% lower confidence bound is above `0.5`.

An inconclusive or weaker candidate remains an immutable pool member. A cycle/action-ceiling
failure blocks the promotion arena; it is not converted to a draw or silently excluded.

## 9. Stop conditions

Stop and diagnose rather than spending more CPU if any of these occur:

- sampler failure or observation/legal-action mismatch;
- compact replay incompatibility, illegal action, final-hash mismatch, or duplicate episode ID;
- nonfinite loss, prediction, gradient-related failure, or corrupted checkpoint digest;
- empty train/validation partition;
- validation Brier fails to improve over the trivial reference;
- widespread temperature-zero action-ceiling failures;
- an unexplained major seat asymmetry; or
- throughput/RSS regression large enough to invalidate the planned run size.

Do not weaken strict validation, sampler safety, replay verification, or checkpoint compatibility
to get a run to finish.

## 10. Non-goals

Unless pilot evidence forces a separate project decision, Milestone 3 does not include:

- encoder-v2 features or information memory;
- a policy head, recurrence, MCTS, belief learning, or multi-ply search;
- treating authoritative hidden state as model input;
- changing game rules or declaring action-ceiling games draws;
- GPU/distributed training, DDP, mixed precision, or remote inference;
- expansions or multiplayer Innovation; or
- tuning many hyperparameters simultaneously.

## 11. Suggested first-agent sequence

1. Read `PROJECT_GOAL.md`, `AGENTS.md`, `docs/MILESTONE_2_PLAN.md`,
   `docs/MILESTONE_2_REPORT.md`, and this document.
2. Run `make install-ai`, `make check`, and `uv run innovation-ai doctor`.
3. Optionally make the determinization soak a committed opt-in fuzz test.
4. Add/verify the compact experiment summary described in Work Package 0.
5. Run `pilot-001` exactly as specified.
6. Review replay, split, optimizer, calibration, throughput, and sampler metrics before arena play.
7. Run the 25-pair random/heuristic preflight.
8. Diagnose completion/cycling before any larger arena.
9. Propose one controlled follow-up experiment and record its hypothesis.
10. Commit and push code/docs only; generated datasets, checkpoints, and reports remain under the
    ignored `artifacts/` tree.

## 12. Milestone completion checklist

- [x] Medium pilot iteration completed and resumability verified with the release smoke workflow.
- [x] Canonical JSON and Markdown experiment summary produced.
- [x] Zero sampler/replay/integrity failures in generated training data.
- [x] Held-out metrics compared with a trivial reference.
- [x] 25-pair random and heuristic preflights completed without ceiling failures under the
      versioned `recent-paid-action-penalty-v1` policy selector on August 30, 2026.
- [x] Seat/reason/length/calibration diagnostics reviewed.
- [x] One controlled follow-up experiment completed: versioned bounded repetition awareness.
- [x] Scale/promotion recommendation written with evidence.
- [x] `make check` passes (785 passed, 1 skipped on September 2, 2026).
- [x] Optional fuzz/soak rerun not required for the selector experiment because sampler and engine
      contracts did not change.
- [x] Milestone 3 implementation commits pushed to `origin` before Milestone 4 planning began.
