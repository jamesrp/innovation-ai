# Milestone 2 implementation report

**Completed:** August 29, 2026

## Release scope

Milestone 2 implements the complete first learned-value pipeline described in
`docs/MILESTONE_2_PLAN.md`:

1. Stable baseline descriptors, SHA-256 seed derivation, bounded actors, compact action events,
   JSONL metrics, and repeatable performance scenarios.
2. Tensor-free public value positions and the frozen encoder-v1 layout:
   - input dimension: **4690**;
   - layout fingerprint:
     `sha256:b472a8911f444bcf7920ff89fab2ff55aa23f054747242b538c32a7851c5b2b5`.
3. The exact CPU PyTorch `4690 -> 128 tanh -> 1 sigmoid` value model, immutable checkpoint
   directories, digest verification, and complete content-addressed policy descriptors.
4. Deterministic gzip compact episodes, verified semantic replay extraction, grouped
   train/validation splits, deterministic NPZ caches, BCE-with-logits terminal training, Brier
   early stopping, calibration, and checkpoint publication.
5. Information-set specifications and hidden-state sampling that accept no live state at the
   sampler boundary, trusted sampled candidate expansion, exact terminal utility, batched routing,
   temperature selection, and strict/heuristic failure modes with no true-state fallback.
6. Predeclared, resumable compact self-play generations with frozen per-seat policies and atomic
   shard sealing.
7. Exact paired, seat-swapped arenas; policy/checkpoint pools; paired percentile-bootstrap
   intervals; immutable champion references; and the fixed 200-pair/lower-bound promotion rule.
8. Warmed, repeated CPU profiling with explicit sweeps, integrity settings, structured JSON, and
   Markdown bottleneck reports.

The immediate afterstate-boundary decision and acceptance evidence are recorded separately in
`docs/MILESTONE_2_AFTERSTATE_SPIKE.md`.

## Reproducible smoke evidence

The following small release smoke was run on the CPU exe.dev development VM on August 29, 2026:

```bash
uv run innovation-ai iterate \
  --run-dir artifacts/runs/m2-smoke \
  --bootstrap-games 4 --learned-games 2 \
  --games-in-flight 2 --shard-size 2 \
  --epochs 2 --patience 1 --batch-size 128 \
  --validation-fraction 0.5 --max-actions 2000 --seed 31415
```

It completed bootstrap replay generation, dataset materialization, first-checkpoint training,
frozen learned self-play, candidate dataset materialization, and candidate training without manual
artifact edits:

- bootstrap examples: **1017** from 4 episodes;
- learned-generation examples: **409** from 2 episodes;
- candidate checkpoint:
  `sha256:7a117edc6f28725e57cf7a18029406180ec127e658e6933406b88b9a49be5bae`.

These IDs describe a smoke artifact in ignored `artifacts/`; they are evidence of pipeline
completion, not permanent release fixtures.

A one-seed-pair candidate-versus-random arena also completed both seat-swapped games and produced
a paired report. Its 0-2 result is deliberately **not** a strength or promotion claim; the release
promotion contract requires 200 predeclared pairs against the incumbent. A learned-versus-learned
arena using these deliberately tiny two-epoch smoke checkpoints reached the explicit action ceiling,
which is reported rather than hidden or converted into a result.

## Baseline and profile observations

A small two-game-per-scenario Stage-0 benchmark measured roughly 410-525 actions/s across full and
cheap validation for random/heuristic matchups. The exact throughput is machine- and game-length-
dependent; the benchmark command records semantic digests separately from live timing.

The final minimal all-category profile used one warmup and one timed sample at batch/thread/actor/
determinization size 1. Representative measurements were:

| category | throughput |
|---|---:|
| engine-only | 1313 actions/s |
| encoding | 1077 positions/s |
| inference | 962 positions/s |
| training | 106 examples/s |
| determinization | 689 samples/s |
| afterstate expansion | 667 candidates/s |
| replay extraction | 449 examples/s |
| self-play | 198 actions/s |
| paired arena | 408 actions/s |

In that minimal profile, self-play was 45.1% of measured wall time. The profiler therefore
recommends **bounded actor processes** as the next scale step before a dedicated inference process
or GPU batching. Larger sweep reports should be regenerated on the target machine rather than
assuming these smoke values are universal.

## Commands

```bash
make install-ai
make check

uv run innovation-ai self-play --run-dir artifacts/runs/bootstrap-001 \
  --games 1000 --player-1 heuristic --player-2 random --seed 1000
uv run innovation-ai dataset build --replays artifacts/runs/bootstrap-001 \
  --output artifacts/runs/bootstrap-001/dataset
uv run innovation-ai train-value \
  --dataset artifacts/runs/bootstrap-001/dataset/manifest.json \
  --output artifacts/runs/bootstrap-001/checkpoints
uv run innovation-ai iterate --config configs/cpu-value-baseline.toml
uv run innovation-ai profile --full --config configs/cpu-value-baseline.toml \
  --output artifacts/profiles/milestone-2
```

Arena execution additionally requires a resolved learned policy descriptor and its checkpoint
root; `innovation-ai arena --help` lists the paired-seed and promotion options.

## Known limitations

- This milestone establishes a reproducible pipeline, not statistically demonstrated strength.
- Learned control is limited to paid `TURN_ACTION` decisions; setup and nested effect choices use
  the simple heuristic fallback.
- The information-set sampler supports stable `PLAY` paid-action boundaries only. Pending effects,
  transient reveals, setup, and terminal states are rejected rather than approximated.
- The encoder is memoryless and does not preserve deductions from earlier public play.
- The initial implementation is single-process CPU. The batch/serialization boundaries are ready
  for process or machine separation, but no distributed scheduler, CUDA, mixed precision, DDP,
  MCTS, recurrent state, expansions, or multiplayer support is included.
- Deterministic heuristic self-play and weak temperature-zero learned checkpoints can cycle for
  some setups; actor and arena action ceilings are explicit and fail loudly without sealing partial
  episodes or inventing arena outcomes.
