# Innovation AI

A complete deterministic Python implementation of the two-player Innovation base game and a
research platform for building a strong self-play-trained opponent, inspired by the end-to-end aim
of Keldon Jones's 2009 Race for the Galaxy AI project.

## Current milestone

**Milestone 2 is complete:** the repository now includes the first end-to-end learned value
baseline on top of the complete Milestone 1 engine.

Implemented components include:

- audited random and heuristic baselines with bounded actor pools and structured metrics;
- frozen encoder-v1 public value positions (`D = 4690`) and a named layout manifest;
- the exact PyTorch `D -> 128 tanh -> 1 sigmoid` value network;
- deterministic compact replay shards, verified extraction, NumPy datasets, and terminal training;
- immutable content-addressed checkpoints and complete policy descriptors;
- current-information determinizations and information-safe batched one-ply selection;
- resumable frozen-checkpoint self-play generations;
- paired seat-swapped arenas, policy/checkpoint pools, bootstrap confidence intervals, and champion
  references; and
- CPU profiling/reporting across engine, encoding, inference, training, determinization,
  afterstates, replay extraction, self-play, and arena execution.

Milestone 2 is a pipeline baseline, not a playing-strength claim. The next scale step identified on
the CPU development box is bounded parallel actor processes; recurrent knowledge, belief models,
MCTS, GPU/distributed training, expansions, and multiplayer remain future work.

- Project contract: `PROJECT_GOAL.md`
- Supplied rules/data: `game-rules-plaintext/`
- Implementation sequence: `docs/MILESTONE_1_PLAN.md`
- First ML milestone plan: `docs/MILESTONE_2_PLAN.md`
- Milestone 2 implementation/report: `docs/MILESTONE_2_REPORT.md`
- Afterstate feasibility decision: `docs/MILESTONE_2_AFTERSTATE_SPIKE.md`
- Frozen encoder layout: `docs/encoder_v1_manifest.json`
- WP2 state/geometry contract: `docs/WP2_STATE_CONTRACT.md`
- WP3 protocol contract: `docs/WP3_PROTOCOL_CONTRACT.md`
- WP5 Freeze-B/effect contract: `docs/WP5_FREEZE_B_CONTRACT.md`
- WP6 achievement/terminal contract: `docs/WP6_ACHIEVEMENT_CONTRACT.md`
- WP9 serialization/replay contract: `docs/WP9_REPLAY_CONTRACT.md`
- Rules interpretations: `docs/RULES_DECISIONS.md`
- Future roadmap: `docs/ROADMAP.md`
- Agent conventions: `AGENTS.md`

## Development environment

```bash
make install       # Core package and development tools
make install-ai    # Also install NumPy and CPU-only PyTorch for ML work
make check         # Lint, type-check, and full default test/coverage gate
INNOVATION_LARGE_FUZZ_SEEDS=100 uv run pytest -q -m fuzz tests/innovation/test_fuzz.py
make run           # Verify the environment
make web           # Serve the hot-seat browser QA table on port 8000

# First learned-value workflow
uv run innovation-ai self-play --run-dir artifacts/runs/bootstrap-001 \
  --games 1000 --player-1 heuristic --player-2 random --seed 1000
uv run innovation-ai dataset build --replays artifacts/runs/bootstrap-001 \
  --output artifacts/runs/bootstrap-001/dataset
uv run innovation-ai train-value \
  --dataset artifacts/runs/bootstrap-001/dataset/manifest.json \
  --output artifacts/runs/bootstrap-001/checkpoints
uv run innovation-ai iterate --config configs/cpu-value-baseline.toml
```

## Browser QA table

The Milestone 1 engine includes a deliberately thin, non-production hot-seat UI for manual rules
verification:

```bash
make web
# then open http://localhost:8000
```

Choose actions for both players from the same browser. The table renders only the current
chooser's player-safe observation, includes the printed card reference text, supports one-step
undo by deterministic replay, accepts a new setup seed, and downloads the current replayable game
log. Games live only in server memory and reset when the process restarts; this is a QA surface,
not a network multiplayer service.

Direct invocation and custom binding:

```bash
uv run innovation-ai web --host 0.0.0.0 --port 8000 --seed 0
```

Direct equivalents:

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy src tests
uv run innovation-ai doctor
uv run innovation-ai play --seed 0 --log artifacts/game.json
uv run innovation-ai replay artifacts/game.json
uv run innovation-ai inspect-encoding --seed 0 --steps 2
uv run innovation-ai profile --config configs/cpu-value-baseline.toml \
  --output artifacts/profiles/milestone-2
```
