# Innovation AI

A complete deterministic Python implementation of the two-player Innovation base game and a
research platform for building a strong self-play-trained opponent, inspired by the end-to-end aim
of Keldon Jones's 2009 Race for the Galaxy AI project.

## Current milestone

**Milestones 1–3 are complete. Milestone 4 is advancing through deterministic-rollout
feasibility gates.** The repository now includes the complete deterministic engine, the first
learned-value pipeline and measured pilot, and the Milestone 4 player-safe search/diagnostic
foundation.

Milestone 4 implemented:

- `public-covered-v1` as the default for new games, while preserving legacy replay/checkpoint
  compatibility;
- player-safe determinizations for paid turns, simultaneous starting melds, and pending effects;
- preserved historical complete-turn sampled minimax plus a distinct owner-approved deterministic
  turn-rollout selector using the same player-safe determinizations and leaf evaluator;
- tagged minimax/rollout contracts, schema-v3 learned-policy identities, compact-replay schema v2,
  and scheduler/self-play/arena routing;
- complete trusted-private compressed traces plus public redacted summaries; and
- exact reproduction of the historical seed-50000 action-ceiling cycle.

The September 3, 2026 exhaustive-search experiments remain historical failures. On September 5,
2026 the owner approved the deterministic turn-rollout recovery. Its expanded 12-root core
benchmark completed all 322 routes with no failures or ceiling hits. The September 6 full
reference protocol then completed 251/251 seeded/all-card/high-choice soak roots and 1,860/1,860
routes with no failures or ceiling hits and a maximum route length of 18. End-to-end rollout
baseline generation and non-strength rollout-vs-random arena smokes sustained 12.328 and 17.925
committed actions/second, respectively. Every R4 feasibility gate passed. On September 6, 2026 the exact tested descriptor was
production-frozen as `sha256:c4704991445a58acc3747dd4117b65b2c1663fbaecc3db5fd08501342bd88ce1`
(two determinizations; strict 1,024-transition route ceiling), with rollout agent
`sampled-turn-rollout-heuristic-v1-c0c5487a9c2c5444`. Fixed R5 heuristic validation is now in
progress; fresh training remains gated on its completion.

- Project contract: `PROJECT_GOAL.md`
- Supplied rules/data: `game-rules-plaintext/`
- Implementation sequence: `docs/MILESTONE_1_PLAN.md`
- First ML milestone plan: `docs/MILESTONE_2_PLAN.md`
- Milestone 2 implementation/report: `docs/MILESTONE_2_REPORT.md`
- Milestone 3 training prototype/report: `docs/MILESTONE_3_TRAINING_PROTOTYPE.md`,
  `docs/MILESTONE_3_REPORT.md`
- Milestone 4 plan, feasibility evidence, and approved rollout recovery:
  `docs/MILESTONE_4_PLAN.md`, `docs/MILESTONE_4_FEASIBILITY_ADDENDUM.md`,
  `docs/MILESTONE_4_DETERMINISTIC_ROLLOUT_PLAN.md`
- Afterstate feasibility decision: `docs/MILESTONE_2_AFTERSTATE_SPIKE.md`
- Frozen encoder layouts: `docs/encoder_v1_manifest.json`,
  `docs/encoder_v1_public_covered_manifest.json`
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

# Full R4 reference protocol: 12-root benchmark, 251-root route soak, and both throughput smokes.
# The output directory must be fresh. Measured wall time on the two-vCPU reference host is about
# 71 seconds (allow 1–2 minutes); peak process RSS is about 234 MiB.
time uv run innovation-ai rollout-feasibility \
  --output artifacts/runs/milestone-4/deterministic-turn-rollout-r4-full \
  --measure-remaining-gates

# Historical learned-value workflow. Do not start a new Milestone 4 iteration until the
# passing R4 artifacts are reviewed and a production rollout descriptor is frozen.
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
