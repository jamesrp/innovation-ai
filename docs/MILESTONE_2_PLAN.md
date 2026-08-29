# Milestone 2 Plan — First Learned Innovation AI

**Implementation status (August 29, 2026): complete.** Milestone 2 now includes the frozen public
value-position/encoder contracts, value model and immutable checkpoints, compact replay and
terminal training, information-set determinizations, batched afterstate selection, resumable
self-play, paired arenas/promotion artifacts, CLI workflows, and CPU profiling. See
`docs/MILESTONE_2_REPORT.md` for the release evidence and remaining limitations.

## 1. Milestone outcome

Milestone 2 delivers the first end-to-end learned Innovation opponent:

- verified seeded-random and deterministic simple-heuristic baselines;
- a versioned, flat, viewpoint-relative position encoder;
- a PyTorch `D -> 128 tanh -> 1 sigmoid` value network;
- terminal-outcome training from compact semantic-action replays;
- batched, information-safe one-ply afterstate selection with temperature exploration;
- iterative self-play in which actors use immutable frozen checkpoints;
- an arena with paired setup seeds, seat swaps, checkpoint pools, and confidence intervals;
- profiling that identifies the CPU bottleneck and the cleanest future parallel-actor/GPU path.

This is a pipeline milestone, not a playing-strength claim. Completion requires a reproducible
iteration that generates games, trains a checkpoint, evaluates it, and records all artifacts. A
checkpoint beating the heuristic baseline is desirable evidence, but not a release gate until the
pipeline has enough scale and tuning to make that expectation meaningful.

## 2. Repository facts and reuse points

The implementation should extend, not replace, these existing contracts:

- `innovation.actions.Decision` contains a chooser-safe `GameObservation`, deterministic legal
  `SemanticAction` values, source metadata, and `DecisionContext`.
- `innovation.protocol.current_decisions()` and `apply_action()` are the only rules-transition
  boundary required by normal play. `apply_action()` is pure.
- `innovation.observations.observe()` constructs detached observations under the versioned
  information policy. Opponent private cards, supply order, normal-achievement card identities,
  and private covered-board information are not exposed.
- `harness.InnovationEngineAdapter`, `PullGameRunner.pending()`, and
  `PullGameRunner.submit()` already separate games from external action selection and collect
  decisions across independent games.
- `agents.RandomAgent` and `agents.SimpleHeuristicAgent` already exist and are tested. Stage 0
  treats them as completed baseline implementations to audit and benchmark, not greenfield work.
- `innovation.logs.GameLog` is a full audit/replay format. It deliberately stores full decisions
  and per-transition hashes and is therefore not the compact training format.
- `harness.GameRecord` is lightweight but lacks a portable, versioned file schema and ML run
  provenance. It should inform, but not become, the training replay contract.
- NumPy and CPU-only PyTorch are already isolated in the optional `ai` dependency group. Core
  engine modules have no ML dependency.
- The current exe.dev box has 2 vCPUs, about 7.7 GiB RAM, no CUDA device, and CPU PyTorch. The
  first implementation must avoid assumptions that require multiprocessing or a GPU to function.

## 3. Scope and non-goals

### In scope

1. Two-player base-game value learning from final game outcomes.
2. Memoryless current-information observations under the existing default information policy.
3. A scalar value model, with exact terminal utility and no learned policy head.
4. Learned selection for paid `TURN_ACTION` decisions.
5. Existing heuristic handling for `STARTING_MELD` and `EFFECT_CHOICE` decisions in the first
   learned policy. Their actions remain in replays and their resulting positions may be training
   examples where safe.
6. Single-process CPU acting/training first, with batch and serialization boundaries suitable for
   later process or machine separation.
7. Immutable checkpoint generations and reproducible evaluation.

### Explicitly out of scope

- recurrent memory or definite-knowledge tracking;
- belief networks, learned hidden-state models, MCTS, or deep multi-ply search;
- policy-gradient, temporal-difference, bootstrapped, or auxiliary losses;
- expansion sets, multiplayer Innovation, or alternate editions;
- distributed scheduling, remote inference, CUDA kernels, mixed precision, or DDP;
- changing engine observations to contain tensors or model-specific feature arrays;
- using authoritative hidden state directly to choose an action.

## 4. Architecture boundaries and proposed modules

The rules package remains ML-free. New code should use these ownership boundaries.

```text
src/innovation_ai/
  innovation/                 # unchanged rules/state/action/observation/replay authority
  agents/
    random.py                 # existing baseline
    heuristic.py              # existing fallback/baseline
  harness/
    policy.py                 # torch-free batch-policy and value-position contracts
    afterstates.py            # trusted engine expansion; never exposed to the model
    actor_pool.py             # bounded refill/retirement over pull-runner semantics
    arena.py                  # paired matches, pools, statistics, reports
    metrics.py                # small metric/timer sink protocols
  training/
    encoding.py               # versioned flat encoder; no engine mutation
    model.py                  # PyTorch value network only
    inference.py              # batching, device placement, temperature selection
    determinizations.py       # current-information hidden-state sampler
    compact_replay.py         # compact episode schema and deterministic gzip shards
    dataset.py                # replay verification and encoded-example materialization
    checkpoint.py             # immutable checkpoint bundles and compatibility checks
    optimize.py               # terminal-outcome optimizer/evaluation loop
    self_play.py              # actor orchestration and frozen-generation loop
    profiling.py              # benchmark scenarios and timing aggregation
```

Tests should mirror these packages under `tests/harness/` and `tests/training/`. Importing
`innovation_ai.innovation` or the existing non-ML agents must continue to work without installing
the `ai` extra. Modules that import NumPy or PyTorch should not be re-exported from core package
initializers in a way that makes those imports eager.

### 4.1 Framework-free value-position contract

Add a small immutable contract in `harness.policy` that is richer than `GameObservation` but still
player-safe and tensor-free:

```python
@dataclass(frozen=True, slots=True)
class PublicTurnProgress:
    self_tucked: int
    self_scored: int
    opponent_tucked: int
    opponent_scored: int


@dataclass(frozen=True, slots=True)
class PublicBoundary:
    decision_kind: DecisionKind | None
    chooser_relation: PlayerRelation
    executor_relation: PlayerRelation
    dogma_activator_relation: PlayerRelation
    source: DecisionSource | None
    context: PublicDecisionContext | None
    turn_progress: PublicTurnProgress


@dataclass(frozen=True, slots=True)
class ValuePosition:
    viewer: PlayerId
    observation: GameObservation
    boundary: PublicBoundary
```

`PlayerRelation` is `SELF`, `OPPONENT`, or `NONE`; it prevents canonical seat identity from
becoming an accidental strategic feature. `PublicTurnProgress` exposes viewpoint-relative
scored/tucked-this-turn counts because they affect Monument eligibility; these counts are public
rules history even though the current `GameObservation` schema does not carry them.
`PublicDecisionContext` is a sanitized form of `DecisionContext`: source card/effect,
demand/shared/nested flags, icon and count data, selection bounds, and only selection identities
visible to `viewer`. Hidden selected identities become counts plus explicit unknown masks.

A `ValuePosition` may describe the current decision or a candidate post-action state. It never
contains `GameState`, setup piles, state hashes, opponent-private observations, transient legal
indices, or tensors. When an action's next decision belongs to the opponent, construct a fresh
`observe(afterstate, original_chooser)`; never reuse that opponent decision's embedded
observation.

### 4.2 Batch evaluator and routing contracts

Keep inference routing independent of PyTorch:

```python
class BatchValueEvaluator(Protocol):
    def evaluate(self, positions: Sequence[ValuePosition], /) -> tuple[float, ...]: ...


@dataclass(frozen=True, slots=True)
class CandidateRoute:
    game_id: str
    decision_id: int
    action: SemanticAction
    sample_index: int
    evaluator_key: str


@dataclass(frozen=True, slots=True)
class PolicySelection:
    policy_id: str
    game_id: str
    decision_id: int
    action: SemanticAction
    mean_value: float
    temperature: float
```

The trusted afterstate layer produces candidate routes and player-safe positions. The evaluator
receives only positions. The selector groups returned values by semantic action and produces
existing `Submission` objects. `evaluator_key` routes candidates to an immutable loaded evaluator;
checkpoint-pool games are grouped into one model call per evaluator rather than mixing weights in
one tensor batch. A frozen evaluator cache loads each policy assigned to a generation once. This
is the seam that can later become a process queue or remote GPU inference service without changing
the engine, encoder manifest, or actor protocol.

### 4.3 Metrics boundary

Use a no-op-by-default `MetricSink`/`TimerSink` protocol in orchestration code. Emit structured
names and numeric values; do not add profiling fields to `GameState`, `Decision`, or
`SemanticAction`. The initial sink writes JSON Lines. Future Prometheus or distributed tracing can
adapt the same events.

## 5. Key design decisions

### 5.1 The value target

The value is the probability-like terminal utility for the encoded viewer:

- viewer is a winner: `1.0`;
- opponent is the sole winner: `0.0`;
- no winner: `0.5`.

There is no discounting and no intermediate reward in this milestone. Card-effect wins, normal
achievement wins, draw-above-age-10 outcomes, and explicit draws use the same mapping from
`TerminalResult`.

### 5.2 Initial learned-control boundary

The model selects only `DecisionKind.TURN_ACTION`. Existing `SimpleHeuristicAgent` resolves
starting melds and nested effect choices. This keeps the first value policy focused on strategic
paid actions and avoids making the initial hidden-state sampler reproduce arbitrary paused effect
VM states. The replay format still records every semantic action, and the architecture does not
prevent a later encoder/policy version from learning setup and effect choices.

### 5.3 Direct true-state afterstate expansion is forbidden

Applying every candidate to the real authoritative state and evaluating the resulting observation
would be clairvoyant. A hypothetical Draw could expose the actual next card before the policy has
committed to Draw; automatic dogma work could similarly expose private cards or hidden supply
order. Masking the original state before model inference does not repair that leak, because the
candidate outcome itself was conditioned on hidden truth.

Milestone 2 therefore uses **current-information determinizations** at stable paid-action
boundaries. A trusted `InformationSetSpecBuilder` may inspect the live state only to extract an
audited public protocol skeleton (including turn progress and monotonic IDs); it emits an immutable
`InformationSetSpec` containing the chooser's observation, public boundary, current semantic legal
actions, catalog/version data, and explicit hidden-allocation constraints. The
`InformationSetSampler` accepts only that spec plus its own RNG—not the real `GameState`.

1. Build one or more sampled authoritative states from `InformationSetSpec`, catalog, and a
   policy-owned RNG.
2. The sampler must not condition on the real identities/order/count allocation of fields hidden
   from the chooser.
3. Use the same sampled state for all legal actions in one decision (common random numbers).
4. Apply each semantic candidate to each sample through the normal engine.
5. For a nonterminal result, build `ValuePosition` from the original chooser's post-action view.
   For a terminal result, use exact utility without model inference.
6. Average sample values per semantic action, then apply temperature exploration.
7. Only after selection does the runner submit the chosen action to the real game state.

The initial default is one determinization per self-play decision and four per arena decision,
subject to Stage 7 measurements. More samples reduce variance but multiply engine work. A sampler
failure must never fall back to evaluating the real hidden state. It either uses the configured
heuristic fallback and records the failure or fails loudly in strict/test mode; the acceptance
corpus must have zero failures.

The first sampler supports `PLAY` states at `TURN_ACTION` boundaries with no pending effect stack
or physical reveal. It reconstructs hidden assignments consistent with the current observation:
known cards are fixed; opponent hand and score age multisets are preserved; visible splayed icons
are constraints; supply counts and one normal achievement per age are preserved; public turn
counters and paid-action state are copied through the audited skeleton; opponent unsplayed covered
cards, removed cards, hidden achievement identities, private-zone identities, and supply order are
sampled rather than copied. It creates synthetic setup provenance and does not retain the real
setup piles. Generated states must pass engine invariants, reproduce the same chooser observation
and public turn progress, and expose the same current semantic legal actions.

The sampling law is a versioned part of policy identity. Its manifest fixes canonical constraint
ordering, hidden-slot allocation algorithm, rejection/backtracking order, retry limit, synthetic
setup convention, and domain-separated SHA-256 RNG derivation. A behavior change increments the
sampler/RNG version and cannot silently alter an existing arena policy.

"One ply" means one submitted semantic `TURN_ACTION` and the deterministic engine work performed
by that `apply_action()` call, stopping at its returned decision or terminal boundary. A Dogma
candidate may therefore end at an immediate effect-choice boundary. The model does not choose that
effect action, but it values the position under the configured heuristic continuation policy.
Stage 1 must spike this immediate-boundary semantics against the alternative of rolling the
fallback through the full Dogma before the encoder is frozen; immediate one-transition semantics
is the default unless the spike shows that public effect context cannot be represented safely.

This is an intentionally memoryless information set, matching the existing observation policy. It
will not preserve deductions a human could remember from earlier public play; that remains a later
knowledge/memory milestone.

### 5.4 Flat viewpoint-relative encoder

`FlatObservationEncoder` accepts `ValuePosition` only and returns one contiguous `float32` vector.
It must not accept `GameState`. The manifest fixes card order by canonical `CardId`, enum order,
feature offsets, normalization, input dimension `D`, schema versions, and a SHA-256 layout
fingerprint.

Version 1 feature families are:

- global phase, relative active player, paid actions remaining, clipped/normalized turn number;
- ten normalized supply counts;
- available normal- and special-achievement bitsets;
- currently revealed card and announced-color bitsets;
- two player blocks ordered `[self, opponent]`, each containing:
  - hand and score age histograms;
  - legally known hand and score card-identity bitsets;
  - claimed normal- and special-achievement bitsets;
  - five color-stack blocks in canonical color order, each containing top-card identity/empty,
    splay direction, covered count plus a known-count mask, legally known covered-card identities,
    and visible covered-icon counts;
- public boundary features: decision kind; relative chooser, executor, and dogma activator; source
  card/effect; demand/shared/nested flags; featured icon and frozen counts; selection bounds;
  visible selected-card identities plus unknown count; incremental-selection kind; viewpoint-
  relative tucked/scored-this-turn counters for both players;
- an explicit current-position/afterstate marker.

Unknown is never encoded as numeric zero without a companion mask. Raw `player-1`/`player-2`,
`decision_id`, `dogma_action_id`, game ID, setup seed, action-list position, state hash, and hidden
history are excluded. Redundant one-hot/multi-hot features are acceptable in this small first model;
the expected `D` is only a few thousand. Stage 1 generates and commits the exact manifest and
asserts its dimension rather than hand-maintaining offsets.

### 5.5 Value network and loss

The network is deliberately small and architecture-frozen for the milestone:

```text
float32[D] -> Linear(D, 128) -> tanh -> Linear(128, 1) -> sigmoid
```

No embeddings, recurrence, dropout, batch normalization, policy head, or action head. Initialize
linear weights with Xavier uniform and biases to zero. Expose `forward_logits()` for training with
`BCEWithLogitsLoss`; expose `forward()`/`predict()` as sigmoid probabilities. This is numerically
stable while preserving the requested network semantics.

Default optimizer configuration for the CPU baseline is AdamW, learning rate `1e-3`, weight decay
`1e-5`, batch size `1024`, deterministic seed, and early stopping on held-out episode Brier score.
These are resolved run configuration, not permanent architecture constants. Training and
validation split by full setup-provenance digest, never by individual position, so positions from
one game—or duplicate equivalent setups—cannot cross the split.

Report terminal-outcome BCE, Brier score, mean prediction, simple fixed-bin calibration, examples
per second, and held-out game count. Do not add scikit-learn solely for metrics.

### 5.6 Compact replay versus audit logs

Add a separate versioned compact episode format. Do not remove fields from `GameLog` or weaken its
hash-verifying purpose. Use deterministic gzip-compressed JSON Lines initially so the format needs
no new runtime dependency and remains inspectable.

Each episode contains:

- format/schema version and episode ID;
- engine/rules/information/action/decision/observation/terminal schema versions;
- card-data and effect-program fingerprints plus complete producer `PolicyDescriptor` IDs and
  agent-RNG versions; policy metadata is provenance, not an encoder compatibility requirement;
- explicit `SetupProvenance`, including shuffled piles, not seed alone;
- generation, seat-to-policy/checkpoint mapping, exploration and determinization configuration;
- ordered semantic action payloads only, plus transition count;
- final `TerminalResult` and final authoritative state hash;
- producer run ID and resolved configuration digest.

Full decisions, observations, logits, and per-transition hashes are omitted. Dataset building
replays the semantic actions once, validates legality/invariants and the final hash, reconstructs
player-safe post-action `ValuePosition` examples, and attaches the terminal label. The default
training extraction excludes starting-meld afterstates and terminal positions; it includes
nonterminal post-action positions from paid turns and effect choices, with the original action
chooser as viewer. Extraction policy is versioned in the dataset manifest.

Repeated training epochs should use materialized NumPy shards (`.npz` initially) containing
features, targets, episode IDs, and small categorical metadata. Encoded shards are disposable
caches: compact semantic replays remain the durable source and may be re-encoded by future
encoders. Dataset manifests—not compact replay compatibility checks—record the selected encoder
fingerprint, source shard hashes, extraction policy, counts, and split membership.

Shard production is deterministic and resumable: episode IDs and shard membership are assigned
before acting; episodes are serialized in canonical episode-ID order regardless of completion
order; gzip uses `mtime=0` and fixed headers; and writers seal by atomic temporary-file rename.
Dataset splits use the digest of full setup provenance, not seed alone, so duplicate/equivalent
setups cannot cross train and validation.

### 5.7 Temperature exploration

Candidate values are grouped by exact `SemanticAction`, averaged across determinizations, and
sampled with:

```text
P(action) proportional to exp((value - max_value) / temperature)
```

- `temperature = 0`: deterministic argmax with existing legal-action order as tie breaker;
- `temperature > 0`: categorical sample from a policy-owned RNG;
- terminal candidates use exact `0`, `0.5`, or `1` utility before sampling;
- values are clamped only for numeric safety, not renormalized across games.

Actor RNG streams are derived by a versioned, domain-separated SHA-256 construction from run seed,
generation, game ID, chooser, and decision ID, not from Python's process-randomized `hash()` or
batch order. Rebatching or changing the number of concurrent games must not change a game's action
stream when all other inputs are equal.

### 5.8 Policy identity and frozen checkpoints

A checkpoint identifies model weights, not a complete playing policy. Add an immutable
`PolicyDescriptor` whose content-derived `policy_id` includes:

- checkpoint ID and encoder fingerprint;
- afterstate-boundary semantics version;
- information-set sampler and RNG versions;
- determinization count;
- fallback agent/version for non-learned decisions;
- temperature/selector version and RNG derivation version;
- information policy and engine/card/effect compatibility fields.

Self-play pools, arena manifests, and results reference policy IDs. CLI checkpoint shorthand is
allowed only if it resolves and writes a complete policy descriptor before games begin.

A checkpoint is an immutable directory, never an overwritten filename:

```text
checkpoints/<checkpoint-id>/
  manifest.json
  model.pt
  optimizer.pt          # optional resumable-training state
  metrics.json
```

The manifest contains network architecture, `D`, encoder fingerprint, all engine schema and card
fingerprints, training dataset IDs, parent checkpoint(s), generation, optimizer config, PyTorch
version, creation command, and SHA-256 digests for files. `model.pt` and `optimizer.pt` contain
state dictionaries only. Verify their digests before loading and use
`torch.load(..., weights_only=True, map_location=...)` where supported; actors never load optimizer
state. Save model tensors on CPU; inference device is runtime configuration. A content-derived
checkpoint ID plus atomic temporary-directory rename prevents partial or mutable generations.

Actors resolve every policy assigned to the generation, load each distinct checkpoint once into a
frozen evaluator cache, and keep all models in evaluation mode for every game in that generation.
The trainer never mutates actor weights. This invariant is required even in the single-process
implementation because it is the contract future parallel actors will depend on.

## 6. Staged implementation plan

### Stage 0 — Baseline audit and performance fixture

**Dependencies:** Milestone 1. **Code risk:** low.

Deliverables:

- confirm `RandomAgent` and `SimpleHeuristicAgent` consume only `Decision` and immutable catalog
  data;
- define stable agent descriptors and seed derivation used in run manifests;
- add a bounded actor lifecycle/recording seam: completed games can be retired and replaced
  without retaining every final `GameState` or `RecordedAction`, while existing `PullGameRunner`
  behavior remains the compatibility default;
- let compact/self-play recording consume semantic action events and request only initial/final
  fingerprints; per-transition fingerprints remain enabled for existing `GameRecord` users;
- add a repeatable baseline command/scenario for random-vs-random, heuristic-vs-random, and
  heuristic self-play under both full and cheap invariant validation;
- record current CPU, memory, Python, NumPy/PyTorch versions, and thread counts.

Acceptance:

- repeated seeded runs produce byte-identical semantic action records and terminal summaries;
- the bounded pool never grows retained state/record count above its configured games in flight;
- hash-disabled compact recording produces the same semantic actions and final result as the
  existing per-transition-hash runner mode;
- batched and sequential baseline results remain equal;
- no baseline policy receives `GameState`;
- a baseline performance report establishes games/s, actions/s, decision count, game length, and
  peak RSS for later comparison.

### Stage 1 — Public value-position and encoder freeze

**Dependencies:** Stage 0. **Critical freeze:** yes.

Deliverables:

- a pre-freeze feasibility spike over a broad paid-turn corpus, including Draw, automatic and
  choice-producing Dogma, second-action Monument progress, Fission aftermath, private unsplayed
  stacks, and terminal candidates; compare immediate-transition and fallback-rolled Dogma
  boundaries and record the chosen semantics;
- `PublicBoundary`, `PublicDecisionContext`, `PublicTurnProgress`, `ValuePosition`, and relation
  enums;
- audited builders for current decisions and original-viewer afterstates;
- `EncoderManifest` and `FlatObservationEncoder`;
- committed encoder-v1 layout fixture containing exact feature names, offsets, scales, `D`, and
  fingerprint;
- compact feature inspection/debug command that prints named nonzero features without importing
  engine state into model code.

Acceptance:

- fixed shape and `float32` dtype for every game phase and decision kind;
- the feasibility spike demonstrates that the chosen afterstate boundary can be represented
  without private next-chooser context before encoder v1 freezes;
- positions with equal `GameObservation` but different public Monument turn progress encode
  differently and produce correct achievement outcomes;
- deterministic bytes across processes and hash seeds;
- swapping canonical seats while preserving the same self/opponent position yields equal vectors;
- existing hidden-equivalent-state fixtures encode equally;
- `covered_count=None` differs from a known zero-card count;
- no engine/model import cycle and no PyTorch dependency in the contracts or encoder tests;
- schema/fingerprint mismatch fails loudly.

### Stage 2 — Value model, evaluator, and checkpoints

**Dependencies:** Stage 1. **May run in parallel with early Stage 3 schema work.**

Deliverables:

- exact `D -> 128 tanh -> 1 sigmoid` module;
- single and batched CPU `BatchValueEvaluator` using `torch.inference_mode()`;
- configurable microbatch limit and PyTorch thread settings;
- immutable checkpoint bundle save/load, digest verification, compatibility errors, and
  state-dict-only safe loading;
- immutable `PolicyDescriptor` schema plus frozen evaluator cache keyed by checkpoint/evaluator;
- a learned-policy test double that routes synthetic candidate groups through the evaluator.

Acceptance:

- output shape is `[N]`, finite, and in `[0, 1]` for batches including `N=1`;
- batched outputs match concatenated scalar outputs within a documented tolerance;
- checkpoint round-trip is prediction-identical;
- two policy descriptors using different checkpoints can be loaded once and routed in the same
  actor/arena batch without cross-contamination;
- incompatible encoder, card/effect fingerprint, schema, or architecture is rejected;
- core imports and `make install` still work without the `ai` extra;
- fixed fixture weights produce stable expected values on CPU.

### Stage 3 — Compact replay, dataset extraction, and terminal training

**Dependencies:** Stage 1; model training portion also depends on Stage 2.

Deliverables:

- strict compact replay schema, canonical serializer, deterministic gzip (`mtime=0`) shard
  writer/reader, preassigned episode/shard manifest, and atomic sealing;
- actor-side episode recorder that captures semantic actions and provenance without full
  decisions or per-step hashes;
- replay verifier and `ValuePosition` example extractor;
- materialized NumPy dataset shards and episode-level train/validation split;
- terminal-outcome optimizer, metrics, early stopping, and checkpoint writer;
- tiny deterministic fixture dataset committed only if small enough for normal tests.

Acceptance:

- compact replay regenerates the same terminal result and final state hash;
- edited, truncated, incompatible, or illegal episodes fail loudly;
- generated features are identical to online encoding of the same positions;
- only the terminal scalar label may depend on future play; features never read future actions,
  hidden setup, or authoritative state fields outside the audited position builder;
- split leakage by setup-provenance digest is impossible, including duplicate seeds/setups;
- completion order changes do not change sealed shard bytes;
- a tiny CPU dataset overfits predictably and loss decreases;
- repeated training with the same config/seed is reproducible within documented PyTorch CPU
  tolerance;
- compact replay bytes/game are reported against full `GameLog` bytes/game.

### Stage 4 — Information-safe batched afterstate policy

**Dependencies:** Stages 1 and 2. **Highest design risk.**

Deliverables:

- current-information determinization sampler whose public API accepts `InformationSetSpec`, not
  live `GameState`, for stable paid-turn boundaries;
- trusted candidate expander using normal `apply_action()` and original-viewer observation;
- candidate router, exact terminal handling, batched evaluation, sample averaging, and temperature
  selection;
- heuristic fallback for setup/effect decisions and typed sampler failure policy;
- batch scheduler over `PullGameRunner.pending()` that flattens candidates from many games into
  one model call and routes semantic actions back without legal indices.

Acceptance:

- every selected action is one of the exact current `legal_actions`;
- hidden-equivalent authoritative states plus the same policy RNG produce identical
  `InformationSetSpec` values, sampled candidate feature batches, and selected-action
  distributions;
- changing the real hidden next card cannot change the candidate batch before commitment;
- each sampled state reproduces the chooser observation and current legal semantic actions and
  passes invariants;
- a terminal candidate bypasses the network and receives exact viewer-relative utility;
- candidate afterstates are observed as the original chooser even when the next decision belongs
  to the opponent;
- batch inference and one-game-at-a-time inference give the same result at temperature zero;
- rebatching does not change stochastic selections with fixed per-decision RNG seeds;
- simultaneous setup, nested dogma decisions, terminal-mid-batch behavior, and sampler failure do
  not corrupt another game;
- acceptance corpus records zero sampler failures and no true-state fallback path exists.

### Stage 5 — Frozen-checkpoint iterative self-play

**Dependencies:** Stages 3 and 4.

Deliverables:

- actor loop that keeps a configurable number of games in flight and refills completed slots;
- generation manifest with run seed, policy pool, seat assignments, temperatures,
  determinization count, validation level, shard limits, and action ceiling;
- immutable replay shard sealing and atomic checkpoint publication;
- bootstrap generation using heuristic/random policies, then learned generations using frozen
  policy descriptors and checkpoints;
- resumable iteration state that detects complete/incomplete shards and never silently duplicates
  episode IDs;
- graceful stop at episode or shard boundaries.

Reasonable CPU defaults:

- one actor process;
- 32 games in flight;
- PyTorch inference threads set to 1 unless profiling shows 2 is faster;
- one determinization per decision;
- cheap engine validation during long generation, with a small mirrored full-validation sample;
- 256 episodes per compact replay shard;
- temperature `0.15` for learned self-play, configurable by generation;
- latest learned policy in both seats for the simplest first learned generation, followed by a
  pool mix once two or more learned policy descriptors exist.

After a pool exists, the default opponent sampling mix is 50% latest learned policy, 25% previous
learned policy, and 25% uniform from older retained policies/baselines. Matchups and seats are
written before games start so resumption cannot change the sample distribution.

Acceptance:

- one command performs bootstrap replay generation, dataset build, training, checkpoint freeze,
  learned self-play, and candidate training without manual file edits;
- actors never observe a partially trained checkpoint or partially resolved policy descriptor;
- interruption and resume produce no duplicate/missing episode IDs;
- re-running a sealed generation from its manifest reproduces semantic action streams at
  temperature zero and reproduces the configured stochastic streams otherwise;
- all output files are beneath one run directory and `artifacts/` remains git-ignored;
- a small end-to-end CPU smoke iteration completes inside the test/CI time budget under a `slow`
  marker; normal tests use synthetic/tiny fixtures.

### Stage 6 — Paired arena and policy/checkpoint pools

**Dependencies:** Stages 2 and 4; promotion workflow also depends on Stage 5.

Deliverables:

- immutable `ArenaManifest`, `MatchPair`, `ArenaResult`, `PolicyDescriptor`, and policy/checkpoint
  pool schemas;
- for every base setup seed, two games with agents' seats swapped;
- deterministic temperature-zero evaluation by default;
- candidate-versus-incumbent, recent-policy/checkpoint pool, random, and heuristic match suites;
- W/D/L, utility score, seat-specific results, terminal-reason distribution, game length, and
  paired confidence intervals;
- JSON and human-readable table reports;
- champion pointer/manifest that references a policy descriptor and checkpoint; it never copies or
  overwrites either artifact.

Statistics default:

- candidate score is `1` for win, `0.5` for draw, `0` for loss;
- one statistical unit is the mean candidate score across the two seat-swapped games sharing a
  setup seed;
- report the mean and a deterministic 95% percentile bootstrap interval over seed-pair units
  using exactly 10,000 resamples; the manifest fixes the bootstrap algorithm/RNG version;
- also report raw W/D/L with seat breakdown; do not treat the two games in a pair as independent
  for the primary interval;
- compare each pool opponent separately; any weighted aggregate is stratified by opponent and
  uses weights fixed in the manifest before games begin.

The first compatible learned checkpoint becomes the bootstrap champion and is marked as such
without a statistical promotion claim. Every later promotion arena predeclares exactly 200 seed
pairs before play; the pair count cannot be extended after inspecting results. The candidate
replaces the incumbent only if its 95% lower bound is above `0.5`. Inconclusive candidates remain
immutable pool members but do not replace the champion. Different fixed pair counts or thresholds
are allowed only through a new pre-game manifest; adaptive repeated peeking is out of scope.

Acceptance:

- seat swap is exact and setup seed pairs are complete or rejected;
- reversing candidate/opponent labels complements the reported score;
- repeated arena runs from one manifest are byte-identical at temperature zero;
- confidence interval fixtures match independently checked small examples;
- checkpoint/policy pool membership and weighting are explicit and reproducible;
- reports identify first-player/seat bias instead of hiding it in aggregate win rate.

### Stage 7 — Profiling, tuning, and release gate

**Dependencies:** all prior stages.

Deliverables:

- `profile` scenarios for engine-only play, encoding, model inference/training, determinization,
  afterstate expansion, replay extraction, self-play, and arena;
- machine-readable timing/throughput JSON plus a concise Markdown summary;
- selected CPU defaults for games in flight, inference microbatch, PyTorch thread count,
  determinizations, validation level, and replay shard size;
- a bottleneck analysis and explicit recommendation for the next scale step: more actor
  processes, a dedicated inference process, or GPU batching.

Acceptance:

- all measurements include warmup, repeated timed samples, exact command/config, environment,
  wall time, CPU time where available, and peak RSS;
- batch-size sweeps justify chosen defaults rather than assuming larger is better;
- the profiler can disable integrity hashes and full invariants only through explicit config, and
  correctness spot checks compare cheap/self-play mode with full validation;
- `make check` passes without requiring a long training run;
- the documented slow smoke, one complete learned iteration, and the final paired arena all pass;
- artifacts and docs identify remaining performance and strength limitations honestly.

## 7. Dependency graph and parallel work

```text
Stage 0
   |
Stage 1: public position + encoder freeze
   |-------------------\
Stage 2: model/checkpoint  Stage 3a: replay schema
   |                    /
Stage 3b: dataset/training
   |
Stage 4: safe afterstate policy
   |-------------------\
Stage 5: iterative self-play  Stage 6a: arena/statistics
   \-------------------/
          Stage 7
```

Do not parallelize encoder implementations before Stage 1 freezes the manifest. The compact replay
schema can begin beside Stage 2, but feature materialization must wait for the encoder fingerprint.
Arena pairing/statistics can be built against baselines while self-play is underway, but promotion
integration waits for immutable checkpoint IDs.

## 8. Expected CLI workflows

Names may be adjusted for argparse consistency, but the workflow and artifact boundaries should
remain stable.

```bash
# Install optional CPU ML dependencies and inspect the environment.
make install-ai
uv run innovation-ai doctor

# Generate compact bootstrap games from existing non-ML agents.
uv run innovation-ai self-play \
  --run-dir artifacts/runs/bootstrap-001 \
  --games 1000 --player-1 heuristic --player-2 random --seed 1000

# Verify/replay compact episodes and materialize encoder-v1 arrays.
uv run innovation-ai dataset build \
  --replays artifacts/runs/bootstrap-001/replays/manifest.json \
  --output artifacts/runs/bootstrap-001/dataset

# Train and freeze the first value checkpoint.
uv run innovation-ai train-value \
  --dataset artifacts/runs/bootstrap-001/dataset/manifest.json \
  --output artifacts/runs/bootstrap-001/checkpoints

# Generate one learned generation with frozen weights and a resolved policy descriptor.
uv run innovation-ai self-play \
  --run-dir artifacts/runs/generation-001 \
  --policy artifacts/runs/bootstrap-001/policies/<policy-id>.json \
  --games 2000 --temperature 0.15 --determinizations 1 --seed 2000

# Run an iterative resolved configuration.
uv run innovation-ai iterate --config configs/cpu-value-baseline.toml

# Paired, seat-swapped arena against incumbent and baselines.
uv run innovation-ai arena \
  --candidate-policy artifacts/runs/generation-001/policies/<policy-id>.json \
  --opponents champion,heuristic,random \
  --seed-start 50000 --seed-pairs 200 \
  --output artifacts/runs/generation-001/arena

# Produce final CPU throughput sweeps and bottleneck report.
uv run innovation-ai profile \
  --config configs/cpu-value-baseline.toml \
  --output artifacts/profiles/milestone-2
```

Hand-authored run configuration should use TOML readable with the standard library. Every command
writes a fully resolved canonical JSON copy so defaults cannot change the interpretation of an old
run.

## 9. Artifact layout

```text
artifacts/runs/<run-id>/
  resolved-config.json
  environment.json
  run-manifest.json
  metrics.jsonl
  replays/
    manifest.json
    shard-00000.jsonl.gz
  dataset/
    manifest.json
    train-00000.npz
    validation-00000.npz
  checkpoints/
    <checkpoint-id>/
      manifest.json
      model.pt
      optimizer.pt
      metrics.json
  policies/
    <policy-id>.json
    champion.json
  arena/
    manifest.json
    matches.jsonl.gz
    report.json
    report.md
  profiles/
    report.json
    report.md
```

Large/generated artifacts remain ignored. Commit only schemas, tiny deterministic fixtures, and
human-readable benchmark summaries deliberately selected for regression documentation.

## 10. Required final throughput measurements

Run all measurements on the target 2-vCPU CPU-only devbox and record exact software/thread
configuration.

### Engine and baseline agents

- random-vs-random and heuristic-vs-random games/s;
- submitted actions/s and decisions/s;
- mean, p50, p95, and max actions/game and decisions/game;
- full versus cheap validation cost;
- state-hash-on versus state-hash-off cost where orchestration makes hashing optional.

### Encoder

- positions/s and p50/p95 latency at batch sizes `1, 32, 256, 1024`;
- final `D`, bytes/encoded position, and peak temporary allocation;
- current-decision versus afterstate encoding cost.

### Network

- forward positions/s and latency at batch sizes `1, 32, 128, 512, 2048`;
- one versus two PyTorch intra-op threads;
- training examples/s, optimizer steps/s, epoch wall time, peak RSS;
- checkpoint size and cold/warm load latency.

### Information-set sampling and afterstates

- sampler states/s, retries/sample, and failure count;
- legal branching factor mean, p50, p95, and max;
- candidate transitions/s and end-to-end policy decisions/s;
- p50/p95 decision latency for one and four determinizations;
- fraction of terminal candidates and heuristic fallbacks;
- time split among sampling, engine transitions, observation building, encoding, model forward,
  grouping, and submission.

### Replay and datasets

- compressed bytes/game and bytes/action;
- size ratio versus full audit `GameLog` on the same games;
- verified replay games/s and actions/s;
- extracted examples/s, materialized bytes/example, and shard build wall time.

### End-to-end self-play and arena

- games/hour, actions/s, decisions/s, and examples/hour;
- achieved inference batch-size distribution and actor idle fraction;
- replay writer throughput and peak queue depth;
- peak RSS and disk growth per 1,000 games;
- paired arena games/hour and total time for 200 seed pairs per opponent.

The release report must identify the top three bottlenecks and estimate whether the next improvement
should come from reducing engine/speculation work, adding CPU actors, separating inference, or
moving the evaluator/trainer to GPU. Absolute performance targets should be set after Stage 0;
relative acceptance requires batched inference to outperform a Python scalar loop and the chosen
CPU configuration to complete the documented smoke iteration without swapping or exceeding the
action ceiling.

## 11. Assumptions and details to verify during implementation

The plan proceeds with these defaults rather than blocking on questions:

1. `RandomAgent`, `SimpleHeuristicAgent`, and `PullGameRunner` are accepted Milestone 1
   deliverables; only audit, descriptors, metrics, and integration changes are expected.
2. A draw target is `0.5`, and terminal outcome is the only training label.
3. The first learned policy controls paid turn actions only; heuristic setup/effect handling is an
   intentional hybrid baseline.
4. The existing memoryless `rulebook-private-covered-v1` observation policy remains the policy for
   training and arena play.
5. One current-information determinization is the CPU self-play default; four is the initial arena
   default. Stage 7 may lower arena sampling if throughput is impractical.
6. Compact durable replays include explicit shuffled piles for portability even though this costs
   more than storing only a seed.
7. Full audit logs remain available for selected debugging games but are not emitted for every
   self-play episode.
8. The exact encoder dimension is frozen from a generated feature manifest only after the Stage 1
   information-set/afterstate spike. Network checkpoints never infer `D` from arbitrary input at
   load time.
9. Training uses all eligible nonterminal post-action examples uniformly at first. If long dogma
   sequences dominate, per-episode weighting is a later measured adjustment, not an undocumented
   default.
10. CPU reproducibility is required for fixed software/hardware configuration. Cross-version
    bitwise PyTorch reproducibility is not assumed; manifests reject or clearly warn on version
    mismatch.
11. The arena's bootstrap interval is descriptive for the paired seed population. It is not a
    claim that games are IID across arbitrary future opponent distributions.
12. No model-strength threshold beyond pipeline integrity is required for Milestone 2. Arena
    results determine the next milestone's data/model/search priorities.

Details that must be verified before their owning stage freezes:

- whether every public field needed by `PublicBoundary`, including Monument turn progress, can be
  derived without exposing private effect context;
- whether immediate one-transition Dogma afterstates or fallback-rolled macro afterstates provide
  the safer and more learnable public boundary; immediate transition is the initial default;
- the exact versioned allocation law, retry rate, and feasibility of the observation-consistent
  sampler across a broad paid-turn corpus, especially after Fission and with opponent unsplayed
  stacks;
- whether training should exclude any effect-choice afterstates whose public boundary cannot be
  represented unambiguously;
- the practical full-versus-cheap validation sampling ratio;
- PyTorch deterministic-algorithm support and best thread count for the locked CPU build;
- final compact-shard and NumPy-shard sizes on real games;
- a realistic seed-pair count and promotion interval after arena throughput is measured.

Any change that permits direct true-hidden-state candidate scoring, adds tensors to engine
contracts, or changes the information policy is a project-level decision rather than a local
optimization.

## 12. Milestone definition of done

- Existing baseline agents are reproducible and benchmarked.
- Encoder v1 is viewpoint-relative, includes audited public turn progress, is leak-tested,
  versioned, and frozen with exact `D`.
- The value model and immutable checkpoint bundle pass compatibility and round-trip tests.
- Compact semantic replays regenerate games and build terminal-labeled datasets.
- Training produces a finite, loadable checkpoint with improving tiny/real training metrics.
- The learned policy batches one-ply candidates across games without hidden-state clairvoyance.
- Frozen-checkpoint self-play completes and resumes reproducibly.
- The arena runs paired seeds and seat swaps against immutable policy pools and baselines with a
  valid paired confidence interval.
- Final CPU profiling reports every measurement in Section 10 and selects justified defaults.
- Normal `make check` is green; documented slow smoke/iteration/arena commands are green; generated
  artifacts remain out of git; docs match the implemented contracts.
