# Milestone 1 Plan — Complete Innovation Game Engine

## 1. Milestone contract

Milestone 1 delivers a complete deterministic engine for the supplied two-player Innovation
base game, including all 105 cards. It also supplies simple agents and runners so complete games
can be played and replayed. It does **not** include neural models or training.

The design must preserve the later ML path:

- every choice is a first-class decision;
- actions are semantic structured data;
- authoritative state and player observations are separate;
- the engine never invokes an agent;
- partially resolved effects are serializable;
- logs are versioned and deterministic;
- multiple waiting games can later be batched for an external policy.

Source precedence:

1. `PROJECT_GOAL.md` for architecture and scope.
2. `game-rules-plaintext/innovation_2p_base_rules.md` for general rules and clarifications.
3. `game-rules-plaintext/cards.csv` for card identities, icons, and printed effects.
4. `game-rules-plaintext/special_achievements.csv` as an index. Full rule text and linked-card
   text override its abbreviations.
5. A reviewed rules-decision record and tests for unavoidable ambiguities.

Natural-language card text is not parsed at runtime. Card behavior is implemented as structured,
reviewable effects keyed by canonical card IDs.

## 2. Facts established from the supplied data

- 105 unique cards: 15 in age 1 and 10 in every age 2–10.
- Five colors with 21 cards each.
- 158 printed dogma effects: 55 one-effect cards, 47 two-effect cards, and three three-effect
  cards (`COAL`, `SATELLITES`, `THE INTERNET`).
- Card image/icon positions are the meaningful CSV columns `Symbol 1`, `Symbol 4`, `Symbol 5`,
  and `Symbol 6`; `Symbol 2` and `Symbol 3` are empty.
- Each card has exactly one `hex` card-image position and three functional icons.
- `Main Symbol` is the featured dogma icon; it must not be inferred from icon frequency.
- Normalize `bulb` to the lightbulb icon and `hex` to a non-icon image slot.
- The special-achievement linked cards grant alternate routes whose predicates differ from the
  ordinary automatic predicates.

## 3. Contracts to freeze before parallel implementation

These contracts are decided and documented before card agents fan out. Changing them later
invalidates logs, fixtures, and agent action encodings.

### 3.1 Canonical identity and geometry

Define stable enums/value types for:

- `CardId`, `PlayerId`, `Color`, `Icon`, `IconSlot`, `SplayDirection`;
- normal and special achievement IDs;
- action, decision, event, effect-frame, and terminal-reason kinds.

Canonical card IDs are semantic slugs independent of display names. Define explicit normalization
for punctuation and spacing (`A.I.`, `THE WHEEL`, `CITY STATES`) and an alias for the rulebook's
`Publication` clarification versus CSV `PUBLICATIONS`.

Freeze this slot mapping:

| CSV | Position | Covered-card visibility |
|---|---|---|
| `Symbol 1` | top-left | right |
| `Symbol 4` | bottom-left | right or up |
| `Symbol 5` | bottom-center | up |
| `Symbol 6` | bottom-right | left or up |

Top cards expose all three functional icons. `hex` never contributes an icon.

### 3.2 State and transition boundary

The authoritative state contains:

- ordered age 1–10 supply piles and hidden normal-achievement identities;
- each player's hand, five ordered color stacks and splay directions, score pile, and claimed
  achievements;
- active player, turn number, paid actions remaining, and per-turn counters;
- pending serializable effect frames and effect-scoped variables;
- monotonic decision/event IDs;
- rules version, information policy, terminal result, and setup provenance.

The public transition contract is conceptually:

```text
apply_action(state, action) -> (new_state, next_decision_or_terminal)
```

The input state is not observably mutated. Internal controlled mutation is allowed, but every
paused state—including mid-dogma and mid-repeat—must serialize, hash, clone, and resume.

### 3.3 Actions and decisions

A `Decision` contains:

- stable decision ID and semantic decision kind;
- acting/choosing player;
- source card/effect where relevant;
- player-safe observation;
- deterministically ordered legal actions.

An `Action` uses semantic IDs and structured fields. It never relies on display strings or an
index into the current legal-action list. Multi-card and ordering choices use card IDs, explicit
incremental selection, and a semantic finish/decline action. Arbitrary stack rearrangement uses an
ordered tuple of card IDs rather than positional indices.

Every `may`, tied highest/lowest choice, target, color, value, branch, return order, meld/tuck
order, and nested choice is represented explicitly. Mandatory partial execution offers only the
choices that remain legal.

### 3.4 Effect execution and provenance

Use explicit resumable effect frames, not Python generator state or arbitrary callbacks. Frames
hold serializable progress and return one of: continue, await decision, complete, abort dogma, or
terminal.

All card behavior composes shared primitives for:

- draw, reveal, keep;
- meld, tuck, score, return, transfer, exchange, remove;
- splay, unsplay, rearrange;
- claim achievement;
- choose cards/colors/players/values/directions/branches/order;
- condition, repeat, all-or-none, bounded selection, and nested non-demand execution.

Every mutation emits provenance: actor, chooser, executor, dogma activator, source card/effect,
source/destination, moved card IDs, demand/shared/nested flags, turn ID, and dogma-action ID. This
is required for `if you do`, demand-dependent later effects, the sharing bonus, and Monument.

### 3.5 Observation and information policy

`observe(state, player, policy)` returns a detached immutable observation. The default follows
the supplied rulebook convention:

- supply counts/values and hand/score counts and card values are public as stated by the rules;
- opponent hand and score identities are hidden;
- normal-achievement identities and supply order are hidden from everyone;
- opponent covered-card identities and unsplayed-stack sizes are hidden;
- a player sees all identities in their own hand, score pile, and board.

The policy is versioned and recorded in every log. Leak tests compare states differing only in
hidden information and require equal observations.

### 3.6 Setup randomness

Consume an explicit seeded RNG during setup only: shuffle each age, set aside ages 1–9 normal
achievements, and deal. Thereafter ordered piles plus submitted actions fully determine play.
Starting-card selections are simultaneous decisions; alphabetical title order determines the
first player after both choices are submitted.

### 3.7 Serialization and logs

Version state, decisions, actions, effect frames, observations, terminal results, and game logs.
A game log records at least:

- format/engine/rules/information-policy versions;
- card-data fingerprint;
- setup seed or explicit setup;
- semantic action per decision;
- deterministic state hash after every transition;
- terminal result and reason.

Replay rejects incompatible versions/fingerprints and verifies every state hash.

## 4. Rules-decision gate

Before shared contracts are frozen, one agent should create `docs/RULES_DECISIONS.md` with a
reviewed answer and tests for each item below. Coding agents must not choose independently.

1. **Value-zero draw:** no top card or empty score can produce value 0, but no age-0 pile exists.
   Proposed default: requested age 0 starts at the age-1 supply.
2. **Sharing-bonus change:** define whether non-card changes such as achievement claims count.
   Proposed default: use the rule's listed physical/reveal/movement changes; claiming alone does
   not qualify.
3. **Special-achievement check order:** define deterministic handling when one atomic operation
   makes different players eligible for different special achievements and either could be a
   sixth achievement.
4. **Bulk operation boundaries:** identify which exchanges/transfers/returns are simultaneous for
   achievement checks and observations.
5. **Multiple-card ordering:** define who chooses order when multiple cards return to one supply
   or meld/tuck into one stack.
6. **Nested execution:** confirm that Computers/Satellites/Robotics/Self Service/Software execute
   only selected non-demand effects, without a new share pass or sharing bonus, while retaining
   causal attribution to an outer shared execution.
7. **Fission abort:** confirm whether its mass removal skips all remaining dogma work and the
   sharing bonus while allowing later turns if no ordinary win condition fired.
8. **Societies:** decide whether its non-`top` wording permits selecting a covered board card.
9. **A.I.:** decide whether Robotics and Software may be top cards on different players' boards.
10. **Empty universal predicates:** decide Translation/Astronomy behavior when the relevant set of
    top cards is empty.
11. **Democracy with zero returns:** define comparison against previous executors with no returns.
12. **Automatic-achievement timing:** define the atomic points at which Empire, World, Wonder, and
    Universe are checked, and active-player priority for simultaneous eligibility.
13. **Hidden equivalent choices:** define canonical internal selection when public information
    cannot distinguish multiple hidden cards of the same value.
14. **Splay after unusual movement:** define preservation/collapse for covered-card transfer,
    exchange, remove, and rearrange beyond the explicit meld/tuck/Publication rules.

## 5. Implementation work packages

Each package is a handoff-sized effort with dependencies and acceptance criteria. Agents should
own only the listed files after contracts freeze.

### WP0 — Reconcile scaffold and documentation

**Dependencies:** none. **Parallelism:** one owner.

Deliverables:

- align package/project name with `innovation-ai`;
- store `PROJECT_GOAL.md` and this plan;
- make core dependencies ML-free and retain NumPy/PyTorch only in the AI extra;
- align `AGENTS.md`, README, Makefile, coverage, and test markers;
- create the rules-decision document without implementing engine behavior.

Acceptance: `make check` passes; no engine behavior has been added.

### WP1 — Card catalog and immutable value types

**Dependencies:** WP0 and reviewed rules decisions. **Parallelism:** one owner.

Deliverables:

- canonical enums/IDs and card value object;
- packaged card data loaded with `importlib.resources`;
- immutable card registry and data fingerprint;
- static validation of all supplied CSV assumptions;
- ordinary special-achievement definitions separated from linked-card alternate routes.

Acceptance:

- exactly 105 unique cards, correct age/color histograms;
- exactly three functional icons and one image slot per card;
- featured icon is valid;
- source and packaged data are byte-identical;
- wheel-build smoke test can load the registry.

### WP2 — Authoritative state, zones, board geometry, and queries

**Dependencies:** WP1. **Parallelism:** one owner; may run alongside WP3 after state field names
are frozen.

Deliverables:

- state/player/stack/supply/achievement structures;
- setup pile construction and card conservation model;
- shared zone operations and change records;
- score, top/bottom/beneath, selector, splay, and visible-icon queries;
- clone and deterministic state hash.

Acceptance:

- splay matrix tests for every direction and icon slot;
- upward-only draw fallback and return-to-bottom tests;
- stack collapse at zero/one card;
- exchange with an empty side;
- conservation and unique-location invariants after every primitive.

### WP3 — Action, decision, observation, setup, and turn protocol

**Dependencies:** WP1 plus frozen state skeleton. **Parallelism:** one owner, parallel with WP2.

Deliverables:

- semantic actions/decisions/terminal results and typed errors;
- deterministic legal action enumeration;
- simultaneous setup meld choice and first-player selection;
- one-action first turn, two-action normal turns;
- information-policy-aware observations.

Acceptance:

- every enumerated action applies; non-legal actions raise `IllegalAction`;
- legal ordering is stable across processes/hash seeds;
- hidden-information equality/property tests;
- setup and first-turn action-count tests.

**Freeze point A:** merge WP1–WP3 and publish exact contracts before downstream parallel work.

### WP4 — Resumable effects and shared primitives

**Dependencies:** Freeze point A. **Parallelism:** single owner; critical path.

Deliverables:

- serializable frame stack and resume protocol;
- effect context and scoped variables;
- movement, choice, condition, repeat, batch, and nested-execution primitives;
- event/change provenance;
- serialization round-trip of paused effects.

Use representative specification tests before broad card work:

- Pottery: optional bounded multi-select and order;
- Metalworking: repeated draw/reveal/branch;
- Machinery: demand plus mandatory exchange;
- Publications: arbitrary stack ordering;
- Fission: dogma abort and mass removal;
- Self Service: nested non-demand execution.

These tests can use synthetic effect fixtures until actual card modules are assigned.

Acceptance: each representative flow can pause at every choice, serialize, restore, and reach an
identical state hash.

### WP5 — Dogma orchestration

**Freeze-B subset status (August 25, 2026): complete.** The shared runtime contract and six-card
vertical slice are documented in `docs/WP5_FREEZE_B_CONTRACT.md`; the remaining 99 production
card programs stay in WP7 scope.

**Dependencies:** WP4. **Parallelism:** one owner, often the WP4 owner.

Deliverables:

- freeze featured-icon counts at dogma start;
- demand vulnerability/immunity;
- per-effect opponent-first sharing then active execution;
- yourself-only/nested no-share behavior;
- partial execution;
- at most one causally justified sharing-bonus draw;
- immediate unwind on terminal and explicit unwind on dogma abort.

Acceptance: scenario matrix covers vulnerable, equal, and stronger opponent icon counts; icon
changes mid-dogma do not alter eligibility; shared effects resolve in correct order; demands do
not produce sharing credit; no-op sharing gives no free draw.

### WP6 — Achievements and terminal handling

**Dependencies:** WP2 and WP4. **Parallelism:** may run alongside WP5 against frozen primitives.

Deliverables:

- normal achievement legality/claiming;
- five automatic predicates and five linked-card routes;
- per-turn score/tuck counters with provenance exclusions;
- sixth-achievement, draw-above-10, card-effect win, unique-most/lowest, tie, and draw results;
- atomic check order from the reviewed rules decisions.

Acceptance: immediate wins stop all remaining effects; transferred/exchanged score cards do not
count for Monument; same-special simultaneous eligibility follows active-player priority; deck
exhaustion uses score then achievement count then draw.

**Freeze point B:** the shared primitive/effect signatures and behavior are frozen by
`docs/WP5_FREEZE_B_CONTRACT.md`. Card agents may now fan out; contract changes return to the
owning work package instead of being improvised. The freeze does not waive the 99 unimplemented
WP7 card programs listed in that document.

### WP7 — Card implementations, waves 1–7

**Dependencies:** Freeze point B. **Parallelism:** multiple agents with exclusive module/test
ownership. Scope is all 105 cards.

Implementation modules should be partitioned by age to avoid merge conflicts, while merge order
follows capability waves. No agent edits a shared registry list; registration is discoverable or
assembled from age modules by an owner-controlled mechanism.

#### Wave 1 — basic zones, draw destinations, optional costs, and subset choices

`THE WHEEL`, `WRITING`, `SAILING`, `EXPERIMENTATION`, `AGRICULTURE`, `MATHEMATICS`,
`DOMESTICATION`, `CALENDAR`, `POTTERY`, `TOOLS`, `CURRENCY`, `ECOLOGY`, `STEM CELLS`,
`SUBURBIA`, `QUANTUM THEORY`.

#### Wave 2 — board geometry, icons, splaying, and positional movement

`CODE OF LAWS`, `PHILOSOPHY`, `FERMENTING`, `PAPER`, `PRINTING PRESS`, `REFORMATION`,
`COAL`, `MEASUREMENT`, `ATOMIC THEORY`, `CANNING`, `INDUSTRIALIZATION`, `METRIC SYSTEM`,
`PUBLICATIONS`, `RAILROAD`, `FLIGHT`, `THE INTERNET`.

#### Wave 3 — batches, reveals, exchanges, aggregate tests, and all-or-none operations

`CLOTHING`, `MASONRY`, `CANAL BUILDING`, `EDUCATION`, `ALCHEMY`, `TRANSLATION`,
`PERSPECTIVE`, `CHEMISTRY`, `STEAM ENGINE`, `PHYSICS`, `ENCYCLOPEDIA`, `MACHINE TOOLS`,
`BICYCLE`, `ELECTRICITY`, `LIGHTING`, `ANTIBIOTICS`, `EVOLUTION`, `GENETICS`,
`MINIATURIZATION`.

#### Wave 4 — demands and demand-scoped provenance

`ARCHERY`, `CITY STATES`, `CONSTRUCTION`, `MAPMAKING`, `MONOTHEISM`, `ENGINEERING`,
`FEUDALISM`, `ANATOMY`, `GUNPOWDER`, `NAVIGATION`, `BANKING`, `STATISTICS`,
`THE PIRATE CODE`, `EMANCIPATION`, `VACCINATION`, `COMBUSTION`, `EXPLOSIVES`,
`REFRIGERATION`, `CORPORATIONS`, `MOBILITY`, `DATABASES`, `GLOBALIZATION`.

#### Wave 5 — complex cross-player movement, targeting, and source-stack references

`ROAD BUILDING`, `COMPASS`, `MACHINERY`, `MEDICINE`, `OPTICS`, `ENTERPRISE`, `SOCIETIES`,
`SANITATION`, `CLASSIFICATION`, `SERVICES`, `SPECIALIZATION`, `ROCKETRY`, `MASS MEDIA`,
`SOCIALISM`, `SKYSCRAPERS`, `COMPOSITES`, `BIOENGINEERING`.

#### Wave 6 — repetition, action-scoped state, direct achievements, and interrupts

`METALWORKING`, `MYSTICISM`, `OARS`, `COLONIALISM`, `ASTRONOMY`, `INVENTION`, `DEMOCRACY`,
`EMPIRICISM`, `COLLABORATION`, `FISSION`, `A.I.`.

#### Wave 7 — nested execution

`COMPUTERS`, `SATELLITES`, `ROBOTICS`, `SELF SERVICE`, `SOFTWARE`.

Per-card acceptance matrix, where applicable:

1. no-effect/minimum state;
2. ordinary successful state;
3. optional decline and bounded stop;
4. all material branches;
5. tied highest/lowest choices;
6. partial execution;
7. supply escalation and draw-above-10;
8. opponent sharing before active execution;
9. demand vulnerable/equal/immune states;
10. automatic achievement or terminal interruption;
11. deterministic replay through every nested decision.

A registry coverage test must report exactly which cards are implemented. Milestone 1 is not done
until all 105 are registered and tested; missing cards fail loudly and never act as no-ops.

### WP8 — Basic agents and batch-ready runners

**Dependencies:** Freeze point A; integration tests depend on cards. **Parallelism:** independent
owner alongside WP4–WP7.

Deliverables:

- minimal `Agent` protocol consuming one `Decision` and returning one `Action`;
- seeded random agent;
- scripted agent for fixtures;
- simple heuristic agent;
- optional human CLI adapter;
- single-game runner and pull-based multi-game runner with `pending()`/`submit()` semantics.

Acceptance: the engine never imports or invokes agents; same setup and agent seeds produce
identical records; multi-game runner results equal sequential runs.

### WP9 — Versioned serialization, logs, replay, and CLI

**Dependencies:** Freeze point A; effect-frame serialization completes after WP4.
**Parallelism:** independent owner.

Deliverables:

- state/action/decision/terminal/log schemas;
- save/load and replay with version/fingerprint/hash checks;
- CLI commands for play and replay while retaining `doctor`.

Acceptance: play → log → replay reproduces every state hash; edited/truncated/incompatible logs
fail loudly; a mid-effect state round-trips and resumes identically.

### WP10 — Invariants, property tests, and fuzzing

**Dependencies:** begins with WP2 and grows throughout. **Parallelism:** dedicated owner.

Deliverables:

- conservation, unique location, score consistency, icon geometry, turn progression, legal-action
  completeness, no-hidden-info-leak, transition purity, and terminal immutability properties;
- deterministic random-vs-random fuzzing over many seeds;
- small golden game records.

Acceptance: default suite remains fast; a larger marked fuzz run completes without engine errors,
state-hash divergence, leaked observations, or step-ceiling hangs.

## 6. Dependency graph and parallel handoff

```text
WP0 -> rules decisions -> WP1
                       -> WP2 ----\
                       -> WP3 -----+--> Freeze A -> WP4 -> WP5 --\
                                  |             \-> WP6 --------+-> Freeze B -> WP7 waves
                                  +------------------------------+-> WP8
                                  +------------------------------+-> WP9
                                  \--------------------------------> WP10 continuously
```

Recommended handoff order:

1. One foundation agent: WP0 plus rules-decision draft.
2. One catalog/contracts agent: WP1 and state skeleton.
3. Two parallel agents: WP2 and WP3.
4. One critical-path effect agent: WP4–WP5; one achievement agent: WP6.
5. In parallel from Freeze A: runner agent (WP8), serialization agent (WP9), invariant agent
   (WP10).
6. After Freeze B: card agents with exclusive age-module/test ownership, merging waves in order.
7. Final integration owner: full-game fuzzing, all-card coverage, docs, and release checklist.

Do not split WP4 among agents. Do not fan out card work before Freeze B. Those shortcuts create
incompatible action schemas and duplicate primitives.

## 7. Suggested file ownership after contracts freeze

| Area | Exclusive owner |
|---|---|
| `innovation/catalog*`, packaged data | WP1 |
| `innovation/state*`, `zones*`, `board*`, queries | WP2 |
| `innovation/actions*`, `decisions*`, `observation*`, setup/turn | WP3 |
| `innovation/effects/`, primitive contracts | WP4/WP5 |
| achievements/terminal modules | WP6 |
| one `cards/ageNN*` and matching tests | one card agent |
| `agents/`, runner | WP8 |
| logs/replay/CLI | WP9 |
| invariant/fuzz suites | WP10 |
| shared contracts and plan docs | integration owner only after freeze |

If a card agent needs a missing primitive, they file a contract change for the effect owner rather
than adding a private card-specific mutation path.

## 8. Milestone 1 definition of done

- Complete two-player games run using all 105 base cards.
- Every setup, turn, and nested dogma choice is a first-class decision.
- No action uses a display string or transient legal-action index.
- The transition boundary is deterministic and does not mutate its input.
- Partially resolved state serializes, clones, hashes, and resumes.
- Observation leak tests cover deck order, achievements, hands, scores, and covered boards.
- All general rules, official card clarifications, five special achievements, and 105 card effects
  have focused tests.
- Random, scripted, heuristic, and human-compatible control can play through the same protocol.
- The pull-based runner handles many waiting games without coupling the engine to policy code.
- Versioned logs replay with per-step hash verification.
- Full random-vs-random fuzzing completes without engine errors or invariant failures.
- `make check` is green, the working tree is clean, and docs match behavior.

## 9. Per-agent handoff template

Every implementation request should state:

- work-package ID and owned files;
- exact prerequisite commit/freeze version;
- rules sections and cards in scope;
- contracts that may not change;
- required acceptance tests;
- commands to run;
- unresolved question escalation path;
- explicit out-of-scope areas.

Every agent response should include:

- files changed and behavior added;
- rules interpretations used;
- tests added and commands/results;
- remaining ambiguities or follow-up contract requests;
- confirmation that no out-of-scope shared contract was changed.
