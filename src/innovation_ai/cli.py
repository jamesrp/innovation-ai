"""Command-line utilities for the development workspace."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import platform
import sys
import tomllib
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from innovation_ai import __version__
from innovation_ai.innovation.logs import GameLogError, load_game_log, save_game_log
from innovation_ai.innovation.replay import GameLogRecorder, ReplayError, replay_game_log
from innovation_ai.innovation.state import build_setup_state


def _doctor() -> int:
    """Print a concise environment report."""
    torch_available = importlib.util.find_spec("torch") is not None
    print(f"innovation-ai {__version__}")
    print(f"python {platform.python_version()}")
    print(f"pytorch {'available' if torch_available else 'not installed (run: make install-ai)'}")
    print("device cpu")
    return 0


def _play(seed: int, log_path: Path, max_transitions: int) -> int:
    """Run the current deterministic first-legal-action policy and save its log."""

    recorder = GameLogRecorder(build_setup_state(seed))
    for _ in range(max_transitions):
        decisions = recorder.decisions()
        if not decisions:
            break
        recorder.submit(decisions[0].legal_actions[0])
    game_log = recorder.game_log()
    if game_log.terminal_result is None:
        print(
            f"play stopped without a terminal result after {game_log.transition_count} transitions",
            file=sys.stderr,
        )
        return 2
    save_game_log(game_log, log_path)
    winners = ",".join(player.value for player in game_log.terminal_result.winners) or "draw"
    print(
        f"saved {game_log.transition_count}-transition game to {log_path} "
        f"({game_log.terminal_result.reason.value}: {winners})"
    )
    return 0


def _replay(log_path: Path) -> int:
    """Load and hash-verify one game log."""

    result = replay_game_log(load_game_log(log_path))
    print(
        f"verified {result.transitions_replayed} transitions; "
        f"outcome {result.outcome.value}; hash replay matched"
    )
    return 0


def _baseline(args: argparse.Namespace) -> int:
    """Run the deterministic Stage-0 baseline suite and write JSON/Markdown reports."""

    from innovation_ai.harness.benchmark import (
        BaselineBenchmarkConfig,
        run_baseline_benchmark,
    )
    from innovation_ai.innovation.zones import ValidationLevel

    levels = tuple(ValidationLevel(item) for item in args.validation)
    report = run_baseline_benchmark(
        BaselineBenchmarkConfig(
            run_seed=args.seed,
            games_per_scenario=args.games,
            games_in_flight=args.games_in_flight,
            validation_levels=levels,
            max_actions_per_game=args.max_actions,
        )
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "baseline.json").write_text(f"{report.to_json()}\n", encoding="utf-8")
    (args.output / "baseline.md").write_text(report.to_markdown(), encoding="utf-8")
    print(report.to_markdown(), end="")
    return 0


def _baseline_seat_policy(name: str) -> Any:
    from innovation_ai.agents.descriptors import (
        RANDOM_AGENT_DESCRIPTOR,
        SIMPLE_HEURISTIC_AGENT_DESCRIPTOR,
    )
    from innovation_ai.training.self_play import SeatPolicy

    descriptor = {
        "random": RANDOM_AGENT_DESCRIPTOR,
        "heuristic": SIMPLE_HEURISTIC_AGENT_DESCRIPTOR,
    }.get(name)
    if descriptor is None:
        raise ValueError(f"unsupported baseline policy {name!r}")
    return SeatPolicy(descriptor.descriptor_id, "baseline", descriptor=descriptor)


def _self_play(args: argparse.Namespace) -> int:
    """Generate one immutable compact-replay generation."""

    from innovation_ai.training.checkpoint import load_policy_descriptor
    from innovation_ai.training.self_play import (
        GenerationConfig,
        SeatPolicy,
        plan_generation,
        run_generation,
    )

    if args.policy is not None:
        descriptor = replace(
            load_policy_descriptor(args.policy),
            temperature=args.temperature,
            determinization_count=args.determinizations,
        )
        policy = SeatPolicy(descriptor.policy_id, "learned", learned=descriptor)
        policies = (policy,)
        seat_pairs = ((policy.policy_id, policy.policy_id),)
        policy_dir = args.run_dir / "policies"
        policy_dir.mkdir(parents=True, exist_ok=True)
        descriptor.save(policy_dir / f"{descriptor.policy_id}.json")
    else:
        player_1 = _baseline_seat_policy(args.player_1)
        player_2 = _baseline_seat_policy(args.player_2)
        policies = tuple({item.policy_id: item for item in (player_1, player_2)}.values())
        seat_pairs = ((player_1.policy_id, player_2.policy_id),)
    manifest = plan_generation(
        GenerationConfig(
            args.run_dir.name,
            args.seed,
            args.generation,
            max_games_in_flight=args.games_in_flight,
            shard_episode_limit=args.shard_size,
            action_ceiling=args.max_actions,
            validation_level=args.validation,
        ),
        policies,
        seat_pairs,
        args.games,
    )
    sealed = run_generation(
        args.run_dir,
        manifest,
        checkpoint_root=args.checkpoint_root,
    )
    print(f"sealed {len(sealed)} replay shards beneath {args.run_dir}")
    return 0


def _replay_sources(path: Path) -> tuple[Path, ...]:
    """Resolve a replay directory, self-play manifest, or one gzip shard."""

    if path.is_dir():
        directory = path / "replays" if (path / "replays").is_dir() else path
        sources = tuple(sorted(directory.glob("*.jsonl.gz")))
    elif path.name == "run-manifest.json":
        sources = tuple(sorted((path.parent / "replays").glob("*.jsonl.gz")))
    else:
        sources = (path,)
    if not sources:
        raise ValueError(f"no compact replay shards found at {path}")
    return sources


def _dataset_build(args: argparse.Namespace) -> int:
    """Verify compact episodes and materialize deterministic encoder arrays."""

    from innovation_ai.training.dataset import materialize_dataset

    manifest = materialize_dataset(
        _replay_sources(args.replays),
        args.output,
        validation_fraction=args.validation_fraction,
        split_salt=args.split_salt,
        episodes_per_shard=args.episodes_per_shard,
    )
    print(
        f"materialized {manifest.counts.example_count} examples from "
        f"{manifest.counts.episode_count} episodes at {args.output / 'manifest.json'}"
    )
    return 0


def _train_value(args: argparse.Namespace) -> int:
    """Train terminal outcomes and publish a checkpoint plus complete policy descriptor."""

    from innovation_ai.training.checkpoint import (
        PolicyDescriptor,
        load_checkpoint_manifest,
    )
    from innovation_ai.training.optimize import TrainingConfig, train_terminal_outcomes

    result = train_terminal_outcomes(
        args.dataset,
        args.output,
        config=TrainingConfig(
            seed=args.seed,
            max_epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            torch_num_threads=args.torch_threads,
        ),
        generation=args.generation,
        creation_command="innovation-ai train-value",
    )
    checkpoint = load_checkpoint_manifest(result.checkpoint_directory)
    policy = PolicyDescriptor.from_checkpoint(
        checkpoint,
        temperature=args.temperature,
        determinization_count=args.determinizations,
    )
    policy_dir = args.policy_output or args.output.parent / "policies"
    policy_dir.mkdir(parents=True, exist_ok=True)
    policy_path = policy_dir / f"{policy.policy_id}.json"
    policy.save(policy_path)
    print(
        f"saved checkpoint {checkpoint.checkpoint_id} at {result.checkpoint_directory}; "
        f"policy {policy.policy_id} at {policy_path}"
    )
    return 0


def _inspect_encoding(args: argparse.Namespace) -> int:
    """Print named nonzero encoder-v1 features from a deterministic engine boundary."""

    from innovation_ai.harness.policy import (
        ValuePositionKind,
        build_current_value_position,
        build_value_position,
    )
    from innovation_ai.innovation.protocol import apply_action, current_decision
    from innovation_ai.innovation.state import build_setup_state
    from innovation_ai.innovation.types import PlayerId
    from innovation_ai.training.encoding import FlatObservationEncoder

    state = build_setup_state(args.seed)
    for _ in range(args.steps):
        decision = current_decision(state)
        if decision is None:
            break
        state = apply_action(state, decision.legal_actions[0]).state
    decision = current_decision(state)
    if decision is not None:
        position = build_current_value_position(state, decision)
    else:
        viewer = (
            state.terminal_result.winners[0]
            if state.terminal_result and state.terminal_result.winners
            else PlayerId.PLAYER_1
        )
        position = build_value_position(
            state,
            viewer,
            None,
            position_kind=ValuePositionKind.AFTERSTATE,
        )
    encoder = FlatObservationEncoder()
    print(
        json.dumps(
            {
                "dimension": encoder.manifest.input_dimension,
                "fingerprint": encoder.manifest.layout_fingerprint,
                "nonzero": encoder.inspect_nonzero(position),
            },
            sort_keys=True,
        )
    )
    return 0


def _balanced_split_salt(source_paths: tuple[Path, ...], validation_fraction: float) -> str:
    """Find a deterministic split salt that gives a nonempty train and validation set."""

    from innovation_ai.training.compact_replay import (
        read_compact_replay_shard,
        setup_provenance_digest,
    )
    from innovation_ai.training.dataset import DatasetSplit, split_for_setup_provenance

    digests = tuple(
        setup_provenance_digest(episode.setup)
        for path in source_paths
        for episode in read_compact_replay_shard(path, verify=True)
    )
    for index in range(10_000):
        salt = f"innovation-ai-iterate-split-v1-{index}"
        splits = {
            split_for_setup_provenance(
                digest,
                validation_fraction=validation_fraction,
                split_salt=salt,
            )
            for digest in digests
        }
        if splits == {DatasetSplit.TRAIN, DatasetSplit.VALIDATION}:
            return salt
    raise ValueError("could not derive a nonempty deterministic train/validation split")


def _iterate(args: argparse.Namespace) -> int:
    """Run a complete bootstrap, train, learned-self-play, and candidate-train iteration."""

    from innovation_ai.agents.descriptors import (
        RANDOM_AGENT_DESCRIPTOR,
        SIMPLE_HEURISTIC_AGENT_DESCRIPTOR,
    )
    from innovation_ai.training.checkpoint import (
        PolicyDescriptor,
        load_checkpoint_manifest,
    )
    from innovation_ai.training.dataset import materialize_dataset
    from innovation_ai.training.optimize import TrainingConfig, train_terminal_outcomes
    from innovation_ai.training.self_play import (
        GenerationConfig,
        SeatPolicy,
        plan_generation,
        run_generation,
    )

    config_data: dict[str, Any] = {}
    if args.config is not None:
        with args.config.open("rb") as stream:
            loaded = tomllib.load(stream)
        config_data = dict(loaded.get("iterate", loaded))
    root = Path(config_data.get("run_dir", args.run_dir))
    root.mkdir(parents=True, exist_ok=True)
    seed = int(config_data.get("seed", args.seed))
    bootstrap_games = int(config_data.get("bootstrap_games", args.bootstrap_games))
    learned_games = int(config_data.get("learned_games", args.learned_games))
    games_in_flight = int(config_data.get("games_in_flight", args.games_in_flight))
    shard_size = int(config_data.get("shard_size", args.shard_size))
    validation_fraction = float(config_data.get("validation_fraction", args.validation_fraction))
    train_config = TrainingConfig(
        seed=seed,
        max_epochs=int(config_data.get("epochs", args.epochs)),
        patience=int(config_data.get("patience", args.patience)),
        batch_size=int(config_data.get("batch_size", args.batch_size)),
        torch_num_threads=int(config_data.get("torch_threads", args.torch_threads)),
    )
    checkpoint_root = root / "checkpoints"
    policy_root = root / "policies"
    policy_root.mkdir(exist_ok=True)

    heuristic = SeatPolicy(
        SIMPLE_HEURISTIC_AGENT_DESCRIPTOR.descriptor_id,
        "baseline",
        descriptor=SIMPLE_HEURISTIC_AGENT_DESCRIPTOR,
    )
    random = SeatPolicy(
        RANDOM_AGENT_DESCRIPTOR.descriptor_id,
        "baseline",
        descriptor=RANDOM_AGENT_DESCRIPTOR,
    )
    bootstrap_manifest = plan_generation(
        GenerationConfig(
            f"{root.name}-bootstrap",
            seed,
            0,
            max_games_in_flight=games_in_flight,
            shard_episode_limit=shard_size,
            action_ceiling=args.max_actions,
        ),
        (heuristic, random),
        (
            (heuristic.policy_id, random.policy_id),
            (random.policy_id, heuristic.policy_id),
        ),
        bootstrap_games,
    )
    bootstrap_dir = root / "bootstrap"
    run_generation(bootstrap_dir, bootstrap_manifest)
    bootstrap_sources = _replay_sources(bootstrap_dir)
    bootstrap_dataset_dir = bootstrap_dir / "dataset"
    bootstrap_dataset = materialize_dataset(
        bootstrap_sources,
        bootstrap_dataset_dir,
        validation_fraction=validation_fraction,
        split_salt=_balanced_split_salt(bootstrap_sources, validation_fraction),
    )
    first = train_terminal_outcomes(
        bootstrap_dataset_dir / "manifest.json",
        checkpoint_root,
        config=train_config,
        generation=0,
        creation_command="innovation-ai iterate bootstrap",
    )
    first_manifest = load_checkpoint_manifest(first.checkpoint_directory)
    first_policy = PolicyDescriptor.from_checkpoint(
        first_manifest,
        temperature=args.temperature,
        determinization_count=args.determinizations,
    )
    first_policy.save(policy_root / f"{first_policy.policy_id}.json")

    learned_seat = SeatPolicy(first_policy.policy_id, "learned", learned=first_policy)
    learned_manifest = plan_generation(
        GenerationConfig(
            f"{root.name}-learned",
            seed + 1,
            1,
            max_games_in_flight=games_in_flight,
            shard_episode_limit=shard_size,
            action_ceiling=args.max_actions,
        ),
        (learned_seat,),
        ((learned_seat.policy_id, learned_seat.policy_id),),
        learned_games,
    )
    learned_dir = root / "learned"
    run_generation(learned_dir, learned_manifest, checkpoint_root=checkpoint_root)
    learned_sources = _replay_sources(learned_dir)
    learned_dataset_dir = learned_dir / "dataset"
    learned_dataset = materialize_dataset(
        learned_sources,
        learned_dataset_dir,
        validation_fraction=validation_fraction,
        split_salt=_balanced_split_salt(learned_sources, validation_fraction),
    )
    candidate = train_terminal_outcomes(
        learned_dataset_dir / "manifest.json",
        checkpoint_root,
        config=replace(train_config, seed=seed + 1),
        parent_checkpoint_ids=(first_manifest.checkpoint_id,),
        generation=1,
        creation_command="innovation-ai iterate candidate",
    )
    candidate_manifest = load_checkpoint_manifest(candidate.checkpoint_directory)
    candidate_policy = PolicyDescriptor.from_checkpoint(
        candidate_manifest,
        temperature=args.temperature,
        determinization_count=args.determinizations,
    )
    candidate_policy.save(policy_root / f"{candidate_policy.policy_id}.json")
    summary = {
        "bootstrap_dataset": bootstrap_dataset.counts.example_count,
        "learned_dataset": learned_dataset.counts.example_count,
        "bootstrap_policy_id": first_policy.policy_id,
        "candidate_policy_id": candidate_policy.policy_id,
        "candidate_checkpoint_id": candidate_manifest.checkpoint_id,
    }
    (root / "iteration-summary.json").write_text(
        json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


def _arena(args: argparse.Namespace) -> int:
    """Run a deterministic paired arena and write raw plus summarized artifacts."""

    from innovation_ai.agents.descriptors import (
        RANDOM_AGENT_DESCRIPTOR,
        SIMPLE_HEURISTIC_AGENT_DESCRIPTOR,
    )
    from innovation_ai.harness.arena import (
        ArenaManifest,
        BootstrapConfig,
        PolicyPool,
        PoolEntry,
        dumps_arena_manifest,
        plan_match_pair,
        render_arena_report_table,
    )
    from innovation_ai.harness.arena import (
        PolicyDescriptor as ArenaPolicyDescriptor,
    )
    from innovation_ai.harness.arena_runner import (
        ArenaRunner,
        loads_champion_manifest,
        loads_champion_pointer,
        promotion_outcome,
        write_champion,
        write_execution_artifacts,
    )
    from innovation_ai.harness.engine import InnovationEngineAdapter
    from innovation_ai.training.checkpoint import load_policy_descriptor
    from innovation_ai.training.inference import FrozenEvaluatorCache

    candidate = replace(
        load_policy_descriptor(args.candidate_policy),
        temperature=0.0,
        determinization_count=args.determinizations,
    )
    policies: dict[str, ArenaPolicyDescriptor] = {
        candidate.policy_id: ArenaPolicyDescriptor(
            candidate.policy_id, "learned", candidate.checkpoint_id
        )
    }
    learned = {candidate.policy_id: candidate}
    opponent_ids: list[str] = []
    for name in (item.strip() for item in args.opponents.split(",") if item.strip()):
        if name == "heuristic":
            policy_id = SIMPLE_HEURISTIC_AGENT_DESCRIPTOR.descriptor_id
            policies[policy_id] = ArenaPolicyDescriptor(policy_id, "heuristic")
        elif name == "random":
            policy_id = RANDOM_AGENT_DESCRIPTOR.descriptor_id
            policies[policy_id] = ArenaPolicyDescriptor(policy_id, "random")
        else:
            raise ValueError(f"unknown arena opponent {name!r}")
        opponent_ids.append(policy_id)
    for path in args.opponent_policy:
        descriptor = replace(
            load_policy_descriptor(path),
            temperature=0.0,
            determinization_count=args.determinizations,
        )
        learned[descriptor.policy_id] = descriptor
        policies[descriptor.policy_id] = ArenaPolicyDescriptor(
            descriptor.policy_id, "learned", descriptor.checkpoint_id
        )
        opponent_ids.append(descriptor.policy_id)
    if not opponent_ids:
        raise ValueError("arena requires at least one opponent")
    pool = PolicyPool(
        f"{args.output.name}-opponents",
        tuple(PoolEntry(policy_id, 1) for policy_id in dict.fromkeys(opponent_ids)),
    )
    pairs = tuple(
        plan_match_pair(
            f"{opponent_index:03d}-{pair_index:06d}",
            args.seed_start + pair_index,
            candidate.policy_id,
            opponent_id,
        )
        for opponent_index, opponent_id in enumerate(opponent_ids)
        for pair_index in range(args.seed_pairs)
    )
    manifest = ArenaManifest(
        args.output.name,
        candidate.policy_id,
        pool,
        pairs,
        BootstrapConfig(seed=args.bootstrap_seed),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    resolved_policy_dir = args.output / "policies"
    resolved_policy_dir.mkdir(exist_ok=True)
    for descriptor in learned.values():
        descriptor.save(resolved_policy_dir / f"{descriptor.policy_id}.json")
    (args.output / "arena-manifest.json").write_text(
        f"{dumps_arena_manifest(manifest)}\n", encoding="utf-8"
    )
    cache = FrozenEvaluatorCache(args.checkpoint_root)
    execution = ArenaRunner(
        InnovationEngineAdapter(),
        policies,
        learned_policies=learned,
        evaluator_cache=cache,
        max_actions=args.max_actions,
        run_seed=args.seed_start,
    ).execute(manifest)
    write_execution_artifacts(args.output, execution)
    table = render_arena_report_table(execution.report)
    (args.output / "arena-report.md").write_text(table, encoding="utf-8")
    if args.promote:
        pointer_path = args.champion_dir / "champion.json"
        incumbent = None
        if pointer_path.exists():
            pointer = loads_champion_pointer(pointer_path.read_text(encoding="utf-8"))
            incumbent_path = args.champion_dir / "champions" / f"{pointer.manifest_sha256[7:]}.json"
            incumbent = loads_champion_manifest(incumbent_path.read_text(encoding="utf-8"))
        outcome = promotion_outcome(
            incumbent, policies[candidate.policy_id], manifest, execution.report
        )
        if outcome.decision.value != "retained":
            write_champion(args.champion_dir, outcome.champion)
        print(f"promotion: {outcome.decision.value}")
    print(table, end="")
    return 0


def _profile(args: argparse.Namespace) -> int:
    """Run representative CPU sweeps and write deterministic profile reports."""

    import tempfile

    if args.config is not None:
        with args.config.open("rb") as stream:
            loaded_profile = tomllib.load(stream).get("profile", {})
        for key in (
            "seed",
            "games",
            "max_actions",
            "warmup",
            "samples",
            "batch_sizes",
            "torch_threads",
            "games_in_flight",
            "determinizations",
            "full",
        ):
            if key in loaded_profile:
                setattr(args, key, loaded_profile[key])

    import torch
    from torch.optim import AdamW

    from innovation_ai.harness.afterstates import TrustedCandidateExpander
    from innovation_ai.harness.engine import InnovationEngineAdapter
    from innovation_ai.harness.policy import build_current_value_position
    from innovation_ai.innovation.protocol import apply_action, current_decision
    from innovation_ai.innovation.state import build_setup_state
    from innovation_ai.training.determinizations import (
        InformationSetSampler,
        InformationSetSpecBuilder,
    )
    from innovation_ai.training.encoding import FlatObservationEncoder
    from innovation_ai.training.model import ValueNetwork
    from innovation_ai.training.profiling import (
        IntegrityConfig,
        ProfileConfig,
        ScenarioWork,
        afterstate_expansion_scenario,
        arena_scenario,
        determinization_scenario,
        dumps_profile_report,
        encoding_scenario,
        engine_baseline_play_scenario,
        inference_scenario,
        replay_extraction_scenario,
        run_profile,
        self_play_scenario,
        training_scenario,
    )

    state = build_setup_state(args.seed)
    for _ in range(2):
        decision = current_decision(state)
        assert decision is not None
        state = apply_action(state, decision.legal_actions[0]).state
    decision = current_decision(state)
    assert decision is not None
    position = build_current_value_position(state, decision)
    encoder = FlatObservationEncoder()
    spec = InformationSetSpecBuilder().build(state)
    model = ValueNetwork(encoder.manifest.input_dimension)

    def encode_work(invocation: Any) -> ScenarioWork:
        encoder.encode_batch(tuple(position for _ in range(invocation.batch_size)))
        return ScenarioWork(invocation.batch_size, "positions")

    def inference_work(invocation: Any) -> ScenarioWork:
        features = torch.from_numpy(
            encoder.encode_batch(tuple(position for _ in range(invocation.batch_size)))
        )
        model.eval()
        with torch.inference_mode():
            model(features)
        return ScenarioWork(invocation.batch_size, "positions")

    def training_work(invocation: Any) -> ScenarioWork:
        local = ValueNetwork(encoder.manifest.input_dimension)
        optimizer = AdamW(local.parameters(), lr=1e-3, weight_decay=1e-5)
        features = torch.from_numpy(
            encoder.encode_batch(tuple(position for _ in range(invocation.batch_size)))
        )
        targets = torch.full((invocation.batch_size,), 0.5, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            local.forward_logits(features), targets
        )
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        return ScenarioWork(invocation.batch_size, "examples")

    def determinization_work(invocation: Any) -> ScenarioWork:
        samples = InformationSetSampler(seed=args.seed).sample_many(
            spec, invocation.determinizations
        )
        return ScenarioWork(len(samples), "samples")

    def afterstate_work(invocation: Any) -> ScenarioWork:
        samples = InformationSetSampler(seed=args.seed).sample_many(
            spec, invocation.determinizations
        )
        safe = tuple(sample for sample in samples if sample is not None)
        expansion = TrustedCandidateExpander().expand(
            spec,
            safe,
            game_id="profile-afterstate",
            evaluator_key="profile",
        )
        return ScenarioWork(len(expansion.all_routes), "candidates")

    config = ProfileConfig(
        args.output.name,
        tuple(
            sys.argv
            if args.config is None
            else ("innovation-ai", "profile", "--config", str(args.config))
        ),
        warmup_samples=args.warmup,
        timed_samples=args.samples,
        batch_sizes=tuple(args.batch_sizes),
        torch_num_threads=tuple(args.torch_threads),
        games_in_flight=tuple(args.games_in_flight),
        determinizations=tuple(args.determinizations),
        integrity=IntegrityConfig(correctness_spot_checks=False),
    )
    scenarios: tuple[Any, ...] = (
        engine_baseline_play_scenario(
            "engine-only",
            InnovationEngineAdapter(),
            games=args.games,
            run_seed=args.seed,
            max_actions_per_game=args.max_actions,
        ),
        encoding_scenario("encoding", encode_work),
        inference_scenario("inference", inference_work),
        training_scenario("training", training_work),
        determinization_scenario("determinization", determinization_work),
        afterstate_expansion_scenario("afterstate", afterstate_work),
    )
    if args.full:
        from innovation_ai.agents.descriptors import (
            RANDOM_AGENT_DESCRIPTOR,
            SIMPLE_HEURISTIC_AGENT_DESCRIPTOR,
        )
        from innovation_ai.harness.arena import (
            ArenaManifest,
            BootstrapConfig,
            PolicyPool,
            PoolEntry,
            plan_match_pair,
        )
        from innovation_ai.harness.arena import (
            PolicyDescriptor as ArenaPolicyDescriptor,
        )
        from innovation_ai.harness.arena_runner import ArenaRunner
        from innovation_ai.innovation.state import build_setup_state
        from innovation_ai.innovation.types import PlayerId
        from innovation_ai.training.compact_replay import (
            CompactReplayProvenance,
            CompactReplayRecorder,
            DeterminizationProvenance,
            ExplorationProvenance,
            SeatPolicyProvenance,
            read_compact_replay_shard,
            sha256_digest,
        )
        from innovation_ai.training.dataset import extract_value_position_examples
        from innovation_ai.training.self_play import (
            GenerationConfig,
            SeatPolicy,
            plan_generation,
            run_generation,
        )

        provenance = CompactReplayProvenance(
            "profile-replay",
            sha256_digest("profile-replay-config"),
            0,
            (
                SeatPolicyProvenance(
                    PlayerId.PLAYER_1,
                    SIMPLE_HEURISTIC_AGENT_DESCRIPTOR.descriptor_id,
                    None,
                    "deterministic-v1",
                ),
                SeatPolicyProvenance(
                    PlayerId.PLAYER_2,
                    RANDOM_AGENT_DESCRIPTOR.descriptor_id,
                    None,
                    "python-mt19937-randrange-v1",
                ),
            ),
            ExplorationProvenance("temperature-softmax-v1", 0.0, "sha256-domain-separated-v1"),
            DeterminizationProvenance(
                "information-set-sampler-v1", "sha256-counter-v1", 0, None, True
            ),
        )
        replay_recorder = CompactReplayRecorder(
            build_setup_state(args.seed), "profile-episode", provenance
        )
        for _ in range(args.max_actions):
            pending = replay_recorder.decisions()
            if not pending:
                break
            replay_recorder.submit(pending[0].legal_actions[0])
        replay_episode = replay_recorder.episode()

        def replay_work(invocation: Any) -> ScenarioWork:
            del invocation
            examples = extract_value_position_examples(replay_episode)
            return ScenarioWork(len(examples), "examples")

        heuristic_seat = SeatPolicy(
            SIMPLE_HEURISTIC_AGENT_DESCRIPTOR.descriptor_id,
            "baseline",
            descriptor=SIMPLE_HEURISTIC_AGENT_DESCRIPTOR,
        )
        random_seat = SeatPolicy(
            RANDOM_AGENT_DESCRIPTOR.descriptor_id,
            "baseline",
            descriptor=RANDOM_AGENT_DESCRIPTOR,
        )

        def self_play_work(invocation: Any) -> ScenarioWork:
            with tempfile.TemporaryDirectory() as directory:
                manifest = plan_generation(
                    GenerationConfig(
                        "profile-self-play",
                        1001,
                        0,
                        max_games_in_flight=invocation.games_in_flight,
                        shard_episode_limit=1,
                        action_ceiling=args.max_actions,
                    ),
                    (heuristic_seat, random_seat),
                    ((heuristic_seat.policy_id, random_seat.policy_id),),
                    1,
                )
                run_generation(directory, manifest)
                episodes = read_compact_replay_shard(
                    Path(directory) / "replays" / "shard-00000.jsonl.gz"
                )
                return ScenarioWork(len(episodes[0].actions), "actions")

        candidate_id = SIMPLE_HEURISTIC_AGENT_DESCRIPTOR.descriptor_id
        opponent_id = RANDOM_AGENT_DESCRIPTOR.descriptor_id
        arena_manifest = ArenaManifest(
            "profile-arena",
            candidate_id,
            PolicyPool("profile-pool", (PoolEntry(opponent_id, 1),)),
            (plan_match_pair("profile-pair", 1, candidate_id, opponent_id),),
            BootstrapConfig(seed=1, resamples=100),
        )
        arena_policies = {
            candidate_id: ArenaPolicyDescriptor(candidate_id, "heuristic"),
            opponent_id: ArenaPolicyDescriptor(opponent_id, "random"),
        }

        def arena_work(invocation: Any) -> ScenarioWork:
            execution = ArenaRunner(
                InnovationEngineAdapter(),
                arena_policies,
                max_actions=args.max_actions,
                run_seed=invocation.sample_index,
            ).execute(arena_manifest)
            return ScenarioWork(sum(game.game_length for game in execution.result.games), "actions")

        scenarios = (
            *scenarios,
            replay_extraction_scenario("replay-extraction", replay_work),
            self_play_scenario("self-play", self_play_work),
            arena_scenario("arena", arena_work),
        )
    report = run_profile(config, scenarios)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "profile.json").write_text(f"{dumps_profile_report(report)}\n", encoding="utf-8")
    (args.output / "profile.md").write_text(report.to_markdown(), encoding="utf-8")
    print(report.to_markdown(), end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the project command-line parser."""

    parser = argparse.ArgumentParser(prog="innovation-ai")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="verify the local development environment")

    play = subparsers.add_parser("play", help="play a deterministic baseline game and save a log")
    play.add_argument("--seed", type=int, default=0, help="explicit setup seed (default: 0)")
    play.add_argument("--log", type=Path, required=True, help="output game-log path")
    play.add_argument(
        "--max-transitions",
        type=int,
        default=1000,
        help="safety ceiling before refusing to write an incomplete log",
    )
    replay = subparsers.add_parser("replay", help="verify and replay a saved game log")
    replay.add_argument("log", type=Path, help="game-log path")

    baseline = subparsers.add_parser("baseline", help="benchmark random/heuristic baselines")
    baseline.add_argument("--output", type=Path, required=True)
    baseline.add_argument("--seed", type=int, default=1000)
    baseline.add_argument("--games", type=int, default=32)
    baseline.add_argument("--games-in-flight", type=int, default=32)
    baseline.add_argument("--max-actions", type=int, default=10_000)
    baseline.add_argument(
        "--validation", nargs="+", default=("full", "cheap"), choices=("full", "cheap", "off")
    )

    self_play = subparsers.add_parser("self-play", help="generate compact replay self-play")
    self_play.add_argument("--run-dir", type=Path, required=True)
    self_play.add_argument("--games", type=int, required=True)
    self_play.add_argument("--seed", type=int, default=1000)
    self_play.add_argument("--generation", type=int, default=0)
    self_play.add_argument("--player-1", choices=("heuristic", "random"), default="heuristic")
    self_play.add_argument("--player-2", choices=("heuristic", "random"), default="random")
    self_play.add_argument("--policy", type=Path)
    self_play.add_argument("--checkpoint-root", type=Path)
    self_play.add_argument("--temperature", type=float, default=0.15)
    self_play.add_argument("--determinizations", type=int, default=1)
    self_play.add_argument("--games-in-flight", type=int, default=32)
    self_play.add_argument("--shard-size", type=int, default=256)
    self_play.add_argument("--max-actions", type=int, default=10_000)
    self_play.add_argument("--validation", choices=("full", "cheap", "off"), default="cheap")

    dataset = subparsers.add_parser("dataset", help="compact replay dataset operations")
    dataset_subparsers = dataset.add_subparsers(dest="dataset_command", required=True)
    dataset_build = dataset_subparsers.add_parser("build", help="materialize encoder arrays")
    dataset_build.add_argument("--replays", type=Path, required=True)
    dataset_build.add_argument("--output", type=Path, required=True)
    dataset_build.add_argument("--validation-fraction", type=float, default=0.2)
    dataset_build.add_argument("--split-salt", default="innovation-ai-value-dataset-split-v1")
    dataset_build.add_argument("--episodes-per-shard", type=int, default=256)

    train = subparsers.add_parser("train-value", help="train and freeze a value checkpoint")
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--policy-output", type=Path)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--generation", type=int, default=0)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--patience", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=1024)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-5)
    train.add_argument("--torch-threads", type=int, default=1)
    train.add_argument("--temperature", type=float, default=0.15)
    train.add_argument("--determinizations", type=int, default=1)

    inspect = subparsers.add_parser("inspect-encoding", help="print named encoder features")
    inspect.add_argument("--seed", type=int, default=0)
    inspect.add_argument("--steps", type=int, default=2)

    iterate = subparsers.add_parser("iterate", help="run one complete learned iteration")
    iterate.add_argument("--config", type=Path)
    iterate.add_argument("--run-dir", type=Path, default=Path("artifacts/runs/iteration-smoke"))
    iterate.add_argument("--seed", type=int, default=1000)
    iterate.add_argument("--bootstrap-games", type=int, default=32)
    iterate.add_argument("--learned-games", type=int, default=16)
    iterate.add_argument("--games-in-flight", type=int, default=32)
    iterate.add_argument("--shard-size", type=int, default=256)
    iterate.add_argument("--max-actions", type=int, default=10_000)
    iterate.add_argument("--validation-fraction", type=float, default=0.2)
    iterate.add_argument("--epochs", type=int, default=20)
    iterate.add_argument("--patience", type=int, default=5)
    iterate.add_argument("--batch-size", type=int, default=1024)
    iterate.add_argument("--torch-threads", type=int, default=1)
    iterate.add_argument("--temperature", type=float, default=0.15)
    iterate.add_argument("--determinizations", type=int, default=1)

    arena = subparsers.add_parser("arena", help="run a paired seat-swapped arena")
    arena.add_argument("--candidate-policy", type=Path, required=True)
    arena.add_argument("--opponents", default="heuristic,random")
    arena.add_argument("--opponent-policy", type=Path, action="append", default=[])
    arena.add_argument("--checkpoint-root", type=Path, required=True)
    arena.add_argument("--seed-start", type=int, default=50_000)
    arena.add_argument("--seed-pairs", type=int, default=200)
    arena.add_argument("--determinizations", type=int, default=4)
    arena.add_argument("--bootstrap-seed", type=int, default=0)
    arena.add_argument("--max-actions", type=int, default=10_000)
    arena.add_argument("--output", type=Path, required=True)
    arena.add_argument("--promote", action="store_true")
    arena.add_argument("--champion-dir", type=Path, default=Path("artifacts/champion"))

    profile = subparsers.add_parser("profile", help="run CPU throughput sweeps")
    profile.add_argument("--config", type=Path)
    profile.add_argument("--output", type=Path, required=True)
    profile.add_argument("--seed", type=int, default=7000)
    profile.add_argument("--games", type=int, default=1)
    profile.add_argument("--max-actions", type=int, default=1000)
    profile.add_argument("--full", action="store_true")
    profile.add_argument("--warmup", type=int, default=1)
    profile.add_argument("--samples", type=int, default=3)
    profile.add_argument("--batch-sizes", type=int, nargs="+", default=(1, 8, 32))
    profile.add_argument("--torch-threads", type=int, nargs="+", default=(1, 2))
    profile.add_argument("--games-in-flight", type=int, nargs="+", default=(1, 8, 32))
    profile.add_argument("--determinizations", type=int, nargs="+", default=(1, 4))

    web = subparsers.add_parser("web", help="serve the hot-seat browser QA table")
    web.add_argument("--host", default="127.0.0.1", help="listen address (default: 127.0.0.1)")
    web.add_argument("--port", type=int, default=8000, help="listen port (default: 8000)")
    web.add_argument("--seed", type=int, default=0, help="initial deterministic setup seed")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line application."""
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        return _doctor()
    try:
        if args.command == "play":
            return _play(args.seed, args.log, args.max_transitions)
        if args.command == "replay":
            return _replay(args.log)
        if args.command == "baseline":
            return _baseline(args)
        if args.command == "self-play":
            return _self_play(args)
        if args.command == "dataset" and args.dataset_command == "build":
            return _dataset_build(args)
        if args.command == "train-value":
            return _train_value(args)
        if args.command == "inspect-encoding":
            return _inspect_encoding(args)
        if args.command == "iterate":
            return _iterate(args)
        if args.command == "arena":
            return _arena(args)
        if args.command == "profile":
            return _profile(args)
        if args.command == "web":
            if not 1 <= args.port <= 65535:
                raise ValueError("port must be between 1 and 65535")
            from innovation_ai.web.server import serve

            logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
            serve(args.host, args.port, args.seed)
            return 0
    except (GameLogError, ReplayError, OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    raise AssertionError(f"unhandled command: {args.command}")
