from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from innovation_ai.innovation.state import LEGACY_INFORMATION_POLICY_VERSION
from innovation_ai.search.contracts import (
    EXPERIMENTAL_ROLLOUT_SEARCH_DESCRIPTOR,
    MINIMAX_SEARCH_DESCRIPTOR_KIND,
    PRODUCTION_SEARCH_DESCRIPTOR,
    ROLLOUT_SEARCH_DESCRIPTOR_KIND,
)
from innovation_ai.training.checkpoint import (
    DEFAULT_FALLBACK_AGENT,
    DEFAULT_LEARNED_TURN_ACTION_POLICY,
    DEFAULT_ROLLOUT_FALLBACK_AGENT,
    DEFAULT_SEARCH_CONTINUATION_POLICY,
    GENERIC_POLICY_DESCRIPTOR_SCHEMA_VERSION,
    MINIMAX_POLICY_DESCRIPTOR_SCHEMA_VERSION,
    POLICY_DESCRIPTOR_SCHEMA_VERSION,
    CheckpointCompatibilityError,
    CheckpointIntegrityError,
    CheckpointManifest,
    PolicyCompatibilityError,
    PolicyDescriptor,
    assert_policy_compatible,
    load_checkpoint,
    load_policy_descriptor,
    save_checkpoint,
)
from innovation_ai.training.encoding import build_encoder_manifest
from innovation_ai.training.model import ValueNetwork

LEGACY_POLICY_PAYLOAD: dict[str, object] = {
    "action_schema_version": 1,
    "afterstate_boundary_semantics_version": "immediate-one-transition-v1",
    "card_data_fingerprint": (
        "sha256:23fc84b70f401bba3e8d0abaaad0c28978cdaf076aef45e9e0cfc4b5810d5e86"
    ),
    "checkpoint_id": "sha256:293de9146420a204c460683a4af2f3ec4a069fadc8617ba52d0d19c3801c9c20",
    "decision_schema_version": 3,
    "determinization_count": 1,
    "effects_fingerprint": (
        "sha256:8104ec4477ff3c11343e72b4dd4cfbac4330f4cee97d40383d44c8992c65e237"
    ),
    "encoder_layout_fingerprint": (
        "sha256:b472a8911f444bcf7920ff89fab2ff55aa23f054747242b538c32a7851c5b2b5"
    ),
    "engine_version": "0.2.0",
    "fallback_agent": "simple-heuristic",
    "fallback_agent_version": "v1",
    "information_policy_version": "rulebook-private-covered-v1",
    "information_set_sampler_version": "information-set-sampler-v1",
    "observation_schema_version": 3,
    "policy_id": "sha256:0c3f5e8e263c7aae5011deda9fcd269eff06090ec3bce9116022db82a5491d00",
    "public_boundary_schema_version": 1,
    "rules_version": "innovation-base-third-edition-2p-v1",
    "sampler_rng_version": "sha256-counter-v1",
    "schema_version": 1,
    "selector_rng_version": "sha256-domain-separated-v1",
    "selector_version": "temperature-softmax-v1",
    "temperature": 0.15,
    "value_position_schema_version": 1,
}
LEGACY_POLICY_JSON = json.dumps(
    LEGACY_POLICY_PAYLOAD, sort_keys=True, separators=(",", ":"), ensure_ascii=True
)

HISTORICAL_POLICY_V2_JSON = (
    '{"action_schema_version":1,'
    '"afterstate_boundary_semantics_version":"immediate-one-transition-v1",'
    '"card_data_fingerprint":"sha256:3333333333333333333333333333333333333333333333333333333333333333",'
    '"checkpoint_id":"sha256:1111111111111111111111111111111111111111111111111111111111111111",'
    '"decision_schema_version":3,"determinization_count":1,'
    '"effects_fingerprint":"sha256:4444444444444444444444444444444444444444444444444444444444444444",'
    '"encoder_layout_fingerprint":"sha256:2222222222222222222222222222222222222222222222222222222222222222",'
    '"engine_version":"0.2.0","fallback_agent":"sampled-minimax-heuristic",'
    '"fallback_agent_version":"v1","information_policy_version":"public-covered-v1",'
    '"information_set_sampler_version":"information-set-sampler-v1",'
    '"learned_turn_action_policy":"sampled-afterstate-value-v1",'
    '"observation_schema_version":3,'
    '"policy_id":"sha256:44c8acc62e7e094eae93c4710883e940c0c67725ac5ef0dd50e554b861e87d28",'
    '"public_boundary_schema_version":1,'
    '"rules_version":"innovation-base-third-edition-2p-v1",'
    '"sampler_rng_version":"sha256-counter-v1","schema_version":2,'
    '"search_continuation_policy":"hand-engineered-minimax-both-players-v1",'
    '"search_descriptor_id":"sha256:a3de2116646c297b30b75e25fc4ab065ef1a0674f89b932eecd9ef5eea2b067e",'
    '"selector_rng_version":"sha256-domain-separated-v1",'
    '"selector_version":"temperature-softmax-v1","temperature":0.15,'
    '"value_position_schema_version":1}'
)


def _model(input_dimension: int, output_bias: float = 0.0) -> ValueNetwork:
    model = ValueNetwork(input_dimension)
    with torch.no_grad():
        model.output.bias.fill_(output_bias)
    return model


def _experimental_rollout_policy(
    manifest: CheckpointManifest,
    *,
    temperature: float = 0.0,
) -> PolicyDescriptor:
    """Build parseable schema-v3 metadata without claiming checkpoint-lineage approval."""

    minimax = PolicyDescriptor.from_checkpoint(manifest, temperature=temperature)
    return replace(
        minimax,
        determinization_count=EXPERIMENTAL_ROLLOUT_SEARCH_DESCRIPTOR.determinization_count,
        fallback_agent=DEFAULT_ROLLOUT_FALLBACK_AGENT,
        fallback_algorithm_kind=ROLLOUT_SEARCH_DESCRIPTOR_KIND,
        fallback_algorithm_descriptor_id=(EXPERIMENTAL_ROLLOUT_SEARCH_DESCRIPTOR.descriptor_id),
        hypothetical_continuation_policy=(
            EXPERIMENTAL_ROLLOUT_SEARCH_DESCRIPTOR.continuation_agent_descriptor_id
        ),
    )


def test_checkpoint_round_trip_is_prediction_identical_and_state_dict_only(tmp_path: Path) -> None:
    encoder = build_encoder_manifest()
    model = _model(encoder.input_dimension, 0.25)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    features = torch.zeros((2, encoder.input_dimension), dtype=torch.float32)
    expected = model.predict(features).detach().clone()

    directory = save_checkpoint(
        tmp_path,
        model,
        encoder,
        optimizer=optimizer,
        metrics={"validation_brier": 0.125},
        training_dataset_ids=("dataset-a",),
        generation=3,
        optimizer_config={"name": "AdamW", "learning_rate": 0.001},
        creation_command="innovation-ai train-value",
    )
    loaded = load_checkpoint(directory, load_optimizer=True)

    assert directory.name == loaded.manifest.checkpoint_id
    assert {path.name for path in directory.iterdir()} == {
        "manifest.json",
        "model.pt",
        "optimizer.pt",
        "metrics.json",
    }
    torch.testing.assert_close(loaded.model.predict(features), expected, rtol=0.0, atol=0.0)
    assert loaded.metrics == {"validation_brier": 0.125}
    assert loaded.optimizer_state is not None
    assert all(parameter.device.type == "cpu" for parameter in loaded.model.parameters())
    with pytest.raises(FileExistsError, match="immutable checkpoint"):
        save_checkpoint(
            tmp_path,
            model,
            encoder,
            optimizer=optimizer,
            metrics={"validation_brier": 0.125},
            training_dataset_ids=("dataset-a",),
            generation=3,
            optimizer_config={"name": "AdamW", "learning_rate": 0.001},
            creation_command="innovation-ai train-value",
        )


def test_checkpoint_rejects_tampering_and_encoder_compatibility_mismatch(tmp_path: Path) -> None:
    encoder = build_encoder_manifest()
    directory = save_checkpoint(tmp_path, _model(encoder.input_dimension), encoder)

    with pytest.raises(CheckpointCompatibilityError, match="encoder_layout_fingerprint"):
        load_checkpoint(
            directory,
            encoder_manifest=replace(
                encoder,
                card_data_fingerprint="sha256:" + "0" * 64,
                layout_fingerprint="",
            ),
        )

    (directory / "model.pt").write_bytes(b"tampered")
    with pytest.raises(CheckpointIntegrityError, match="digest mismatch"):
        load_checkpoint(directory)


def test_legacy_checkpoint_loads_implicitly_but_rejects_an_explicit_public_encoder(
    tmp_path: Path,
) -> None:
    legacy_encoder = build_encoder_manifest(
        information_policy_version=LEGACY_INFORMATION_POLICY_VERSION
    )
    directory = save_checkpoint(tmp_path, _model(legacy_encoder.input_dimension), legacy_encoder)

    loaded = load_checkpoint(directory)
    assert loaded.manifest.information_policy_version == LEGACY_INFORMATION_POLICY_VERSION
    with pytest.raises(PolicyCompatibilityError, match="public-covered-v1"):
        PolicyDescriptor.from_checkpoint(loaded.manifest)
    with pytest.raises(CheckpointCompatibilityError, match="encoder_layout_fingerprint"):
        load_checkpoint(directory, encoder_manifest=build_encoder_manifest())


def test_policy_descriptor_identity_is_content_derived_and_checkpoint_compatible(
    tmp_path: Path,
) -> None:
    encoder = build_encoder_manifest()
    directory = save_checkpoint(tmp_path, _model(encoder.input_dimension), encoder)
    manifest = load_checkpoint(directory).manifest
    cold = PolicyDescriptor.from_checkpoint(manifest, temperature=0.0)
    repetition_aware = replace(cold, selector_version="recent-paid-action-penalty-v1")
    warm = PolicyDescriptor.from_checkpoint(manifest, temperature=0.2, determinization_count=4)
    descriptor_path = tmp_path / "policy.json"
    cold.save(descriptor_path)

    assert cold.policy_id != warm.policy_id
    assert cold.policy_id != repetition_aware.policy_id
    assert repetition_aware.checkpoint_id == cold.checkpoint_id
    assert load_policy_descriptor(descriptor_path) == cold
    assert_policy_compatible(cold, manifest)
    with pytest.raises(CheckpointCompatibilityError, match="encoder_layout_fingerprint"):
        assert_policy_compatible(
            replace(cold, encoder_layout_fingerprint="sha256:" + "f" * 64),
            manifest,
        )


def test_real_pilot_001_schema_v1_policy_preserves_payload_and_identity(tmp_path: Path) -> None:
    path = tmp_path / "legacy-policy.json"
    path.write_text(LEGACY_POLICY_JSON, encoding="utf-8")
    original = LEGACY_POLICY_JSON
    descriptor = load_policy_descriptor(path)

    assert descriptor.schema_version == 1
    assert descriptor.policy_id == (
        "sha256:0c3f5e8e263c7aae5011deda9fcd269eff06090ec3bce9116022db82a5491d00"
    )
    assert descriptor.search_descriptor_id is None
    assert descriptor.learned_turn_action_policy is None
    assert descriptor.search_continuation_policy is None
    assert descriptor.fallback_agent == "simple-heuristic"
    assert descriptor.dumps() == original
    assert "search_descriptor_id" not in descriptor.payload()


def test_schema_v2_historical_policy_preserves_payload_and_identity(tmp_path: Path) -> None:
    path = tmp_path / "historical-policy-v2.json"
    path.write_text(HISTORICAL_POLICY_V2_JSON, encoding="utf-8")

    descriptor = load_policy_descriptor(path)

    assert descriptor.schema_version == MINIMAX_POLICY_DESCRIPTOR_SCHEMA_VERSION
    assert descriptor.policy_id == (
        "sha256:44c8acc62e7e094eae93c4710883e940c0c67725ac5ef0dd50e554b861e87d28"
    )
    assert descriptor.search_descriptor_id == PRODUCTION_SEARCH_DESCRIPTOR.descriptor_id
    assert descriptor.fallback_algorithm_kind is None
    assert descriptor.fallback_algorithm_descriptor_id is None
    assert descriptor.hypothetical_continuation_policy is None
    assert descriptor.dumps() == HISTORICAL_POLICY_V2_JSON
    assert descriptor.payload() == json.loads(HISTORICAL_POLICY_V2_JSON)


def test_schema_v2_policy_round_trip_search_identity_and_tampering(tmp_path: Path) -> None:
    encoder = build_encoder_manifest()
    directory = save_checkpoint(tmp_path / "checkpoints", _model(encoder.input_dimension), encoder)
    manifest = load_checkpoint(directory).manifest
    descriptor = PolicyDescriptor.from_checkpoint(
        manifest,
        temperature=0.15,
        schema_version=MINIMAX_POLICY_DESCRIPTOR_SCHEMA_VERSION,
    )
    path = tmp_path / "policy-v2.json"
    descriptor.save(path)

    assert descriptor.schema_version == MINIMAX_POLICY_DESCRIPTOR_SCHEMA_VERSION
    assert descriptor.search_descriptor_id == PRODUCTION_SEARCH_DESCRIPTOR.descriptor_id
    assert descriptor.learned_turn_action_policy == DEFAULT_LEARNED_TURN_ACTION_POLICY
    assert descriptor.search_continuation_policy == DEFAULT_SEARCH_CONTINUATION_POLICY
    assert descriptor.fallback_agent == DEFAULT_FALLBACK_AGENT
    assert load_policy_descriptor(path) == descriptor
    assert_policy_compatible(descriptor, manifest)

    alternate_search_id = "sha256:" + "a" * 64
    alternate = PolicyDescriptor.from_checkpoint(
        manifest,
        temperature=0.15,
        search_descriptor_id=alternate_search_id,
    )
    assert alternate.search_descriptor_id == alternate_search_id
    assert alternate.policy_id != descriptor.policy_id

    tampered = descriptor.payload()
    tampered["search_descriptor_id"] = alternate_search_id
    with pytest.raises(PolicyCompatibilityError, match="content-derived ID"):
        PolicyDescriptor.from_payload(tampered)

    malformed = descriptor.payload()
    malformed["search_descriptor_id"] = "sha256:not-a-digest"
    with pytest.raises(PolicyCompatibilityError, match="tagged sha256"):
        PolicyDescriptor.from_payload(malformed)

    missing = descriptor.payload()
    del missing["search_descriptor_id"]
    with pytest.raises(PolicyCompatibilityError, match="fields differ from schema"):
        PolicyDescriptor.from_payload(missing)

    v1_with_v2_field = json.loads(LEGACY_POLICY_JSON)
    v1_with_v2_field["search_descriptor_id"] = PRODUCTION_SEARCH_DESCRIPTOR.descriptor_id
    with pytest.raises(PolicyCompatibilityError, match="fields differ from schema"):
        PolicyDescriptor.from_payload(v1_with_v2_field)


def test_current_schema_v3_uses_generic_minimax_default_and_explicit_rollout(
    tmp_path: Path,
) -> None:
    encoder = build_encoder_manifest()
    directory = save_checkpoint(tmp_path / "checkpoints", _model(encoder.input_dimension), encoder)
    manifest = load_checkpoint(directory).manifest

    historical_v2 = PolicyDescriptor.from_checkpoint(
        manifest,
        temperature=0.15,
        schema_version=MINIMAX_POLICY_DESCRIPTOR_SCHEMA_VERSION,
    )
    current_default = PolicyDescriptor.from_checkpoint(manifest, temperature=0.15)
    minimax = PolicyDescriptor.from_checkpoint(
        manifest,
        temperature=0.15,
        fallback_algorithm_kind=MINIMAX_SEARCH_DESCRIPTOR_KIND,
        fallback_algorithm_descriptor_id=PRODUCTION_SEARCH_DESCRIPTOR.descriptor_id,
        hypothetical_continuation_policy=DEFAULT_SEARCH_CONTINUATION_POLICY,
    )
    with pytest.raises(PolicyCompatibilityError, match="rollout-trained checkpoint lineage"):
        PolicyDescriptor.from_checkpoint(
            manifest,
            determinization_count=EXPERIMENTAL_ROLLOUT_SEARCH_DESCRIPTOR.determinization_count,
            fallback_agent=DEFAULT_ROLLOUT_FALLBACK_AGENT,
            temperature=0.15,
            fallback_algorithm_kind=ROLLOUT_SEARCH_DESCRIPTOR_KIND,
            fallback_algorithm_descriptor_id=(EXPERIMENTAL_ROLLOUT_SEARCH_DESCRIPTOR.descriptor_id),
            hypothetical_continuation_policy=(
                EXPERIMENTAL_ROLLOUT_SEARCH_DESCRIPTOR.continuation_agent_descriptor_id
            ),
        )
    rollout = _experimental_rollout_policy(manifest, temperature=0.15)

    assert POLICY_DESCRIPTOR_SCHEMA_VERSION == GENERIC_POLICY_DESCRIPTOR_SCHEMA_VERSION == 3
    assert historical_v2.schema_version == MINIMAX_POLICY_DESCRIPTOR_SCHEMA_VERSION
    assert current_default.schema_version == GENERIC_POLICY_DESCRIPTOR_SCHEMA_VERSION
    assert current_default == minimax
    assert minimax.schema_version == rollout.schema_version == POLICY_DESCRIPTOR_SCHEMA_VERSION
    assert minimax.policy_id != historical_v2.policy_id
    assert rollout.policy_id != minimax.policy_id
    assert PolicyDescriptor.from_payload(minimax.payload()) == minimax
    assert rollout.search_descriptor_id is None
    assert rollout.search_continuation_policy is None
    assert rollout.fallback_algorithm_kind == ROLLOUT_SEARCH_DESCRIPTOR_KIND
    assert (
        rollout.fallback_algorithm_descriptor_id
        == EXPERIMENTAL_ROLLOUT_SEARCH_DESCRIPTOR.descriptor_id
    )
    assert rollout.hypothetical_continuation_policy == (
        EXPERIMENTAL_ROLLOUT_SEARCH_DESCRIPTOR.continuation_agent_descriptor_id
    )
    assert "search_descriptor_id" not in rollout.payload()
    assert "search_continuation_policy" not in rollout.payload()
    rollout_path = tmp_path / "rollout-v3.json"
    rollout.save(rollout_path)
    assert load_policy_descriptor(rollout_path) == rollout
    with pytest.raises(PolicyCompatibilityError, match="rollout-trained checkpoint lineage"):
        assert_policy_compatible(rollout, manifest)


def test_schema_v3_rejects_tampering_and_algorithm_family_incompatibility(
    tmp_path: Path,
) -> None:
    encoder = build_encoder_manifest()
    directory = save_checkpoint(tmp_path / "checkpoints", _model(encoder.input_dimension), encoder)
    manifest = load_checkpoint(directory).manifest
    rollout = _experimental_rollout_policy(manifest)

    tampered = rollout.payload()
    tampered["fallback_algorithm_descriptor_id"] = "sha256:" + "a" * 64
    with pytest.raises(PolicyCompatibilityError, match="content-derived ID"):
        PolicyDescriptor.from_payload(tampered)

    malformed = rollout.payload()
    malformed["fallback_algorithm_descriptor_id"] = "sha256:not-a-digest"
    with pytest.raises(PolicyCompatibilityError, match="tagged sha256"):
        PolicyDescriptor.from_payload(malformed)

    with pytest.raises(PolicyCompatibilityError, match="agent and algorithm kind"):
        replace(rollout, fallback_algorithm_kind=MINIMAX_SEARCH_DESCRIPTOR_KIND)
    with pytest.raises(PolicyCompatibilityError, match="agent and algorithm kind"):
        replace(rollout, fallback_agent=DEFAULT_FALLBACK_AGENT)
    with pytest.raises(PolicyCompatibilityError, match="public-covered-v1"):
        replace(rollout, information_policy_version=LEGACY_INFORMATION_POLICY_VERSION)
    with pytest.raises(PolicyCompatibilityError, match="complete generic fallback fields"):
        PolicyDescriptor.from_checkpoint(
            manifest,
            fallback_algorithm_kind=ROLLOUT_SEARCH_DESCRIPTOR_KIND,
        )
    with pytest.raises(PolicyCompatibilityError, match="minimax-named search_descriptor_id"):
        PolicyDescriptor.from_checkpoint(
            manifest,
            fallback_agent=DEFAULT_ROLLOUT_FALLBACK_AGENT,
            fallback_algorithm_kind=ROLLOUT_SEARCH_DESCRIPTOR_KIND,
            fallback_algorithm_descriptor_id=(EXPERIMENTAL_ROLLOUT_SEARCH_DESCRIPTOR.descriptor_id),
            hypothetical_continuation_policy=(
                EXPERIMENTAL_ROLLOUT_SEARCH_DESCRIPTOR.continuation_agent_descriptor_id
            ),
            search_descriptor_id=PRODUCTION_SEARCH_DESCRIPTOR.descriptor_id,
        )

    incompatible = rollout.payload()
    incompatible["search_descriptor_id"] = PRODUCTION_SEARCH_DESCRIPTOR.descriptor_id
    with pytest.raises(PolicyCompatibilityError, match="fields differ from schema"):
        PolicyDescriptor.from_payload(incompatible)
