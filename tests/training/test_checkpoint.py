from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from innovation_ai.innovation.state import LEGACY_INFORMATION_POLICY_VERSION
from innovation_ai.search.contracts import PRODUCTION_SEARCH_DESCRIPTOR
from innovation_ai.training.checkpoint import (
    DEFAULT_FALLBACK_AGENT,
    DEFAULT_LEARNED_TURN_ACTION_POLICY,
    DEFAULT_SEARCH_CONTINUATION_POLICY,
    CheckpointCompatibilityError,
    CheckpointIntegrityError,
    PolicyCompatibilityError,
    PolicyDescriptor,
    assert_policy_compatible,
    load_checkpoint,
    load_policy_descriptor,
    save_checkpoint,
)
from innovation_ai.training.encoding import build_encoder_manifest
from innovation_ai.training.model import ValueNetwork


def _model(input_dimension: int, output_bias: float = 0.0) -> ValueNetwork:
    model = ValueNetwork(input_dimension)
    with torch.no_grad():
        model.output.bias.fill_(output_bias)
    return model


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


def test_real_pilot_001_schema_v1_policy_preserves_payload_and_identity() -> None:
    path = (
        Path(__file__).parents[2]
        / "artifacts/runs/pilot-001/policies"
        / "sha256:0c3f5e8e263c7aae5011deda9fcd269eff06090ec3bce9116022db82a5491d00.json"
    )
    original = path.read_text(encoding="utf-8").strip()
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


def test_schema_v2_policy_round_trip_search_identity_and_tampering(tmp_path: Path) -> None:
    encoder = build_encoder_manifest()
    directory = save_checkpoint(tmp_path / "checkpoints", _model(encoder.input_dimension), encoder)
    manifest = load_checkpoint(directory).manifest
    descriptor = PolicyDescriptor.from_checkpoint(manifest, temperature=0.15)
    path = tmp_path / "policy-v2.json"
    descriptor.save(path)

    assert descriptor.schema_version == 2
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

    v1_with_v2_field = json.loads(
        (
            Path(__file__).parents[2]
            / "artifacts/runs/pilot-001/policies"
            / "sha256:0c3f5e8e263c7aae5011deda9fcd269eff06090ec3bce9116022db82a5491d00.json"
        ).read_text(encoding="utf-8")
    )
    v1_with_v2_field["search_descriptor_id"] = PRODUCTION_SEARCH_DESCRIPTOR.descriptor_id
    with pytest.raises(PolicyCompatibilityError, match="fields differ from schema"):
        PolicyDescriptor.from_payload(v1_with_v2_field)
