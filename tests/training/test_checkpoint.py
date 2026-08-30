from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from innovation_ai.training.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointIntegrityError,
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
