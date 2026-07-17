"""Tests for Morpion training target and prediction scale diagnostics."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest
import torch

from chipiron.environments.morpion.bootstrap.evaluator_family import (
    entity_token_transformer_small_morpion_evaluator_spec,
)
from chipiron.environments.morpion.learning import (
    load_morpion_supervised_rows,
    save_morpion_supervised_rows_streaming,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.entity_tokens import (
    MORPION_ENTITY_TOKEN_MODEL_KIND,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.model import (
    MorpionRegressorArgs,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
    MorpionStreamingTrainingArgs,
    MorpionTrainingArgs,
    train_morpion_regressor_streaming,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training.scale_diagnostics import (
    tensor_scale_stats,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training.service import (
    prediction_scale_stats_for_cached_batches,
)
from chipiron.learning.supervised import TensorSupervisedBatch
from tests.environments.test_morpion_model_bundle import _build_rows_file

if TYPE_CHECKING:
    from pathlib import Path


class _TwoInputScaleRegressor(torch.nn.Module):
    """Small two-input model for cached scale diagnostic coverage."""

    def __init__(self) -> None:
        """Initialize deterministic projections and call observations."""
        super().__init__()
        self.primary_projection = torch.nn.Linear(3, 1)
        self.auxiliary_scale = torch.nn.Parameter(torch.tensor(1.0))
        self.received_primary: torch.Tensor | None = None
        self.received_auxiliary: torch.Tensor | None = None
        self.received_auxiliary_rows: list[torch.Tensor] = []
        with torch.no_grad():
            self.primary_projection.weight.zero_()
            self.primary_projection.bias.zero_()

    def forward(
        self,
        primary: torch.Tensor,
        auxiliary: torch.Tensor,
    ) -> torch.Tensor:
        """Predict using both cached positional inputs."""
        self.received_primary = primary
        self.received_auxiliary = auxiliary
        self.received_auxiliary_rows.append(auxiliary.detach().cpu())
        auxiliary_value = (
            auxiliary.to(dtype=primary.dtype)
            .reshape(primary.shape[0], -1)
            .mean(dim=1, keepdim=True)
        )
        return self.primary_projection(primary) + self.auxiliary_scale * auxiliary_value


def test_tensor_scale_stats_reports_scalar_distribution() -> None:
    """Tensor scale stats should summarize non-empty tensors."""
    stats = tensor_scale_stats(torch.tensor([1.0, 2.0, 3.0]))

    assert stats.count == 3
    assert stats.mean == 2.0
    assert stats.std == pytest.approx(0.8164966)
    assert stats.min == 1.0
    assert stats.max == 3.0
    assert stats.abs_mean == 2.0
    assert stats.abs_max == 3.0


def test_tensor_scale_stats_handles_empty_tensor() -> None:
    """Empty tensors should produce count zero and nullable stats."""
    stats = tensor_scale_stats(torch.empty((0,)))

    assert stats.count == 0
    assert stats.mean is None
    assert stats.std is None
    assert stats.min is None
    assert stats.max is None
    assert stats.abs_mean is None
    assert stats.abs_max is None


def test_entity_output_tanh_defaults_to_false() -> None:
    """Morpion entity-token value regression should default to no tanh."""
    training_args = MorpionTrainingArgs(
        dataset_file="/tmp/morpion_rows.jsonl",
        output_dir="/tmp/morpion_model",
    )
    model_args = MorpionRegressorArgs()
    entity_spec = entity_token_transformer_small_morpion_evaluator_spec()
    explicit_training_args = MorpionTrainingArgs(
        dataset_file="/tmp/morpion_rows.jsonl",
        output_dir="/tmp/morpion_model",
        entity_output_tanh=True,
    )

    assert training_args.entity_output_tanh is False
    assert model_args.entity_output_tanh is False
    assert entity_spec.entity_output_tanh is False
    assert explicit_training_args.entity_output_tanh is True


def test_cached_prediction_scale_diagnostics_forwards_every_tensor() -> None:
    """Cached scale diagnostics should preserve auxiliary rows and device."""
    model = _TwoInputScaleRegressor()
    primary = torch.zeros((4, 3))
    auxiliary = torch.arange(4, dtype=torch.long).reshape(4, 1, 1).expand(-1, 2, 3)
    targets = torch.zeros((4, 1))

    def build_batch(indices: tuple[int, ...]) -> TensorSupervisedBatch:
        index_tensor = torch.tensor(indices, dtype=torch.long)
        return TensorSupervisedBatch(
            input_tensor=primary[index_tensor],
            target_tensor=targets[index_tensor],
            auxiliary_input_tensors=(auxiliary[index_tensor],),
        )

    stats = prediction_scale_stats_for_cached_batches(
        model=model,
        batch_builder=build_batch,
        row_count=4,
        batch_size=2,
        device=torch.device("cpu"),
    )

    received_rows = torch.cat(model.received_auxiliary_rows)[:, 0, 0]
    assert torch.equal(received_rows, torch.arange(4))
    assert model.received_primary is not None
    assert model.received_primary.device.type == "cpu"
    assert model.received_auxiliary is not None
    assert model.received_auxiliary.device.type == "cpu"
    assert model.received_auxiliary.dtype == torch.long
    assert stats.count == 4
    assert stats.mean == pytest.approx(1.5)
    assert stats.std is not None
    assert torch.isfinite(torch.tensor(stats.std))


def test_flat_cached_streaming_training_reports_scale_metrics(
    tmp_path: Path,
) -> None:
    """Flat cached streaming training should expose target/prediction scale."""
    rows_path = _build_jsonl_rows_file(
        tmp_path,
        target_values=(10.0, 20.0, 30.0, 40.0),
    )
    output_dir = tmp_path / "flat_scale_bundle"

    _model, metrics = train_morpion_regressor_streaming(
        MorpionStreamingTrainingArgs(
            training_args=MorpionTrainingArgs(
                dataset_file=rows_path,
                output_dir=output_dir,
                batch_size=2,
                num_epochs=0,
                learning_rate=1e-3,
                shuffle=False,
                validation_fraction=0.25,
                device="cpu",
            ),
            row_chunk_size=2,
        )
    )

    _assert_scale_metric_keys(metrics)
    assert metrics["cached_global_shuffle"] == "true"
    assert metrics["target_mean"] == pytest.approx(25.0)
    assert metrics["target_abs_max"] == pytest.approx(40.0)
    assert metrics["target_zero_prediction_mse"] == pytest.approx(750.0)


def test_entity_token_cached_streaming_training_reports_scale_metrics(
    tmp_path: Path,
) -> None:
    """Entity-token cached training should expose target/prediction scale."""
    rows_path = _build_jsonl_rows_file(
        tmp_path,
        target_values=(10.0, 20.0, 30.0, 40.0),
    )
    output_dir = tmp_path / "entity_token_scale_bundle"

    _model, metrics = train_morpion_regressor_streaming(
        MorpionStreamingTrainingArgs(
            training_args=_small_entity_token_training_args(
                dataset_file=rows_path,
                output_dir=output_dir,
                num_epochs=0,
            ),
            row_chunk_size=2,
        )
    )

    _assert_scale_metric_keys(metrics)
    assert metrics["cached_global_shuffle"] == "true"
    assert metrics["entity_token_cache_used"] == "true"
    assert metrics["target_mean"] == pytest.approx(25.0)


def test_entity_output_tanh_warning_for_large_targets(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Explicit tanh should warn when targets are far outside [-1, 1]."""
    rows_path = _build_jsonl_rows_file(tmp_path, target_values=(20.0, 30.0))
    output_dir = tmp_path / "entity_token_tanh_warning_bundle"
    caplog.set_level(logging.WARNING)

    train_morpion_regressor_streaming(
        MorpionStreamingTrainingArgs(
            training_args=_small_entity_token_training_args(
                dataset_file=rows_path,
                output_dir=output_dir,
                num_epochs=0,
                entity_output_tanh=True,
            ),
            row_chunk_size=2,
        )
    )

    assert "entity_output_tanh=true" in caplog.text
    assert "outputs may be constrained" in caplog.text


def _assert_scale_metric_keys(metrics: dict[str, float | str | None]) -> None:
    """Assert that expected scale metrics are present."""
    for key in (
        "target_mean",
        "target_std",
        "target_min",
        "target_max",
        "target_abs_max",
        "target_zero_prediction_mse",
        "prediction_before_mean",
        "prediction_after_mean",
    ):
        assert key in metrics


def _small_entity_token_training_args(
    *,
    dataset_file: Path,
    output_dir: Path,
    num_epochs: int,
    entity_output_tanh: bool = False,
) -> MorpionTrainingArgs:
    """Return one tiny entity-token training config for scale tests."""
    return MorpionTrainingArgs(
        dataset_file=dataset_file,
        output_dir=output_dir,
        batch_size=2,
        num_epochs=num_epochs,
        learning_rate=1e-3,
        shuffle=False,
        validation_fraction=0.25,
        model_kind=MORPION_ENTITY_TOKEN_MODEL_KIND,
        entity_max_tokens=128,
        entity_d_model=16,
        entity_n_head=4,
        entity_n_layer=1,
        entity_dim_feedforward=32,
        entity_output_tanh=entity_output_tanh,
        device="cpu",
    )


def _build_jsonl_rows_file(
    tmp_path: Path,
    *,
    target_values: tuple[float, ...],
) -> Path:
    """Build one Morpion supervised JSONL rows artifact."""
    json_rows_path = _build_rows_file(tmp_path, target_values=target_values)
    rows = load_morpion_supervised_rows(json_rows_path)
    jsonl_rows_path = tmp_path / "morpion_supervised_rows.jsonl"
    save_morpion_supervised_rows_streaming(
        rows=rows.rows,
        metadata=rows.metadata,
        path=jsonl_rows_path,
    )
    return jsonl_rows_path
