"""Tests for Morpion regression quality training diagnostics."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from chipiron.environments.morpion.players.evaluators.neural_networks import (
    MORPION_GRAPH_MODEL_KIND,
    MORPION_MANIFEST_FILE_NAME,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
    MorpionStreamingTrainingArgs,
    MorpionTrainingArgs,
    train_morpion_regressor_streaming,
)
from tests.environments.test_morpion_flat_tensor_cache import _build_jsonl_rows_file

if TYPE_CHECKING:
    from pathlib import Path


def test_flat_cached_training_reports_quality_metrics_and_metadata(
    tmp_path: Path,
) -> None:
    """Flat cached streaming training should report sampled quality diagnostics."""
    rows_path = _build_jsonl_rows_file(
        tmp_path,
        target_values=(10.0, 20.0, 30.0, 40.0),
    )
    output_dir = tmp_path / "flat_quality_bundle"

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

    _assert_quality_metric_keys(metrics)
    assert metrics["flat_tensor_cache_used"] == "true"
    _assert_manifest_quality_metadata(output_dir)


def test_graph_cached_training_reports_quality_metrics_and_metadata(
    tmp_path: Path,
) -> None:
    """Graph cached streaming training should report sampled quality diagnostics."""
    rows_path = _build_jsonl_rows_file(
        tmp_path,
        target_values=(10.0, 20.0, 30.0, 40.0),
    )
    output_dir = tmp_path / "graph_quality_bundle"

    _model, metrics = train_morpion_regressor_streaming(
        MorpionStreamingTrainingArgs(
            training_args=_small_graph_training_args(
                dataset_file=rows_path,
                output_dir=output_dir,
                num_epochs=0,
            ),
            row_chunk_size=2,
        )
    )

    _assert_quality_metric_keys(metrics)
    assert metrics["graph_token_cache_used"] == "true"
    _assert_manifest_quality_metadata(output_dir)


def _assert_quality_metric_keys(metrics: dict[str, float | str | None]) -> None:
    """Assert that sampled train and validation quality metrics are present."""
    for prefix in ("train_quality", "validation_quality"):
        for suffix in (
            "mse",
            "mean_baseline_mse",
            "r2_vs_mean_baseline",
            "pearson_correlation",
            "prediction_std_over_target_std",
        ):
            assert f"{prefix}_{suffix}" in metrics


def _assert_manifest_quality_metadata(output_dir: Path) -> None:
    """Assert that saved bundle metadata includes regression quality details."""
    manifest_path = output_dir / MORPION_MANIFEST_FILE_NAME
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)

    regression_quality = manifest["metadata"]["regression_quality"]
    assert regression_quality["sample_max_rows"] == 4096
    assert "train" in regression_quality
    assert "validation" in regression_quality
    assert "mse" in regression_quality["train"]
    assert "mse" in regression_quality["validation"]


def _small_graph_training_args(
    *,
    dataset_file: Path,
    output_dir: Path,
    num_epochs: int,
) -> MorpionTrainingArgs:
    """Return one tiny graph-token training config for quality tests."""
    return MorpionTrainingArgs(
        dataset_file=dataset_file,
        output_dir=output_dir,
        batch_size=2,
        num_epochs=num_epochs,
        learning_rate=1e-3,
        shuffle=False,
        validation_fraction=0.25,
        model_kind=MORPION_GRAPH_MODEL_KIND,
        graph_max_tokens=128,
        graph_d_model=16,
        graph_n_head=4,
        graph_n_layer=1,
        graph_dim_feedforward=32,
        device="cpu",
    )
