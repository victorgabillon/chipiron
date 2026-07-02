"""Minimal supervised training helper for Morpion regressors."""

from __future__ import annotations

import json
import logging
import math
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from chipiron.environments.morpion.learning import (
    MorpionSupervisedRow,
    iter_morpion_supervised_row_chunks_from_path,
)
from chipiron.environments.morpion.players.evaluators.datasets.datasets import (
    MorpionGraphSupervisedDataset,
    MorpionGraphSupervisedDatasetArgs,
    MorpionGraphSupervisedSample,
    MorpionSupervisedDataset,
    MorpionSupervisedDatasetArgs,
    MorpionSupervisedSample,
    collate_morpion_graph_supervised_samples,
    process_morpion_supervised_row_to_graph_tensors,
    process_morpion_supervised_row_to_tensors,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.feature_schema import (
    DEFAULT_MORPION_FEATURE_SUBSET_NAME,
    MorpionFeatureSubset,
    resolve_morpion_feature_subset,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.graph_tokens import (
    MORPION_GRAPH_TOKEN_FEATURE_DIM,
    MorpionGraphTokenConverter,
    is_morpion_entity_token_transformer_model_kind,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.state_to_tensor import (
    MorpionFeatureTensorConverter,
)
from chipiron.environments.morpion.types import MorpionDynamics
from chipiron.learning.timing import PhaseDurations, format_phase_durations
from chipiron.learning.torch_runtime import (
    module_device,
    parameter_count,
    resolve_torch_device,
    torch_device_info,
)

from .bundle import MORPION_MANIFEST_FILE_NAME, save_morpion_model_bundle
from .model import MorpionRegressor, MorpionRegressorArgs, build_morpion_regressor

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

LOGGER = logging.getLogger(__name__)

type MorpionRegressionDataset = (
    MorpionSupervisedDataset
    | MorpionGraphSupervisedDataset
    | Subset[MorpionSupervisedSample]
    | Subset[MorpionGraphSupervisedSample]
)


class InvalidValidationFractionError(ValueError):
    """Raised when supervised-training validation splitting is outside bounds."""

    def __init__(self) -> None:
        """Initialize the invalid-validation-fraction error."""
        super().__init__("validation_fraction must be in [0.0, 1.0).")


class UnsupportedMorpionDiagnosticInputFormatError(ValueError):
    """Raised when diagnostics cannot infer a model's supervised-row input path."""

    reason = "unsupported_model_input_format"

    def __init__(self, model_kind: object) -> None:
        """Initialize the unsupported diagnostic input-format error."""
        super().__init__(
            f"Unsupported Morpion diagnostics model input format: {model_kind!r}."
        )


@dataclass(frozen=True, slots=True)
class MorpionTrainingArgs:
    """Arguments for the Morpion supervised-regression training helper."""

    dataset_file: str | os.PathLike[str]
    output_dir: str | os.PathLike[str]
    batch_size: int = 64
    num_epochs: int = 5
    learning_rate: float = 1e-3
    shuffle: bool = True
    model_kind: str = "linear"
    feature_subset_name: str = DEFAULT_MORPION_FEATURE_SUBSET_NAME
    feature_names: tuple[str, ...] = field(default_factory=tuple)
    hidden_sizes: tuple[int, ...] | None = None
    hidden_dim: int | None = None
    graph_max_tokens: int = 1536
    graph_input_feature_dim: int = MORPION_GRAPH_TOKEN_FEATURE_DIM
    graph_d_model: int = 64
    graph_n_head: int = 4
    graph_n_layer: int = 2
    graph_dim_feedforward: int = 256
    graph_dropout_ratio: float = 0.0
    graph_pooling: str = "value_token"
    graph_output_tanh: bool = True
    validation_fraction: float = 0.2
    validation_seed: int = 0
    device: str = "auto"

    def __post_init__(self) -> None:
        """Normalize feature subset metadata into a canonical explicit form."""
        if (
            not math.isfinite(self.validation_fraction)
            or self.validation_fraction < 0.0
            or self.validation_fraction >= 1.0
        ):
            raise InvalidValidationFractionError
        subset = resolve_morpion_feature_subset(
            feature_subset_name=self.feature_subset_name,
            feature_names=None if not self.feature_names else self.feature_names,
        )
        object.__setattr__(self, "feature_subset_name", subset.name)
        object.__setattr__(self, "feature_names", subset.feature_names)

    @property
    def feature_subset(self) -> MorpionFeatureSubset:
        """Return the resolved Morpion feature subset for this training job."""
        return MorpionFeatureSubset(
            name=self.feature_subset_name,
            feature_names=self.feature_names,
        )


@dataclass(frozen=True, slots=True)
class MorpionStreamingTrainingArgs:
    """Arguments for streaming Morpion supervised-regression training."""

    training_args: MorpionTrainingArgs
    row_chunk_size: int = 8192
    max_rows: int | None = None
    progress_callback: Callable[[int, int, int, int], None] | None = None

    def __post_init__(self) -> None:
        """Validate streaming controls."""
        if isinstance(self.row_chunk_size, bool) or self.row_chunk_size <= 0:
            raise ValueError("row_chunk_size must be a positive integer.")  # noqa: TRY003
        if self.max_rows is not None and (
            isinstance(self.max_rows, bool) or self.max_rows < 0
        ):
            raise ValueError("max_rows must be a non-negative integer or None.")  # noqa: TRY003


def train_morpion_regressor(
    args: MorpionTrainingArgs,
) -> tuple[MorpionRegressor, dict[str, float | str | None]]:
    """Train a Morpion regressor on persisted supervised rows."""
    total_started_at = perf_counter()
    timings = PhaseDurations()
    train_timings = PhaseDurations()
    collate_fn: Callable[[Any], Any] | None
    dataset: MorpionSupervisedDataset | MorpionGraphSupervisedDataset
    with timings.time_phase("dataset_load"):
        if is_morpion_entity_token_transformer_model_kind(args.model_kind):
            dataset = MorpionGraphSupervisedDataset(
                MorpionGraphSupervisedDatasetArgs(
                    file_name=os.fspath(args.dataset_file),
                    max_tokens=args.graph_max_tokens,
                )
            )
            collate_fn = collate_morpion_graph_supervised_samples
        else:
            dataset = MorpionSupervisedDataset(
                MorpionSupervisedDatasetArgs(
                    file_name=os.fspath(args.dataset_file),
                    feature_subset_name=args.feature_subset_name,
                    feature_names=args.feature_names,
                )
            )
            collate_fn = None
    train_dataset, validation_dataset = _split_train_validation_dataset(
        dataset,
        validation_fraction=args.validation_fraction,
        validation_seed=args.validation_seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        collate_fn=collate_fn,
    )

    resolved_hidden_sizes = _resolve_hidden_sizes(args)
    model_args = MorpionRegressorArgs(
        model_kind=args.model_kind,
        feature_subset_name=args.feature_subset_name,
        feature_names=args.feature_names,
        hidden_sizes=resolved_hidden_sizes,
        graph_max_tokens=args.graph_max_tokens,
        graph_input_feature_dim=args.graph_input_feature_dim,
        graph_d_model=args.graph_d_model,
        graph_n_head=args.graph_n_head,
        graph_n_layer=args.graph_n_layer,
        graph_dim_feedforward=args.graph_dim_feedforward,
        graph_dropout_ratio=args.graph_dropout_ratio,
        graph_pooling=args.graph_pooling,
        graph_output_tanh=args.graph_output_tanh,
    )
    device = resolve_torch_device(args.device)
    model = build_morpion_regressor(model_args).to(device)
    _log_training_device(
        model=model,
        requested_device=args.device,
        resolved_device=device,
        model_kind=args.model_kind,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.MSELoss()

    model.train()
    for epoch_index in range(args.num_epochs):
        epoch_timings = PhaseDurations()
        batch_count = 0
        with epoch_timings.time_phase("epoch_total"):
            for sample_batch in train_loader:
                batch_count += 1
                with _timed_torch_phase(epoch_timings, "batch_transfer", device):
                    sample_batch = _move_sample_batch_to_device(sample_batch, device)
                with _timed_torch_phase(epoch_timings, "zero_grad", device):
                    optimizer.zero_grad()
                with _timed_torch_phase(epoch_timings, "forward", device):
                    predictions = model(sample_batch.get_input_layer())
                targets = sample_batch.get_target_value()
                with _timed_torch_phase(epoch_timings, "loss", device):
                    loss = criterion(predictions, targets)
                with _timed_torch_phase(epoch_timings, "backward", device):
                    loss.backward()
                with _timed_torch_phase(epoch_timings, "optimizer_step", device):
                    optimizer.step()
        _add_phase_durations(train_timings, epoch_timings.as_dict())
        LOGGER.info(
            "[train-timing] mode=in_memory model_kind=%s epoch=%s/%s samples=%s "
            'batches=%s total_s=%.3f rows_per_s=%.3f phases="%s"',
            args.model_kind,
            epoch_index + 1,
            args.num_epochs,
            len(train_dataset),
            batch_count,
            epoch_timings.get("epoch_total"),
            _rows_per_second(len(train_dataset), epoch_timings.get("epoch_total")),
            format_phase_durations(epoch_timings.as_dict()),
        )
    timings.add_duration("train_epochs_total", train_timings.get("epoch_total"))

    with timings.time_phase("train_metrics_total"):
        train_stats = _evaluate_regression_metrics(
            model,
            train_dataset,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            device=device,
        )
    LOGGER.info(
        "[train-timing] mode=in_memory-eval model_kind=%s split=train samples=%s "
        'elapsed_s=%.3f rows_per_s=%.3f phases="%s"',
        args.model_kind,
        train_stats.sample_count,
        train_stats.elapsed_seconds,
        _rows_per_second(train_stats.sample_count, train_stats.elapsed_seconds),
        format_phase_durations(train_stats.phase_durations),
    )
    validation_loss: float | None
    validation_mae: float | None
    validation_stats: _RegressionEvaluationStats | None
    if len(validation_dataset) > 0:
        with timings.time_phase("validation_metrics_total"):
            validation_stats = _evaluate_regression_metrics(
                model,
                validation_dataset,
                batch_size=args.batch_size,
                collate_fn=collate_fn,
                device=device,
            )
        LOGGER.info(
            "[train-timing] mode=in_memory-eval model_kind=%s split=validation "
            'samples=%s elapsed_s=%.3f rows_per_s=%.3f phases="%s"',
            args.model_kind,
            validation_stats.sample_count,
            validation_stats.elapsed_seconds,
            _rows_per_second(
                validation_stats.sample_count,
                validation_stats.elapsed_seconds,
            ),
            format_phase_durations(validation_stats.phase_durations),
        )
        validation_loss = validation_stats.loss
        validation_mae = validation_stats.mae
    else:
        validation_stats = None
        validation_loss = None
        validation_mae = None
    train_loss = train_stats.loss
    train_mae = train_stats.mae
    final_loss = validation_loss if validation_loss is not None else train_stats.loss

    metrics: dict[str, float | str | None] = {
        "final_loss": final_loss,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "train_mae": train_mae,
        "validation_mae": validation_mae,
        "num_samples": float(len(dataset)),
        "num_train_samples": float(len(train_dataset)),
        "num_validation_samples": float(len(validation_dataset)),
        "num_epochs": float(args.num_epochs),
        "batch_size": float(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "loss_name": "mse",
        "requested_device": args.device,
        "resolved_device": str(device),
        "model_device": str(module_device(model)),
        "parameter_count": float(parameter_count(model)),
    }
    _add_timing_metrics(metrics, prefix="", phase_durations=timings.as_dict())
    _add_timing_metrics(
        metrics, prefix="train", phase_durations=train_timings.as_dict()
    )
    _add_timing_metrics(
        metrics,
        prefix="train_metrics",
        phase_durations=train_stats.phase_durations,
    )
    if validation_stats is not None:
        _add_timing_metrics(
            metrics,
            prefix="validation_metrics",
            phase_durations=validation_stats.phase_durations,
        )
    device_info = torch_device_info(
        requested_device=args.device,
        resolved_device=device,
    )
    timing_metadata: dict[str, object] = {
        "total_s": timings.get("total"),
        "dataset_load_s": timings.get("dataset_load"),
        "train_epochs_total_s": timings.get("train_epochs_total"),
        "train_metrics_total_s": timings.get("train_metrics_total"),
        "validation_metrics_total_s": timings.get("validation_metrics_total"),
        "bundle_save_s": 0.0,
        "train": train_timings.as_dict(),
        "train_metrics": train_stats.phase_durations,
        "validation_metrics": (
            {} if validation_stats is None else validation_stats.phase_durations
        ),
    }
    training_metadata = {
        "dataset_file": os.fspath(args.dataset_file),
        "output_dir": os.fspath(args.output_dir),
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "shuffle": args.shuffle,
        "model_kind": args.model_kind,
        "feature_subset_name": args.feature_subset_name,
        "feature_names": args.feature_names,
        "input_dim": model_args.input_dim,
        "hidden_sizes": resolved_hidden_sizes,
        "graph_max_tokens": args.graph_max_tokens,
        "graph_input_feature_dim": args.graph_input_feature_dim,
        "graph_d_model": args.graph_d_model,
        "graph_n_head": args.graph_n_head,
        "graph_n_layer": args.graph_n_layer,
        "graph_dim_feedforward": args.graph_dim_feedforward,
        "graph_dropout_ratio": args.graph_dropout_ratio,
        "graph_pooling": args.graph_pooling,
        "graph_output_tanh": args.graph_output_tanh,
        "validation_fraction": args.validation_fraction,
        "validation_seed": args.validation_seed,
        "requested_device": args.device,
        "resolved_device": str(device),
        "model_device": str(module_device(model)),
        "cuda_available": device_info.cuda_available,
        "cuda_device_count": device_info.cuda_device_count,
        "cuda_device_name": device_info.cuda_device_name,
        "parameter_count": float(parameter_count(model)),
        "timing": timing_metadata,
    }
    bundle_metadata = {
        **training_metadata,
        **metrics,
    }
    with timings.time_phase("bundle_save"):
        save_morpion_model_bundle(
            model,
            os.fspath(args.output_dir),
            model_args=model_args,
            metadata=bundle_metadata,
        )
    timings.add_duration("total", perf_counter() - total_started_at)
    timing_metadata["bundle_save_s"] = timings.get("bundle_save")
    timing_metadata["total_s"] = timings.get("total")
    metrics["timing_bundle_save_s"] = timings.get("bundle_save")
    metrics["timing_total_s"] = timings.get("total")
    bundle_metadata = {
        **training_metadata,
        **metrics,
    }
    _update_saved_manifest_metadata(args.output_dir, bundle_metadata)
    LOGGER.info(
        "[train-timing] mode=in_memory-summary model_kind=%s total_s=%.3f "
        "train_metrics_s=%.3f validation_metrics_s=%.3f bundle_save_s=%.3f "
        'phases="%s"',
        args.model_kind,
        timings.get("total"),
        timings.get("train_metrics_total"),
        timings.get("validation_metrics_total"),
        timings.get("bundle_save"),
        format_phase_durations(timings.as_dict()),
    )
    return model, metrics


def train_morpion_regressor_streaming(
    args: MorpionStreamingTrainingArgs,
) -> tuple[MorpionRegressor, dict[str, float | str | None]]:
    """Train a Morpion regressor from row chunks without materializing all rows."""
    total_started_at = perf_counter()
    timings = PhaseDurations()
    train_timings = PhaseDurations()
    training_args = args.training_args
    resolved_hidden_sizes = _resolve_hidden_sizes(training_args)
    model_args = MorpionRegressorArgs(
        model_kind=training_args.model_kind,
        feature_subset_name=training_args.feature_subset_name,
        feature_names=training_args.feature_names,
        hidden_sizes=resolved_hidden_sizes,
        graph_max_tokens=training_args.graph_max_tokens,
        graph_input_feature_dim=training_args.graph_input_feature_dim,
        graph_d_model=training_args.graph_d_model,
        graph_n_head=training_args.graph_n_head,
        graph_n_layer=training_args.graph_n_layer,
        graph_dim_feedforward=training_args.graph_dim_feedforward,
        graph_dropout_ratio=training_args.graph_dropout_ratio,
        graph_pooling=training_args.graph_pooling,
        graph_output_tanh=training_args.graph_output_tanh,
    )
    device = resolve_torch_device(training_args.device)
    model = build_morpion_regressor(model_args).to(device)
    _log_training_device(
        model=model,
        requested_device=training_args.device,
        resolved_device=device,
        model_kind=training_args.model_kind,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=training_args.learning_rate)
    criterion = torch.nn.MSELoss()
    split_policy = _streaming_split_policy(training_args.validation_fraction)

    for epoch_index in range(training_args.num_epochs):
        epoch_stats = _train_streaming_epoch(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            args=training_args,
            row_chunk_size=args.row_chunk_size,
            max_rows=args.max_rows,
            epoch_index=epoch_index,
            progress_callback=args.progress_callback,
            device=device,
        )
        _add_phase_durations(train_timings, epoch_stats.phase_durations)
        timings.add_duration("train_epochs_total", epoch_stats.elapsed_seconds)
        LOGGER.info(
            "[train-stream] epoch=%s chunks=%s train_samples=%s "
            "validation_samples=%s train_loss=%s split_policy=%s",
            epoch_index + 1,
            epoch_stats.chunk_count,
            epoch_stats.train_count,
            epoch_stats.validation_count,
            epoch_stats.loss,
            split_policy,
        )
        LOGGER.info(
            "[train-timing] mode=streaming model_kind=%s epoch=%s/%s chunks=%s "
            "train_samples=%s batches=%s elapsed_s=%.3f rows_per_s=%.3f "
            'phases="%s"',
            training_args.model_kind,
            epoch_index + 1,
            training_args.num_epochs,
            epoch_stats.chunk_count,
            epoch_stats.train_count,
            epoch_stats.batch_count,
            epoch_stats.elapsed_seconds,
            _rows_per_second(epoch_stats.train_count, epoch_stats.elapsed_seconds),
            format_phase_durations(epoch_stats.phase_durations),
        )

    with timings.time_phase("train_metrics_total"):
        train_stats = _evaluate_streaming_metrics(
            model=model,
            args=training_args,
            row_chunk_size=args.row_chunk_size,
            max_rows=args.max_rows,
            split="train",
            device=device,
        )
    LOGGER.info(
        "[train-timing] mode=streaming-eval model_kind=%s split=train samples=%s "
        'elapsed_s=%.3f rows_per_s=%.3f phases="%s"',
        training_args.model_kind,
        train_stats.sample_count,
        train_stats.elapsed_seconds,
        _rows_per_second(train_stats.sample_count, train_stats.elapsed_seconds),
        format_phase_durations(train_stats.phase_durations),
    )
    validation_loss: float | None
    validation_mae: float | None
    with timings.time_phase("validation_metrics_total"):
        validation_stats = _evaluate_streaming_metrics(
            model=model,
            args=training_args,
            row_chunk_size=args.row_chunk_size,
            max_rows=args.max_rows,
            split="validation",
            device=device,
        )
    LOGGER.info(
        "[train-timing] mode=streaming-eval model_kind=%s split=validation "
        'samples=%s elapsed_s=%.3f rows_per_s=%.3f phases="%s"',
        training_args.model_kind,
        validation_stats.sample_count,
        validation_stats.elapsed_seconds,
        _rows_per_second(
            validation_stats.sample_count,
            validation_stats.elapsed_seconds,
        ),
        format_phase_durations(validation_stats.phase_durations),
    )
    if validation_stats.sample_count > 0:
        validation_loss = validation_stats.loss
        validation_mae = validation_stats.mae
    else:
        validation_loss = None
        validation_mae = None
    train_loss = train_stats.loss
    train_mae = train_stats.mae
    train_count = train_stats.sample_count
    validation_count = validation_stats.sample_count
    num_samples = train_count + validation_count
    final_loss = validation_loss if validation_loss is not None else train_loss
    metrics: dict[str, float | str | None] = {
        "final_loss": final_loss,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "train_mae": train_mae,
        "validation_mae": validation_mae,
        "num_samples": float(num_samples),
        "num_train_samples": float(train_count),
        "num_validation_samples": float(validation_count),
        "num_epochs": float(training_args.num_epochs),
        "batch_size": float(training_args.batch_size),
        "learning_rate": float(training_args.learning_rate),
        "loss_name": "mse",
        "split_policy": split_policy,
        "requested_device": training_args.device,
        "resolved_device": str(device),
        "model_device": str(module_device(model)),
        "parameter_count": float(parameter_count(model)),
    }
    _add_timing_metrics(metrics, prefix="", phase_durations=timings.as_dict())
    _add_timing_metrics(
        metrics, prefix="train", phase_durations=train_timings.as_dict()
    )
    _add_timing_metrics(
        metrics,
        prefix="train_metrics",
        phase_durations=train_stats.phase_durations,
    )
    _add_timing_metrics(
        metrics,
        prefix="validation_metrics",
        phase_durations=validation_stats.phase_durations,
    )
    device_info = torch_device_info(
        requested_device=training_args.device,
        resolved_device=device,
    )
    timing_metadata: dict[str, object] = {
        "total_s": 0.0,
        "train_epochs_total_s": timings.get("train_epochs_total"),
        "train_metrics_total_s": timings.get("train_metrics_total"),
        "validation_metrics_total_s": timings.get("validation_metrics_total"),
        "bundle_save_s": 0.0,
        "train": train_timings.as_dict(),
        "train_metrics": train_stats.phase_durations,
        "validation_metrics": validation_stats.phase_durations,
    }
    training_metadata = {
        "dataset_file": os.fspath(training_args.dataset_file),
        "output_dir": os.fspath(training_args.output_dir),
        "batch_size": training_args.batch_size,
        "num_epochs": training_args.num_epochs,
        "learning_rate": training_args.learning_rate,
        "shuffle": training_args.shuffle,
        "model_kind": training_args.model_kind,
        "feature_subset_name": training_args.feature_subset_name,
        "feature_names": training_args.feature_names,
        "input_dim": model_args.input_dim,
        "hidden_sizes": resolved_hidden_sizes,
        "graph_max_tokens": training_args.graph_max_tokens,
        "graph_input_feature_dim": training_args.graph_input_feature_dim,
        "graph_d_model": training_args.graph_d_model,
        "graph_n_head": training_args.graph_n_head,
        "graph_n_layer": training_args.graph_n_layer,
        "graph_dim_feedforward": training_args.graph_dim_feedforward,
        "graph_dropout_ratio": training_args.graph_dropout_ratio,
        "graph_pooling": training_args.graph_pooling,
        "graph_output_tanh": training_args.graph_output_tanh,
        "validation_fraction": training_args.validation_fraction,
        "validation_seed": training_args.validation_seed,
        "streaming": True,
        "row_chunk_size": args.row_chunk_size,
        "max_rows": args.max_rows,
        "split_policy": split_policy,
        "requested_device": training_args.device,
        "resolved_device": str(device),
        "model_device": str(module_device(model)),
        "cuda_available": device_info.cuda_available,
        "cuda_device_count": device_info.cuda_device_count,
        "cuda_device_name": device_info.cuda_device_name,
        "parameter_count": float(parameter_count(model)),
        "timing": timing_metadata,
    }
    bundle_metadata = {
        **training_metadata,
        **metrics,
    }
    with timings.time_phase("bundle_save"):
        save_morpion_model_bundle(
            model,
            os.fspath(training_args.output_dir),
            model_args=model_args,
            metadata=bundle_metadata,
        )
    timings.add_duration("total", perf_counter() - total_started_at)
    timing_metadata["bundle_save_s"] = timings.get("bundle_save")
    timing_metadata["total_s"] = timings.get("total")
    metrics["timing_bundle_save_s"] = timings.get("bundle_save")
    metrics["timing_total_s"] = timings.get("total")
    bundle_metadata = {
        **training_metadata,
        **metrics,
    }
    _update_saved_manifest_metadata(training_args.output_dir, bundle_metadata)
    LOGGER.info(
        "[train-timing] mode=streaming-summary model_kind=%s total_s=%.3f "
        "train_epochs_s=%.3f train_metrics_s=%.3f validation_metrics_s=%.3f "
        'bundle_save_s=%.3f phases="%s"',
        training_args.model_kind,
        timings.get("total"),
        timings.get("train_epochs_total"),
        timings.get("train_metrics_total"),
        timings.get("validation_metrics_total"),
        timings.get("bundle_save"),
        format_phase_durations(timings.as_dict()),
    )
    return model, metrics


def predict_morpion_rows_for_diagnostics(
    model: nn.Module,
    row_examples: Sequence[MorpionSupervisedRow],
    *,
    feature_subset_name: str = DEFAULT_MORPION_FEATURE_SUBSET_NAME,
    feature_names: tuple[str, ...] = (),
) -> list[float]:
    """Predict raw Morpion rows using the model family's training input adapter."""
    if not row_examples:
        return []

    model_args = getattr(model, "args", None)
    model_kind = getattr(model_args, "model_kind", None)
    if model_kind is None:
        raise UnsupportedMorpionDiagnosticInputFormatError(model_kind)

    diagnostics_args = _diagnostic_training_args(
        model_args=model_args,
        feature_subset_name=feature_subset_name,
        feature_names=feature_names,
    )
    sample_batch = _rows_to_sample_batch(tuple(row_examples), args=diagnostics_args)
    model.eval()
    with torch.no_grad():
        predictions = model(
            _move_tensor_to_model_device(sample_batch.get_input_layer(), model)
        )
    prediction_values = predictions.squeeze(-1).detach().cpu().tolist()
    if isinstance(prediction_values, float):
        return [float(prediction_values)]
    return [float(prediction) for prediction in prediction_values]


def _split_train_validation_dataset(
    dataset: MorpionSupervisedDataset | MorpionGraphSupervisedDataset,
    *,
    validation_fraction: float,
    validation_seed: int,
) -> tuple[
    Subset[MorpionSupervisedSample] | Subset[MorpionGraphSupervisedSample],
    Subset[MorpionSupervisedSample] | Subset[MorpionGraphSupervisedSample],
]:
    """Return deterministic train/validation subsets for one supervised dataset."""
    sample_count = len(dataset)
    indices = list(range(sample_count))
    if sample_count < 2 or validation_fraction <= 0.0:
        return cast(
            "tuple[Subset[MorpionSupervisedSample] | Subset[MorpionGraphSupervisedSample], Subset[MorpionSupervisedSample] | Subset[MorpionGraphSupervisedSample]]",
            (Subset(dataset, indices), Subset(dataset, [])),
        )

    rng = random.Random(validation_seed)
    rng.shuffle(indices)
    validation_count = max(1, round(sample_count * validation_fraction))
    validation_count = min(sample_count - 1, validation_count)
    validation_indices = indices[:validation_count]
    train_indices = indices[validation_count:]
    return cast(
        "tuple[Subset[MorpionSupervisedSample] | Subset[MorpionGraphSupervisedSample], Subset[MorpionSupervisedSample] | Subset[MorpionGraphSupervisedSample]]",
        (Subset(dataset, train_indices), Subset(dataset, validation_indices)),
    )


def _log_training_device(
    *,
    model: nn.Module,
    requested_device: str,
    resolved_device: torch.device,
    model_kind: str,
) -> None:
    """Log resolved Torch runtime details for one training evaluator."""
    info = torch_device_info(
        requested_device=requested_device,
        resolved_device=resolved_device,
    )
    LOGGER.info(
        "[train-device] model_kind=%s requested_device=%s resolved_device=%s "
        "model_device=%s cuda_available=%s cuda_device_count=%s "
        "cuda_device_name=%s parameter_count=%s",
        model_kind,
        info.requested_device,
        info.resolved_device,
        module_device(model),
        info.cuda_available,
        info.cuda_device_count,
        info.cuda_device_name,
        parameter_count(model),
    )


def _move_sample_batch_to_device(
    sample_batch: MorpionSupervisedSample | MorpionGraphSupervisedSample,
    device: torch.device,
) -> MorpionSupervisedSample | MorpionGraphSupervisedSample:
    """Move one Morpion supervised batch to a concrete Torch device."""
    return type(sample_batch)(
        input_tensor=sample_batch.get_input_layer().to(device),
        target_tensor=sample_batch.get_target_value().to(device),
    )


def _add_phase_durations(
    target: PhaseDurations,
    phase_durations: Mapping[str, float],
) -> None:
    """Accumulate one phase-duration mapping into another timer."""
    for phase, seconds in phase_durations.items():
        target.add_duration(phase, seconds)


def _add_timing_metrics(
    metrics: dict[str, float | str | None],
    *,
    prefix: str,
    phase_durations: Mapping[str, float],
) -> None:
    """Add flat timing fields to a Morpion training metrics mapping."""
    for phase, seconds in phase_durations.items():
        key = f"timing_{prefix}_{phase}_s" if prefix else f"timing_{phase}_s"
        metrics[key] = float(seconds)


def _rows_per_second(row_count: int, elapsed_seconds: float) -> float:
    """Return one safe row-throughput value for logs."""
    if elapsed_seconds <= 0.0:
        return 0.0
    return row_count / elapsed_seconds


def _synchronize_if_cuda(device: torch.device) -> None:
    """Synchronize CUDA work so diagnostic phase timings include GPU execution."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def _timed_torch_phase(
    timings: PhaseDurations,
    phase: str,
    device: torch.device,
) -> Iterator[None]:
    """Measure a Torch phase, synchronizing CUDA before and after the phase."""
    _synchronize_if_cuda(device)
    started_at = perf_counter()
    try:
        yield
    finally:
        _synchronize_if_cuda(device)
        timings.add_duration(phase, perf_counter() - started_at)


def _update_saved_manifest_metadata(
    output_dir: str | os.PathLike[str],
    metadata: dict[str, object],
) -> None:
    """Update a saved Morpion manifest with final post-save metadata."""
    manifest_path = Path(output_dir) / MORPION_MANIFEST_FILE_NAME
    with open(manifest_path, encoding="utf-8") as handle:
        manifest_payload = json.load(handle)
    if isinstance(manifest_payload, dict):
        manifest_payload["metadata"] = metadata
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest_payload, handle, indent=2, sort_keys=True)


@dataclass(frozen=True, slots=True)
class _RegressionEvaluationStats:
    loss: float
    mae: float
    sample_count: int
    phase_durations: dict[str, float]
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class _StreamingEpochStats:
    chunk_count: int
    train_count: int
    validation_count: int
    batch_count: int
    loss: float
    phase_durations: dict[str, float]
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class _StreamingEvaluationStats:
    loss: float
    mae: float
    sample_count: int
    phase_durations: dict[str, float]
    elapsed_seconds: float


def _train_streaming_epoch(
    *,
    model: MorpionRegressor,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    args: MorpionTrainingArgs,
    row_chunk_size: int,
    max_rows: int | None,
    epoch_index: int,
    progress_callback: Callable[[int, int, int, int], None] | None,
    device: torch.device,
) -> _StreamingEpochStats:
    epoch_started_at = perf_counter()
    epoch_timings = PhaseDurations()
    model.train()
    chunk_count = 0
    train_count = 0
    validation_count = 0
    batch_count = 0
    squared_error_sum = 0.0
    value_count = 0
    rng = random.Random(args.validation_seed + epoch_index)
    row_chunk_iter = iter(
        _indexed_row_chunks(
            args.dataset_file,
            chunk_size=row_chunk_size,
            max_rows=max_rows,
        )
    )
    while True:
        try:
            with epoch_timings.time_phase("chunk_load"):
                row_start_index, rows = next(row_chunk_iter)
        except StopIteration:
            break
        chunk_count += 1
        with epoch_timings.time_phase("split_select"):
            train_rows: list[MorpionSupervisedRow] = []
            for offset, row in enumerate(rows):
                row_index = row_start_index + offset
                if _is_streaming_validation_index(
                    row_index,
                    validation_fraction=args.validation_fraction,
                ):
                    validation_count += 1
                else:
                    train_rows.append(row)
            if args.shuffle:
                rng.shuffle(train_rows)
        train_count += len(train_rows)
        row_batch_iter = iter(_row_batches(train_rows, batch_size=args.batch_size))
        while True:
            try:
                with epoch_timings.time_phase("row_batching"):
                    row_batch = next(row_batch_iter)
            except StopIteration:
                break
            batch_count += 1
            with epoch_timings.time_phase("row_to_sample_batch"):
                sample_batch = _rows_to_sample_batch(row_batch, args=args)
            with _timed_torch_phase(epoch_timings, "batch_transfer", device):
                sample_batch = _move_sample_batch_to_device(sample_batch, device)
            with _timed_torch_phase(epoch_timings, "zero_grad", device):
                optimizer.zero_grad()
            with _timed_torch_phase(epoch_timings, "forward", device):
                predictions = model(sample_batch.get_input_layer())
            targets = sample_batch.get_target_value()
            with _timed_torch_phase(epoch_timings, "loss", device):
                loss = criterion(predictions, targets)
            with _timed_torch_phase(epoch_timings, "backward", device):
                loss.backward()
            with _timed_torch_phase(epoch_timings, "optimizer_step", device):
                optimizer.step()
            with _timed_torch_phase(epoch_timings, "metric_accumulation", device):
                errors = predictions.detach() - targets
                squared_error_sum += float(torch.sum(errors * errors).item())
                value_count += int(targets.numel())
        if progress_callback is not None:
            with epoch_timings.time_phase("progress_callback"):
                progress_callback(
                    chunk_count,
                    row_start_index + len(rows),
                    epoch_index + 1,
                    args.num_epochs,
                )
    elapsed_seconds = perf_counter() - epoch_started_at
    epoch_timings.add_duration("epoch_total", elapsed_seconds)
    loss_value = 0.0 if value_count == 0 else squared_error_sum / value_count
    return _StreamingEpochStats(
        chunk_count=chunk_count,
        train_count=train_count,
        validation_count=validation_count,
        batch_count=batch_count,
        loss=loss_value,
        phase_durations=epoch_timings.as_dict(),
        elapsed_seconds=elapsed_seconds,
    )


def _evaluate_streaming_metrics(
    *,
    model: MorpionRegressor,
    args: MorpionTrainingArgs,
    row_chunk_size: int,
    max_rows: int | None,
    split: Literal["train", "validation"],
    device: torch.device,
) -> _StreamingEvaluationStats:
    evaluation_started_at = perf_counter()
    timings = PhaseDurations()
    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    value_count = 0
    model.eval()
    with torch.no_grad():
        row_chunk_iter = iter(
            _indexed_row_chunks(
                args.dataset_file,
                chunk_size=row_chunk_size,
                max_rows=max_rows,
            )
        )
        while True:
            try:
                with timings.time_phase("chunk_load"):
                    row_start_index, rows = next(row_chunk_iter)
            except StopIteration:
                break
            with timings.time_phase("split_select"):
                selected_rows = [
                    row
                    for offset, row in enumerate(rows)
                    if _streaming_row_is_in_split(
                        row_start_index + offset,
                        validation_fraction=args.validation_fraction,
                        split=split,
                    )
                ]
            row_batch_iter = iter(
                _row_batches(selected_rows, batch_size=args.batch_size)
            )
            while True:
                try:
                    with timings.time_phase("row_batching"):
                        row_batch = next(row_batch_iter)
                except StopIteration:
                    break
                with timings.time_phase("row_to_sample_batch"):
                    sample_batch = _rows_to_sample_batch(row_batch, args=args)
                with _timed_torch_phase(timings, "batch_transfer", device):
                    sample_batch = _move_sample_batch_to_device(sample_batch, device)
                with _timed_torch_phase(timings, "forward", device):
                    predictions = model(sample_batch.get_input_layer())
                targets = sample_batch.get_target_value()
                with _timed_torch_phase(timings, "metric_accumulation", device):
                    errors = predictions - targets
                    squared_error_sum += float(torch.sum(errors * errors).item())
                    absolute_error_sum += float(torch.sum(torch.abs(errors)).item())
                    value_count += int(targets.numel())
    elapsed_seconds = perf_counter() - evaluation_started_at
    if value_count == 0:
        return _StreamingEvaluationStats(
            loss=0.0,
            mae=0.0,
            sample_count=0,
            phase_durations=timings.as_dict(),
            elapsed_seconds=elapsed_seconds,
        )
    return _StreamingEvaluationStats(
        loss=squared_error_sum / value_count,
        mae=absolute_error_sum / value_count,
        sample_count=value_count,
        phase_durations=timings.as_dict(),
        elapsed_seconds=elapsed_seconds,
    )


def _indexed_row_chunks(
    path: str | os.PathLike[str],
    *,
    chunk_size: int,
    max_rows: int | None,
) -> Iterable[tuple[int, tuple[MorpionSupervisedRow, ...]]]:
    row_start_index = 0
    row_path = os.fspath(path)
    for rows in iter_morpion_supervised_row_chunks_from_path(
        row_path,
        chunk_size=chunk_size,
        max_rows=max_rows,
    ):
        yield row_start_index, rows
        row_start_index += len(rows)


def _streaming_row_is_in_split(
    row_index: int,
    *,
    validation_fraction: float,
    split: Literal["train", "validation"],
) -> bool:
    is_validation = _is_streaming_validation_index(
        row_index,
        validation_fraction=validation_fraction,
    )
    return is_validation if split == "validation" else not is_validation


def _is_streaming_validation_index(
    row_index: int,
    *,
    validation_fraction: float,
) -> bool:
    if validation_fraction <= 0.0:
        return False
    period = max(2, round(1.0 / validation_fraction))
    return (row_index + 1) % period == 0


def _streaming_split_policy(validation_fraction: float) -> str:
    return morpion_streaming_split_policy(validation_fraction)


def morpion_streaming_split_policy(validation_fraction: float) -> str:
    """Return the manifest/model metadata name for the streaming validation split."""
    if validation_fraction <= 0.0:
        return "none"
    return f"index_modulo_{max(2, round(1.0 / validation_fraction))}"


def _row_batches(
    rows: list[MorpionSupervisedRow],
    *,
    batch_size: int,
) -> Iterable[tuple[MorpionSupervisedRow, ...]]:
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        if batch:
            yield tuple(batch)


def _rows_to_sample_batch(
    rows: tuple[MorpionSupervisedRow, ...],
    *,
    args: MorpionTrainingArgs,
) -> MorpionSupervisedSample | MorpionGraphSupervisedSample:
    dynamics = MorpionDynamics()
    if is_morpion_entity_token_transformer_model_kind(args.model_kind):
        graph_converter = MorpionGraphTokenConverter(
            dynamics=dynamics,
            max_tokens=args.graph_max_tokens,
        )
        return collate_morpion_graph_supervised_samples(
            [
                process_morpion_supervised_row_to_graph_tensors(
                    row,
                    dynamics=dynamics,
                    converter=graph_converter,
                )
                for row in rows
            ]
        )
    feature_converter = MorpionFeatureTensorConverter(
        dynamics=dynamics,
        feature_subset=args.feature_subset,
    )
    samples = [
        process_morpion_supervised_row_to_tensors(
            row,
            dynamics=dynamics,
            converter=feature_converter,
        )
        for row in rows
    ]
    return MorpionSupervisedSample(
        input_tensor=torch.stack([sample.input_tensor for sample in samples]),
        target_tensor=torch.stack([sample.target_tensor for sample in samples]),
    )


def _diagnostic_training_args(
    *,
    model_args: object,
    feature_subset_name: str,
    feature_names: tuple[str, ...],
) -> MorpionTrainingArgs:
    model_kind = getattr(model_args, "model_kind", None)
    if model_kind not in {"linear", "mlp"} and not (
        isinstance(model_kind, str)
        and is_morpion_entity_token_transformer_model_kind(model_kind)
    ):
        raise UnsupportedMorpionDiagnosticInputFormatError(model_kind)
    return MorpionTrainingArgs(
        dataset_file="",
        output_dir="",
        model_kind=model_kind,
        feature_subset_name=str(
            getattr(model_args, "feature_subset_name", feature_subset_name)
        ),
        feature_names=tuple(getattr(model_args, "feature_names", feature_names)),
        hidden_sizes=cast(
            "tuple[int, ...] | None", getattr(model_args, "hidden_sizes", None)
        ),
        graph_max_tokens=int(getattr(model_args, "graph_max_tokens", 1536)),
        graph_input_feature_dim=int(
            getattr(
                model_args,
                "graph_input_feature_dim",
                MORPION_GRAPH_TOKEN_FEATURE_DIM,
            )
        ),
        graph_d_model=int(getattr(model_args, "graph_d_model", 64)),
        graph_n_head=int(getattr(model_args, "graph_n_head", 4)),
        graph_n_layer=int(getattr(model_args, "graph_n_layer", 2)),
        graph_dim_feedforward=int(getattr(model_args, "graph_dim_feedforward", 256)),
        graph_dropout_ratio=float(getattr(model_args, "graph_dropout_ratio", 0.0)),
        graph_pooling=str(getattr(model_args, "graph_pooling", "value_token")),
        graph_output_tanh=bool(getattr(model_args, "graph_output_tanh", True)),
        device="auto",
    )


def _move_tensor_to_model_device(
    tensor: torch.Tensor, model: nn.Module
) -> torch.Tensor:
    return tensor.to(module_device(model))


def _evaluate_regression_metrics(
    model: MorpionRegressor,
    dataset: MorpionRegressionDataset,
    *,
    batch_size: int,
    device: torch.device,
    collate_fn: Callable[[Any], Any] | None = None,
) -> _RegressionEvaluationStats:
    """Compute full-dataset mean MSE and MAE for one regression split."""
    evaluation_started_at = perf_counter()
    timings = PhaseDurations()
    if len(dataset) == 0:
        return _RegressionEvaluationStats(
            loss=0.0,
            mae=0.0,
            sample_count=0,
            phase_durations={},
            elapsed_seconds=0.0,
        )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    value_count = 0
    model.eval()
    with torch.no_grad():
        for sample_batch in data_loader:
            with _timed_torch_phase(timings, "batch_transfer", device):
                sample_batch = _move_sample_batch_to_device(sample_batch, device)
            with _timed_torch_phase(timings, "forward", device):
                predictions = model(sample_batch.get_input_layer())
            targets = sample_batch.get_target_value()
            with _timed_torch_phase(timings, "metric_accumulation", device):
                errors = predictions - targets
                squared_error_sum += float(torch.sum(errors * errors).item())
                absolute_error_sum += float(torch.sum(torch.abs(errors)).item())
                value_count += int(targets.numel())
    elapsed_seconds = perf_counter() - evaluation_started_at
    if value_count == 0:
        return _RegressionEvaluationStats(
            loss=0.0,
            mae=0.0,
            sample_count=0,
            phase_durations=timings.as_dict(),
            elapsed_seconds=elapsed_seconds,
        )
    return _RegressionEvaluationStats(
        loss=squared_error_sum / value_count,
        mae=absolute_error_sum / value_count,
        sample_count=value_count,
        phase_durations=timings.as_dict(),
        elapsed_seconds=elapsed_seconds,
    )


def _resolve_hidden_sizes(args: MorpionTrainingArgs) -> tuple[int, ...] | None:
    """Resolve legacy and current hidden-layer arguments into one tuple."""
    if args.hidden_sizes is not None:
        return args.hidden_sizes
    if args.hidden_dim is not None:
        return (args.hidden_dim,)
    return None


__all__ = [
    "MorpionStreamingTrainingArgs",
    "MorpionTrainingArgs",
    "UnsupportedMorpionDiagnosticInputFormatError",
    "morpion_streaming_split_policy",
    "predict_morpion_rows_for_diagnostics",
    "train_morpion_regressor",
    "train_morpion_regressor_streaming",
]
