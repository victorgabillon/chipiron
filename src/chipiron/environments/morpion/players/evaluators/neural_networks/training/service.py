"""Public Morpion neural-network training services."""

from __future__ import annotations

import logging
import os
import random
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

import torch
from torch.utils.data import DataLoader, Subset

from chipiron.environments.morpion.players.evaluators.datasets.datasets import (
    MorpionGraphSupervisedDataset,
    MorpionGraphSupervisedDatasetArgs,
    MorpionGraphSupervisedSample,
    MorpionSupervisedDataset,
    MorpionSupervisedDatasetArgs,
    MorpionSupervisedSample,
    collate_morpion_graph_supervised_samples,
    collate_morpion_supervised_samples,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.bundle import (
    save_morpion_model_bundle,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.graph_tokens import (
    is_morpion_entity_token_transformer_model_kind,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.model import (
    MorpionRegressor,
    build_morpion_regressor,
)
from chipiron.learning.supervised import (
    RegressionEvaluationStats,
    train_regression_batch,
)
from chipiron.learning.timing import PhaseDurations, format_phase_durations
from chipiron.learning.torch_runtime import (
    module_device,
    parameter_count,
    resolve_torch_device,
    torch_device_info,
)

from .device_logging import log_training_device
from .evaluation import evaluate_regression_metrics, evaluate_streaming_metrics
from .metadata import (
    add_phase_durations,
    add_timing_metrics,
    rows_per_second,
    update_saved_manifest_metadata,
)
from .model_args import (
    morpion_regressor_args_from_training_args,
    resolve_hidden_sizes,
)
from .streaming import streaming_split_policy, train_streaming_epoch

if TYPE_CHECKING:
    from collections.abc import Callable

    from .args import MorpionStreamingTrainingArgs, MorpionTrainingArgs

LOGGER = logging.getLogger(__name__)


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
            collate_fn = collate_morpion_supervised_samples
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

    resolved_hidden_sizes = resolve_hidden_sizes(args)
    model_args = morpion_regressor_args_from_training_args(args)
    device = resolve_torch_device(args.device)
    model = build_morpion_regressor(model_args).to(device)
    log_training_device(
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
                train_regression_batch(
                    model=model,
                    optimizer=optimizer,
                    criterion=criterion,
                    batch=sample_batch,
                    device=device,
                    timings=epoch_timings,
                )
        add_phase_durations(train_timings, epoch_timings.as_dict())
        LOGGER.info(
            "[train-timing] mode=in_memory model_kind=%s epoch=%s/%s samples=%s "
            'batches=%s total_s=%.3f rows_per_s=%.3f phases="%s"',
            args.model_kind,
            epoch_index + 1,
            args.num_epochs,
            len(train_dataset),
            batch_count,
            epoch_timings.get("epoch_total"),
            rows_per_second(len(train_dataset), epoch_timings.get("epoch_total")),
            format_phase_durations(epoch_timings.as_dict()),
        )
    timings.add_duration("train_epochs_total", train_timings.get("epoch_total"))

    with timings.time_phase("train_metrics_total"):
        train_stats = evaluate_regression_metrics(
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
        rows_per_second(train_stats.sample_count, train_stats.elapsed_seconds),
        format_phase_durations(train_stats.phase_durations),
    )
    validation_loss: float | None
    validation_mae: float | None
    validation_stats: RegressionEvaluationStats | None
    if len(validation_dataset) > 0:
        with timings.time_phase("validation_metrics_total"):
            validation_stats = evaluate_regression_metrics(
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
            rows_per_second(
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
    add_timing_metrics(metrics, prefix="", phase_durations=timings.as_dict())
    add_timing_metrics(metrics, prefix="train", phase_durations=train_timings.as_dict())
    add_timing_metrics(
        metrics,
        prefix="train_metrics",
        phase_durations=train_stats.phase_durations,
    )
    if validation_stats is not None:
        add_timing_metrics(
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
    update_saved_manifest_metadata(args.output_dir, bundle_metadata)
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
    resolved_hidden_sizes = resolve_hidden_sizes(training_args)
    model_args = morpion_regressor_args_from_training_args(training_args)
    device = resolve_torch_device(training_args.device)
    model = build_morpion_regressor(model_args).to(device)
    log_training_device(
        model=model,
        requested_device=training_args.device,
        resolved_device=device,
        model_kind=training_args.model_kind,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=training_args.learning_rate)
    criterion = torch.nn.MSELoss()
    split_policy = streaming_split_policy(training_args.validation_fraction)

    for epoch_index in range(training_args.num_epochs):
        epoch_stats = train_streaming_epoch(
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
        add_phase_durations(train_timings, epoch_stats.phase_durations)
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
            rows_per_second(epoch_stats.train_count, epoch_stats.elapsed_seconds),
            format_phase_durations(epoch_stats.phase_durations),
        )

    with timings.time_phase("train_metrics_total"):
        train_stats = evaluate_streaming_metrics(
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
        rows_per_second(train_stats.sample_count, train_stats.elapsed_seconds),
        format_phase_durations(train_stats.phase_durations),
    )
    validation_loss: float | None
    validation_mae: float | None
    with timings.time_phase("validation_metrics_total"):
        validation_stats = evaluate_streaming_metrics(
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
        rows_per_second(
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
    add_timing_metrics(metrics, prefix="", phase_durations=timings.as_dict())
    add_timing_metrics(metrics, prefix="train", phase_durations=train_timings.as_dict())
    add_timing_metrics(
        metrics,
        prefix="train_metrics",
        phase_durations=train_stats.phase_durations,
    )
    add_timing_metrics(
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
    update_saved_manifest_metadata(training_args.output_dir, bundle_metadata)
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
