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
    MorpionEntityTokenSupervisedDataset,
    MorpionEntityTokenSupervisedDatasetArgs,
    MorpionRelationalEntityTokenSupervisedDataset,
    MorpionRelationalEntityTokenSupervisedDatasetArgs,
    MorpionSupervisedDataset,
    MorpionSupervisedDatasetArgs,
    collate_morpion_entity_token_supervised_samples,
    collate_morpion_relational_entity_token_supervised_samples,
    collate_morpion_supervised_samples,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.bundle import (
    save_morpion_model_bundle,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.entity_relations import (
    is_relational_entity_token_model_kind,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.entity_tokens import (
    is_morpion_entity_token_model_kind,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.model import (
    MorpionRegressor,
    build_morpion_regressor,
)
from chipiron.learning.supervised import (
    RegressionEvaluationStats,
    TensorSupervisedBatch,
    move_supervised_batch_to_device,
    regression_quality_stats_to_metadata,
    train_regression_batch,
)
from chipiron.learning.timing import PhaseDurations, format_phase_durations
from chipiron.learning.torch_runtime import (
    module_device,
    parameter_count,
    resolve_torch_device,
    torch_device_info,
)

from .cached_index_schedule import (
    cached_index_schedule,
    split_indices_for_streaming_policy,
)
from .device_logging import log_training_device
from .entity_token_cache import (
    MorpionEntityTokenCache,
    entity_token_cache_batch,
    load_or_materialize_entity_token_cache,
)
from .entity_token_cache_streaming import (
    evaluate_entity_token_cache_streaming_metrics,
    evaluate_relational_entity_token_cache_streaming_metrics,
    train_entity_token_cache_streaming_epoch,
    train_relational_entity_token_cache_streaming_epoch,
)
from .evaluation import evaluate_regression_metrics, evaluate_streaming_metrics
from .flat_cache_streaming import (
    evaluate_flat_cache_streaming_metrics,
    train_flat_cache_streaming_epoch,
)
from .flat_tensor_cache import (
    FlatTensorCache,
    flat_cache_batch,
    is_flat_morpion_training_model_kind,
    load_or_materialize_flat_tensor_cache,
)
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
from .quality_diagnostics import (
    SplitQualityDiagnostics,
    cached_split_regression_quality_diagnostics,
)
from .relational_entity_token_cache import (
    MorpionRelationalEntityTokenCache,
    load_or_materialize_relational_entity_token_cache,
    relational_entity_token_cache_batch,
)
from .scale_diagnostics import (
    TensorScaleStats,
    target_scale_metadata,
    tensor_scale_stats,
    tensor_scale_stats_to_metadata,
)
from .streaming import streaming_split_policy, train_streaming_epoch

if TYPE_CHECKING:
    from collections.abc import Callable

    from .args import MorpionStreamingTrainingArgs, MorpionTrainingArgs

LOGGER = logging.getLogger(__name__)
REGRESSION_QUALITY_SAMPLE_MAX_ROWS = 4096


def train_morpion_regressor(
    args: MorpionTrainingArgs,
) -> tuple[MorpionRegressor, dict[str, float | str | None]]:
    """Train a Morpion regressor on persisted supervised rows."""
    total_started_at = perf_counter()
    timings = PhaseDurations()
    train_timings = PhaseDurations()
    collate_fn: Callable[[Any], Any] | None
    dataset: (
        MorpionSupervisedDataset
        | MorpionEntityTokenSupervisedDataset
        | MorpionRelationalEntityTokenSupervisedDataset
    )
    with timings.time_phase("dataset_load"):
        if is_relational_entity_token_model_kind(args.model_kind):
            dataset = MorpionRelationalEntityTokenSupervisedDataset(
                MorpionRelationalEntityTokenSupervisedDatasetArgs(
                    file_name=os.fspath(args.dataset_file),
                    max_tokens=args.entity_max_tokens,
                )
            )
            collate_fn = collate_morpion_relational_entity_token_supervised_samples
        elif is_morpion_entity_token_model_kind(args.model_kind):
            dataset = MorpionEntityTokenSupervisedDataset(
                MorpionEntityTokenSupervisedDatasetArgs(
                    file_name=os.fspath(args.dataset_file),
                    max_tokens=args.entity_max_tokens,
                )
            )
            collate_fn = collate_morpion_entity_token_supervised_samples
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
        "entity_max_tokens": args.entity_max_tokens,
        "entity_input_feature_dim": args.entity_input_feature_dim,
        "entity_d_model": args.entity_d_model,
        "entity_n_head": args.entity_n_head,
        "entity_n_layer": args.entity_n_layer,
        "entity_dim_feedforward": args.entity_dim_feedforward,
        "entity_dropout_ratio": args.entity_dropout_ratio,
        "entity_pooling": args.entity_pooling,
        "entity_output_tanh": args.entity_output_tanh,
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
    if is_relational_entity_token_model_kind(args.model_kind):
        training_metadata.update({
            "entity_use_validity_feature": args.entity_use_validity_feature,
            "entity_relation_schema": args.entity_relation_schema,
            "entity_relation_type_count": args.entity_relation_type_count,
        })
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
    flat_cache: FlatTensorCache | None = None
    entity_token_cache: MorpionEntityTokenCache | None = None
    relational_entity_token_cache: MorpionRelationalEntityTokenCache | None = None
    flat_tensor_cache_metadata: dict[str, object] = {"used": False}
    entity_token_cache_metadata: dict[str, object] = {"used": False}
    relational_entity_token_cache_metadata: dict[str, object] = {"used": False}
    if is_flat_morpion_training_model_kind(training_args.model_kind):
        flat_cache = load_or_materialize_flat_tensor_cache(
            rows_path=training_args.dataset_file,
            row_chunk_size=args.row_chunk_size,
            max_rows=args.max_rows,
        )
        flat_tensor_cache_metadata = {
            "used": True,
            "rebuilt": flat_cache.rebuilt,
            "path": os.fspath(flat_cache.paths.tensor_path),
            "manifest_path": os.fspath(flat_cache.paths.manifest_path),
            "row_count": flat_cache.manifest.row_count,
            "feature_names": flat_cache.manifest.feature_names,
            "materialize_s": flat_cache.materialize_seconds,
            "load_s": flat_cache.load_seconds,
        }
        LOGGER.info(
            "[train-cache] kind=flat_tensor rows=%s rebuilt=%s materialize_s=%.3f "
            "load_s=%.3f path=%s",
            flat_cache.manifest.row_count,
            flat_cache.rebuilt,
            flat_cache.materialize_seconds,
            flat_cache.load_seconds,
            flat_cache.paths.tensor_path,
        )
    elif is_relational_entity_token_model_kind(training_args.model_kind):
        relational_entity_token_cache = (
            load_or_materialize_relational_entity_token_cache(
                rows_path=training_args.dataset_file,
                row_chunk_size=args.row_chunk_size,
                max_rows=args.max_rows,
                entity_max_tokens=training_args.entity_max_tokens,
            )
        )
        relational_entity_token_cache_metadata = {
            "used": True,
            "rebuilt": relational_entity_token_cache.rebuilt,
            "path": os.fspath(relational_entity_token_cache.paths.tensor_path),
            "manifest_path": os.fspath(
                relational_entity_token_cache.paths.manifest_path
            ),
            "row_count": relational_entity_token_cache.manifest.row_count,
            "entity_max_tokens": (
                relational_entity_token_cache.manifest.entity_max_tokens
            ),
            "input_feature_dim": (
                relational_entity_token_cache.manifest.input_feature_dim
            ),
            "relation_schema": (
                relational_entity_token_cache.manifest.relation_schema
            ),
            "relation_type_count": (
                relational_entity_token_cache.manifest.relation_type_count
            ),
            "materialize_s": relational_entity_token_cache.materialize_seconds,
            "load_s": relational_entity_token_cache.load_seconds,
        }
        LOGGER.info(
            "[train-cache] kind=relational_entity_token rows=%s rebuilt=%s "
            "materialize_s=%.3f load_s=%.3f path=%s",
            relational_entity_token_cache.manifest.row_count,
            relational_entity_token_cache.rebuilt,
            relational_entity_token_cache.materialize_seconds,
            relational_entity_token_cache.load_seconds,
            relational_entity_token_cache.paths.tensor_path,
        )
    elif is_morpion_entity_token_model_kind(training_args.model_kind):
        entity_token_cache = load_or_materialize_entity_token_cache(
            rows_path=training_args.dataset_file,
            row_chunk_size=args.row_chunk_size,
            max_rows=args.max_rows,
            entity_max_tokens=training_args.entity_max_tokens,
        )
        entity_token_cache_metadata = {
            "used": True,
            "rebuilt": entity_token_cache.rebuilt,
            "path": os.fspath(entity_token_cache.paths.tensor_path),
            "manifest_path": os.fspath(entity_token_cache.paths.manifest_path),
            "row_count": entity_token_cache.manifest.row_count,
            "entity_max_tokens": entity_token_cache.manifest.entity_max_tokens,
            "input_feature_dim": entity_token_cache.manifest.input_feature_dim,
            "materialize_s": entity_token_cache.materialize_seconds,
            "load_s": entity_token_cache.load_seconds,
        }
        LOGGER.info(
            "[train-cache] kind=entity_token rows=%s rebuilt=%s materialize_s=%.3f "
            "load_s=%.3f path=%s",
            entity_token_cache.manifest.row_count,
            entity_token_cache.rebuilt,
            entity_token_cache.materialize_seconds,
            entity_token_cache.load_seconds,
            entity_token_cache.paths.tensor_path,
        )
    target_scale: dict[str, object] = {}
    prediction_scale_before_training: dict[str, object] = {}
    prediction_scale_after_training: dict[str, object] = {}
    regression_quality: dict[str, object] = {
        "sample_max_rows": REGRESSION_QUALITY_SAMPLE_MAX_ROWS,
    }
    cached_prediction_batch_builder = _cached_prediction_batch_builder(
        flat_cache=flat_cache,
        entity_token_cache=entity_token_cache,
        relational_entity_token_cache=relational_entity_token_cache,
        training_args=training_args,
    )
    cached_row_count = _cached_row_count(
        flat_cache=flat_cache,
        entity_token_cache=entity_token_cache,
        relational_entity_token_cache=relational_entity_token_cache,
    )
    cached_target_tensor = _cached_target_tensor(
        flat_cache=flat_cache,
        entity_token_cache=entity_token_cache,
        relational_entity_token_cache=relational_entity_token_cache,
    )
    cached_training_schedule = _cached_training_schedule_metadata(
        cached_row_count=cached_row_count,
        max_rows=args.max_rows,
        validation_fraction=training_args.validation_fraction,
        shuffle=training_args.shuffle,
        cache_used=(
            flat_cache is not None
            or entity_token_cache is not None
            or relational_entity_token_cache is not None
        ),
    )
    target_stats: TensorScaleStats | None = None
    if cached_target_tensor is not None:
        target_stats = tensor_scale_stats(cached_target_tensor)
        target_scale = target_scale_metadata(cached_target_tensor)
        warn_if_output_tanh_mismatches_target_scale(
            model_kind=training_args.model_kind,
            entity_output_tanh=training_args.entity_output_tanh,
            target_stats=target_stats,
        )
    if cached_prediction_batch_builder is not None and cached_row_count > 0:
        prediction_before_stats = prediction_scale_stats_for_cached_batches(
            model=model,
            batch_builder=cached_prediction_batch_builder,
            row_count=cached_row_count,
            batch_size=training_args.batch_size,
            device=device,
        )
        prediction_scale_before_training = tensor_scale_stats_to_metadata(
            prediction_before_stats
        )
        _log_scale_before_training(
            model_kind=training_args.model_kind,
            target_stats=target_stats,
            prediction_stats=prediction_before_stats,
        )

    for epoch_index in range(training_args.num_epochs):
        if flat_cache is not None:
            epoch_stats = train_flat_cache_streaming_epoch(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                args=training_args,
                cache=flat_cache,
                row_chunk_size=args.row_chunk_size,
                max_rows=args.max_rows,
                epoch_index=epoch_index,
                progress_callback=args.progress_callback,
                device=device,
            )
        elif entity_token_cache is not None:
            epoch_stats = train_entity_token_cache_streaming_epoch(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                args=training_args,
                cache=entity_token_cache,
                row_chunk_size=args.row_chunk_size,
                max_rows=args.max_rows,
                epoch_index=epoch_index,
                progress_callback=args.progress_callback,
                device=device,
            )
        elif relational_entity_token_cache is not None:
            epoch_stats = train_relational_entity_token_cache_streaming_epoch(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                args=training_args,
                cache=relational_entity_token_cache,
                row_chunk_size=args.row_chunk_size,
                max_rows=args.max_rows,
                epoch_index=epoch_index,
                progress_callback=args.progress_callback,
                device=device,
            )
        else:
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
        if flat_cache is not None:
            train_stats = evaluate_flat_cache_streaming_metrics(
                model=model,
                args=training_args,
                cache=flat_cache,
                row_chunk_size=args.row_chunk_size,
                max_rows=args.max_rows,
                split="train",
                device=device,
            )
        elif entity_token_cache is not None:
            train_stats = evaluate_entity_token_cache_streaming_metrics(
                model=model,
                args=training_args,
                cache=entity_token_cache,
                row_chunk_size=args.row_chunk_size,
                max_rows=args.max_rows,
                split="train",
                device=device,
            )
        elif relational_entity_token_cache is not None:
            train_stats = evaluate_relational_entity_token_cache_streaming_metrics(
                model=model,
                args=training_args,
                cache=relational_entity_token_cache,
                row_chunk_size=args.row_chunk_size,
                max_rows=args.max_rows,
                split="train",
                device=device,
            )
        else:
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
        if flat_cache is not None:
            validation_stats = evaluate_flat_cache_streaming_metrics(
                model=model,
                args=training_args,
                cache=flat_cache,
                row_chunk_size=args.row_chunk_size,
                max_rows=args.max_rows,
                split="validation",
                device=device,
            )
        elif entity_token_cache is not None:
            validation_stats = evaluate_entity_token_cache_streaming_metrics(
                model=model,
                args=training_args,
                cache=entity_token_cache,
                row_chunk_size=args.row_chunk_size,
                max_rows=args.max_rows,
                split="validation",
                device=device,
            )
        elif relational_entity_token_cache is not None:
            validation_stats = (
                evaluate_relational_entity_token_cache_streaming_metrics(
                    model=model,
                    args=training_args,
                    cache=relational_entity_token_cache,
                    row_chunk_size=args.row_chunk_size,
                    max_rows=args.max_rows,
                    split="validation",
                    device=device,
                )
            )
        else:
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
    if cached_prediction_batch_builder is not None and cached_row_count > 0:
        prediction_after_stats = prediction_scale_stats_for_cached_batches(
            model=model,
            batch_builder=cached_prediction_batch_builder,
            row_count=cached_row_count,
            batch_size=training_args.batch_size,
            device=device,
        )
        prediction_scale_after_training = tensor_scale_stats_to_metadata(
            prediction_after_stats
        )
        _log_scale_after_training(
            model_kind=training_args.model_kind,
            prediction_stats=prediction_after_stats,
        )
        regression_quality = _cached_regression_quality_metadata(
            model=model,
            batch_builder=cached_prediction_batch_builder,
            cached_row_count=cached_row_count,
            max_rows=args.max_rows,
            validation_fraction=training_args.validation_fraction,
            batch_size=training_args.batch_size,
            device=device,
            sample_max_rows=REGRESSION_QUALITY_SAMPLE_MAX_ROWS,
        )
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
        "flat_tensor_cache_used": "true" if flat_cache is not None else "false",
        "entity_token_cache_used": "true"
        if entity_token_cache is not None
        else "false",
        "relational_entity_token_cache_used": "true"
        if relational_entity_token_cache is not None
        else "false",
        "cached_global_shuffle": (
            "true"
            if cached_training_schedule.get("global_shuffle") is True
            else "false"
        ),
    }
    if flat_cache is not None:
        metrics.update({
            "flat_tensor_cache_rebuilt": ("true" if flat_cache.rebuilt else "false"),
            "flat_tensor_cache_path": os.fspath(flat_cache.paths.tensor_path),
            "timing_flat_tensor_cache_materialize_s": (flat_cache.materialize_seconds),
            "timing_flat_tensor_cache_load_s": flat_cache.load_seconds,
        })
    if entity_token_cache is not None:
        metrics.update({
            "entity_token_cache_rebuilt": (
                "true" if entity_token_cache.rebuilt else "false"
            ),
            "entity_token_cache_path": os.fspath(entity_token_cache.paths.tensor_path),
            "timing_entity_token_cache_materialize_s": (
                entity_token_cache.materialize_seconds
            ),
            "timing_entity_token_cache_load_s": entity_token_cache.load_seconds,
        })
    if relational_entity_token_cache is not None:
        metrics.update({
            "relational_entity_token_cache_rebuilt": (
                "true" if relational_entity_token_cache.rebuilt else "false"
            ),
            "relational_entity_token_cache_path": os.fspath(
                relational_entity_token_cache.paths.tensor_path
            ),
            "timing_relational_entity_token_cache_materialize_s": (
                relational_entity_token_cache.materialize_seconds
            ),
            "timing_relational_entity_token_cache_load_s": (
                relational_entity_token_cache.load_seconds
            ),
        })
    _add_target_scale_metrics(metrics, target_scale)
    _add_prediction_scale_metrics(
        metrics,
        prefix="prediction_before",
        metadata=prediction_scale_before_training,
    )
    _add_prediction_scale_metrics(
        metrics,
        prefix="prediction_after",
        metadata=prediction_scale_after_training,
    )
    _add_regression_quality_metrics(
        metrics,
        prefix="train_quality",
        metadata=_quality_split_metadata(regression_quality, split="train"),
    )
    _add_regression_quality_metrics(
        metrics,
        prefix="validation_quality",
        metadata=_quality_split_metadata(regression_quality, split="validation"),
    )
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
        "entity_max_tokens": training_args.entity_max_tokens,
        "entity_input_feature_dim": training_args.entity_input_feature_dim,
        "entity_d_model": training_args.entity_d_model,
        "entity_n_head": training_args.entity_n_head,
        "entity_n_layer": training_args.entity_n_layer,
        "entity_dim_feedforward": training_args.entity_dim_feedforward,
        "entity_dropout_ratio": training_args.entity_dropout_ratio,
        "entity_pooling": training_args.entity_pooling,
        "entity_output_tanh": training_args.entity_output_tanh,
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
        "flat_tensor_cache": flat_tensor_cache_metadata,
        "entity_token_cache": entity_token_cache_metadata,
        "relational_entity_token_cache": relational_entity_token_cache_metadata,
        "cached_training_schedule": cached_training_schedule,
        "target_scale": target_scale,
        "prediction_scale_before_training": prediction_scale_before_training,
        "prediction_scale_after_training": prediction_scale_after_training,
        "regression_quality": regression_quality,
        "timing": timing_metadata,
    }
    if is_relational_entity_token_model_kind(training_args.model_kind):
        training_metadata.update({
            "entity_use_validity_feature": (
                training_args.entity_use_validity_feature
            ),
            "entity_relation_schema": training_args.entity_relation_schema,
            "entity_relation_type_count": training_args.entity_relation_type_count,
        })
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


def _cached_prediction_batch_builder(
    *,
    flat_cache: FlatTensorCache | None,
    entity_token_cache: MorpionEntityTokenCache | None,
    training_args: MorpionTrainingArgs,
    relational_entity_token_cache: MorpionRelationalEntityTokenCache | None = None,
) -> Callable[[tuple[int, ...]], TensorSupervisedBatch] | None:
    """Return a small-batch builder for cache-backed prediction diagnostics."""
    if flat_cache is not None:

        def build_flat_batch(row_indices: tuple[int, ...]) -> TensorSupervisedBatch:
            return flat_cache_batch(
                cache=flat_cache,
                row_indices=row_indices,
                requested_feature_names=training_args.feature_names,
            )

        return build_flat_batch
    if entity_token_cache is not None:

        def build_entity_token_batch(
            row_indices: tuple[int, ...],
        ) -> TensorSupervisedBatch:
            return entity_token_cache_batch(
                cache=entity_token_cache, row_indices=row_indices
            )

        return build_entity_token_batch
    if relational_entity_token_cache is not None:

        def build_relational_entity_token_batch(
            row_indices: tuple[int, ...],
        ) -> TensorSupervisedBatch:
            return relational_entity_token_cache_batch(
                cache=relational_entity_token_cache,
                row_indices=row_indices,
            )

        return build_relational_entity_token_batch
    return None


def _cached_row_count(
    *,
    flat_cache: FlatTensorCache | None,
    entity_token_cache: MorpionEntityTokenCache | None,
    relational_entity_token_cache: MorpionRelationalEntityTokenCache | None = None,
) -> int:
    """Return row count for whichever streaming cache is active."""
    if flat_cache is not None:
        return flat_cache.manifest.row_count
    if entity_token_cache is not None:
        return entity_token_cache.manifest.row_count
    if relational_entity_token_cache is not None:
        return relational_entity_token_cache.manifest.row_count
    return 0


def _cached_target_tensor(
    *,
    flat_cache: FlatTensorCache | None,
    entity_token_cache: MorpionEntityTokenCache | None,
    relational_entity_token_cache: MorpionRelationalEntityTokenCache | None = None,
) -> torch.Tensor | None:
    """Return target tensor for whichever streaming cache is active."""
    if flat_cache is not None:
        return flat_cache.target_tensor
    if entity_token_cache is not None:
        return entity_token_cache.target_tensor
    if relational_entity_token_cache is not None:
        return relational_entity_token_cache.target_tensor
    return None


def _cached_training_schedule_metadata(
    *,
    cached_row_count: int,
    max_rows: int | None,
    validation_fraction: float,
    shuffle: bool,
    cache_used: bool,
) -> dict[str, object]:
    """Return persisted metadata for cached training-index scheduling."""
    if not cache_used:
        return {
            "global_shuffle": False,
            "shuffle": shuffle,
        }
    effective_row_count = cached_row_count
    if max_rows is not None:
        effective_row_count = min(effective_row_count, max_rows)
    schedule = cached_index_schedule(
        row_count=effective_row_count,
        validation_fraction=validation_fraction,
    )
    return {
        "global_shuffle": True,
        "shuffle": shuffle,
        "row_count": schedule.row_count,
        "train_count": len(schedule.train_indices),
        "validation_count": len(schedule.validation_indices),
        "split_policy": schedule.split_policy,
    }


def _cached_regression_quality_metadata(
    *,
    model: MorpionRegressor,
    batch_builder: Callable[[tuple[int, ...]], TensorSupervisedBatch],
    cached_row_count: int,
    max_rows: int | None,
    validation_fraction: float,
    batch_size: int,
    device: torch.device,
    sample_max_rows: int,
) -> dict[str, object]:
    """Return sampled cached train/validation regression quality metadata."""
    effective_row_count = cached_row_count
    if max_rows is not None:
        effective_row_count = min(effective_row_count, max_rows)
    train_indices = split_indices_for_streaming_policy(
        row_count=effective_row_count,
        validation_fraction=validation_fraction,
        split="train",
    )
    validation_indices = split_indices_for_streaming_policy(
        row_count=effective_row_count,
        validation_fraction=validation_fraction,
        split="validation",
    )
    train_diagnostics = cached_split_regression_quality_diagnostics(
        model=model,
        batch_builder=batch_builder,
        row_indices=train_indices,
        batch_size=batch_size,
        device=device,
        max_rows=sample_max_rows,
        split="train",
    )
    validation_diagnostics = cached_split_regression_quality_diagnostics(
        model=model,
        batch_builder=batch_builder,
        row_indices=validation_indices,
        batch_size=batch_size,
        device=device,
        max_rows=sample_max_rows,
        split="validation",
    )
    _log_regression_quality(train_diagnostics)
    _log_regression_quality(validation_diagnostics)
    return {
        "sample_max_rows": sample_max_rows,
        "train": _split_quality_metadata(train_diagnostics),
        "validation": _split_quality_metadata(validation_diagnostics),
    }


def _split_quality_metadata(
    diagnostics: SplitQualityDiagnostics,
) -> dict[str, object]:
    """Return JSON-friendly metadata for one split's quality diagnostics."""
    metadata = regression_quality_stats_to_metadata(diagnostics.stats)
    metadata["split"] = diagnostics.split
    metadata["sample_count"] = diagnostics.sample_count
    return metadata


def _log_regression_quality(diagnostics: SplitQualityDiagnostics) -> None:
    """Log one compact regression quality line for a sampled split."""
    stats = diagnostics.stats
    LOGGER.info(
        "[train-quality] split=%s count=%s mse=%s mean_baseline_mse=%s "
        "r2=%s corr=%s target_std=%s prediction_std=%s "
        "pred_std_over_target_std=%s",
        diagnostics.split,
        diagnostics.sample_count,
        stats.mse,
        stats.mean_baseline_mse,
        stats.r2_vs_mean_baseline,
        stats.pearson_correlation,
        stats.target_std,
        stats.prediction_std,
        stats.prediction_std_over_target_std,
    )


def prediction_scale_stats_for_cached_batches(
    *,
    model: torch.nn.Module,
    batch_builder: Callable[[tuple[int, ...]], TensorSupervisedBatch],
    row_count: int,
    batch_size: int,
    device: torch.device,
    max_rows: int = 1024,
) -> TensorScaleStats:
    """Return prediction scale stats over a small prefix of cached rows."""
    sample_count = min(max_rows, row_count)
    if sample_count <= 0:
        return tensor_scale_stats(torch.empty((0,), dtype=torch.float32))
    effective_batch_size = max(1, batch_size)
    predictions: list[torch.Tensor] = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for start in range(0, sample_count, effective_batch_size):
            row_indices = tuple(
                range(start, min(start + effective_batch_size, sample_count))
            )
            sample_batch = batch_builder(row_indices)
            device_batch = move_supervised_batch_to_device(sample_batch, device)
            batch_predictions = model(*device_batch.get_model_input_tensors())
            predictions.append(batch_predictions.detach().cpu().reshape(-1))
    if was_training:
        model.train()
    if not predictions:
        return tensor_scale_stats(torch.empty((0,), dtype=torch.float32))
    return tensor_scale_stats(torch.cat(predictions))


def warn_if_output_tanh_mismatches_target_scale(
    *,
    model_kind: str,
    entity_output_tanh: bool,
    target_stats: TensorScaleStats,
) -> None:
    """Warn when tanh output scaling conflicts with unnormalized targets."""
    if (
        entity_output_tanh
        and target_stats.abs_max is not None
        and target_stats.abs_max > 2.0
    ):
        LOGGER.warning(
            "[train-scale] entity_output_tanh=true with target_abs_max=%s for "
            "model_kind=%s; outputs may be constrained near [-1, 1] while "
            "targets are unnormalized.",
            target_stats.abs_max,
            model_kind,
        )


def _log_scale_before_training(
    *,
    model_kind: str,
    target_stats: TensorScaleStats | None,
    prediction_stats: TensorScaleStats,
) -> None:
    """Log one target/prediction scale summary before training."""
    LOGGER.info(
        "[train-scale] model_kind=%s target_mean=%s target_std=%s "
        "target_min=%s target_max=%s prediction_before_mean=%s "
        "prediction_before_std=%s",
        model_kind,
        None if target_stats is None else target_stats.mean,
        None if target_stats is None else target_stats.std,
        None if target_stats is None else target_stats.min,
        None if target_stats is None else target_stats.max,
        prediction_stats.mean,
        prediction_stats.std,
    )


def _log_scale_after_training(
    *,
    model_kind: str,
    prediction_stats: TensorScaleStats,
) -> None:
    """Log one prediction scale summary after training."""
    LOGGER.info(
        "[train-scale] model_kind=%s prediction_after_mean=%s prediction_after_std=%s",
        model_kind,
        prediction_stats.mean,
        prediction_stats.std,
    )


def _add_target_scale_metrics(
    metrics: dict[str, float | str | None],
    metadata: dict[str, object],
) -> None:
    """Add target scale metadata to flat training metrics."""
    if not metadata:
        return
    metrics["target_count"] = _optional_metric_float(metadata.get("count"))
    metrics["target_mean"] = _optional_metric_float(metadata.get("mean"))
    metrics["target_std"] = _optional_metric_float(metadata.get("std"))
    metrics["target_min"] = _optional_metric_float(metadata.get("min"))
    metrics["target_max"] = _optional_metric_float(metadata.get("max"))
    metrics["target_abs_mean"] = _optional_metric_float(metadata.get("abs_mean"))
    metrics["target_abs_max"] = _optional_metric_float(metadata.get("abs_max"))
    metrics["target_zero_prediction_mse"] = _optional_metric_float(
        metadata.get("zero_prediction_mse")
    )


def _add_prediction_scale_metrics(
    metrics: dict[str, float | str | None],
    *,
    prefix: str,
    metadata: dict[str, object],
) -> None:
    """Add selected prediction scale metadata to flat training metrics."""
    if not metadata:
        return
    metrics[f"{prefix}_mean"] = _optional_metric_float(metadata.get("mean"))
    metrics[f"{prefix}_std"] = _optional_metric_float(metadata.get("std"))
    metrics[f"{prefix}_min"] = _optional_metric_float(metadata.get("min"))
    metrics[f"{prefix}_max"] = _optional_metric_float(metadata.get("max"))


def _quality_split_metadata(
    regression_quality: dict[str, object],
    *,
    split: str,
) -> dict[str, object]:
    """Return one split metadata mapping from regression quality metadata."""
    metadata = regression_quality.get(split)
    if not isinstance(metadata, dict):
        return {}
    return metadata


def _add_regression_quality_metrics(
    metrics: dict[str, float | str | None],
    *,
    prefix: str,
    metadata: dict[str, object],
) -> None:
    """Add sampled regression quality diagnostics to flat training metrics."""
    if not metadata:
        return
    for field_name in (
        "count",
        "target_mean",
        "target_std",
        "target_min",
        "target_max",
        "prediction_mean",
        "prediction_std",
        "prediction_min",
        "prediction_max",
        "residual_mean",
        "residual_std",
        "residual_min",
        "residual_max",
        "mse",
        "mae",
        "mean_baseline_mse",
        "zero_baseline_mse",
        "r2_vs_mean_baseline",
        "pearson_correlation",
        "prediction_std_over_target_std",
    ):
        metrics[f"{prefix}_{field_name}"] = _optional_metric_float(
            metadata.get(field_name)
        )


def _optional_metric_float(value: object) -> float | None:
    """Return one metrics-compatible optional float."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _split_train_validation_dataset(
    dataset: (
        MorpionSupervisedDataset
        | MorpionEntityTokenSupervisedDataset
        | MorpionRelationalEntityTokenSupervisedDataset
    ),
    *,
    validation_fraction: float,
    validation_seed: int,
) -> tuple[Subset[TensorSupervisedBatch], Subset[TensorSupervisedBatch]]:
    """Return deterministic train/validation subsets for one supervised dataset."""
    sample_count = len(dataset)
    indices = list(range(sample_count))
    if sample_count < 2 or validation_fraction <= 0.0:
        return cast(
            "tuple[Subset[TensorSupervisedBatch], Subset[TensorSupervisedBatch]]",
            (Subset(dataset, indices), Subset(dataset, [])),
        )

    rng = random.Random(validation_seed)
    rng.shuffle(indices)
    validation_count = max(1, round(sample_count * validation_fraction))
    validation_count = min(sample_count - 1, validation_count)
    validation_indices = indices[:validation_count]
    train_indices = indices[validation_count:]
    return cast(
        "tuple[Subset[TensorSupervisedBatch], Subset[TensorSupervisedBatch]]",
        (Subset(dataset, train_indices), Subset(dataset, validation_indices)),
    )
