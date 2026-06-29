"""Training helpers shared by Morpion bootstrap workflows."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

from chipiron.environments.morpion.players.evaluators.neural_networks.train import (
    MorpionStreamingTrainingArgs,
    MorpionTrainingArgs,
    train_morpion_regressor,
    train_morpion_regressor_streaming,
)

from .bootstrap_errors import (
    MissingBootstrapSelectedEvaluatorError,
    NoSelectableMorpionEvaluatorError,
    UnknownForcedMorpionEvaluatorError,
)
from .bootstrap_memory import log_after_cycle_gc
from .evaluator_config import MorpionEvaluatorsConfig
from .evaluator_diagnostics import (
    append_evaluator_training_diagnostics_history,
    build_evaluator_training_diagnostics,
    diagnostics_path,
    load_previous_evaluator_for_diagnostics,
    save_evaluator_training_diagnostics,
)
from .history import MorpionEvaluatorMetrics
from .pipeline_artifacts import MorpionPipelineEvaluatorTrainingResult
from .pipeline_memory import log_pipeline_memory

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from chipiron.environments.morpion.learning import (
        MorpionSupervisedRows,
        MorpionSupervisedRowsSource,
    )
    from chipiron.environments.morpion.players.evaluators.neural_networks.model import (
        MorpionRegressor,
    )

    from .bootstrap_args import MorpionBootstrapArgs
    from .bootstrap_paths import MorpionBootstrapPaths
    from .control import MorpionBootstrapControl
    from .evaluator_config import MorpionEvaluatorSpec
    from .memory_diagnostics import MemoryDiagnostics
    from .run_state import MorpionBootstrapRunState

LOGGER = logging.getLogger(__name__)
DEFAULT_STREAMING_DIAGNOSTIC_ROWS = 60
DIAGNOSTIC_SAMPLE_POLICY = "first_n"


class InvalidTrainingMetricError(TypeError):
    """Raised when persisted training metrics have the wrong value shape."""

    @classmethod
    def expected_numeric(cls, key: str) -> InvalidTrainingMetricError:
        """Build the canonical numeric-metric error."""
        return cls(f"Training metric {key!r} must be numeric.")

    @classmethod
    def expected_optional_numeric(cls, key: str) -> InvalidTrainingMetricError:
        """Build the canonical optional-numeric-metric error."""
        return cls(f"Training metric {key!r} must be numeric or None.")

    @classmethod
    def expected_optional_string(cls, key: str) -> InvalidTrainingMetricError:
        """Build the canonical optional-string-metric error."""
        return cls(f"Training metric {key!r} must be a string or None.")


@dataclass(frozen=True, slots=True)
class BootstrapTrainingResult:
    """Metrics and selected evaluator produced by bootstrap model training."""

    generation: int
    evaluator_metrics: dict[str, MorpionEvaluatorMetrics]
    evaluator_results: dict[str, MorpionPipelineEvaluatorTrainingResult]
    model_bundle_paths: dict[str, str]
    selected_evaluator_name: str
    selection_policy: str
    training_duration_s: float


def resolve_previous_model_bundle_path(
    *,
    paths: MorpionBootstrapPaths,
    run_state: MorpionBootstrapRunState,
    evaluator_name: str,
) -> Path | None:
    """Return the previous evaluator bundle path when one exists."""
    if run_state.latest_model_bundle_paths is None:
        return None
    relative_path = run_state.latest_model_bundle_paths.get(evaluator_name)
    if relative_path is None:
        return None
    return paths.resolve_work_dir_path(relative_path)


def morpion_training_args_from_evaluator_spec(
    *,
    spec: MorpionEvaluatorSpec,
    dataset_file: str | Path,
    output_dir: str | Path,
    shuffle: bool,
    validation_fraction: float,
    validation_seed: int,
) -> MorpionTrainingArgs:
    """Build training args for one evaluator spec."""
    return MorpionTrainingArgs(
        dataset_file=dataset_file,
        output_dir=output_dir,
        batch_size=spec.batch_size,
        num_epochs=spec.num_epochs,
        learning_rate=spec.learning_rate,
        shuffle=shuffle,
        model_kind=spec.model_type,
        feature_subset_name=spec.feature_subset_name,
        feature_names=spec.feature_names,
        hidden_sizes=spec.hidden_sizes,
        graph_max_tokens=spec.graph_max_tokens,
        graph_input_feature_dim=spec.graph_input_feature_dim,
        graph_d_model=spec.graph_d_model,
        graph_n_head=spec.graph_n_head,
        graph_n_layer=spec.graph_n_layer,
        graph_dim_feedforward=spec.graph_dim_feedforward,
        graph_dropout_ratio=spec.graph_dropout_ratio,
        graph_pooling=spec.graph_pooling,
        graph_output_tanh=spec.graph_output_tanh,
        validation_fraction=validation_fraction,
        validation_seed=validation_seed,
    )


def persist_evaluator_training_diagnostics(
    *,
    paths: MorpionBootstrapPaths,
    generation: int,
    evaluator_name: str,
    rows: MorpionSupervisedRows,
    created_at: str,
    spec: MorpionEvaluatorSpec,
    model_before: MorpionRegressor | None,
    model_after: MorpionRegressor,
    training_metrics: Mapping[str, object] | None = None,
) -> None:
    """Persist evaluator diagnostics without changing bootstrap semantics."""
    try:
        diagnostics = build_evaluator_training_diagnostics(
            generation=generation,
            evaluator_name=evaluator_name,
            rows=rows,
            created_at=created_at,
            feature_subset_name=spec.feature_subset_name,
            feature_names=spec.feature_names,
            model_before=model_before,
            model_after=model_after,
            training_metrics=training_metrics,
        )
        output_path = diagnostics_path(paths.work_dir, generation, evaluator_name)
        save_evaluator_training_diagnostics(diagnostics, output_path)
        append_evaluator_training_diagnostics_history(diagnostics, paths.work_dir)
        LOGGER.info(
            "[diagnostics] saved generation=%s evaluator=%s path=%s examples=%s worst=%s",
            generation,
            evaluator_name,
            output_path,
            len(diagnostics.representative_examples),
            len(diagnostics.worst_examples),
        )
    except Exception:
        LOGGER.exception(
            "[diagnostics] save_failed generation=%s evaluator=%s",
            generation,
            evaluator_name,
        )


def select_active_evaluator_name(
    evaluator_metrics: Mapping[str, MorpionEvaluatorMetrics],
) -> str:
    """Select the active evaluator using validation loss, falling back to final loss."""
    selectable_losses: list[tuple[str, float]] = []
    for evaluator_name, metrics in evaluator_metrics.items():
        selection_loss = _selection_loss(metrics)
        if selection_loss is not None:
            selectable_losses.append((evaluator_name, selection_loss))
    if not selectable_losses:
        raise NoSelectableMorpionEvaluatorError
    return min(selectable_losses, key=lambda item: item[1])[0]


def _required_metric_float(value: float | None, *, key: str) -> float:
    """Return one required numeric metric value or raise."""
    if value is None:
        raise InvalidTrainingMetricError.expected_numeric(key)
    return value


def _selection_loss(metrics: MorpionEvaluatorMetrics) -> float | None:
    """Return the finite loss used for evaluator selection."""
    if metrics.validation_loss is not None and math.isfinite(metrics.validation_loss):
        return metrics.validation_loss
    if metrics.final_loss is not None and math.isfinite(metrics.final_loss):
        return metrics.final_loss
    return None


def _has_validation_loss(
    evaluator_metrics: Mapping[str, MorpionEvaluatorMetrics],
) -> bool:
    """Return whether any evaluator reports a finite validation loss."""
    return any(
        metrics.validation_loss is not None and math.isfinite(metrics.validation_loss)
        for metrics in evaluator_metrics.values()
    )


def _metric_float(metrics: Mapping[str, object], key: str) -> float:
    value = metrics[key]
    if not isinstance(value, bool) and isinstance(value, int | float):
        return float(value)
    raise InvalidTrainingMetricError.expected_numeric(key)


def _metric_optional_float(metrics: Mapping[str, object], key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    if not isinstance(value, bool) and isinstance(value, int | float):
        return float(value)
    raise InvalidTrainingMetricError.expected_optional_numeric(key)


def _metric_int(
    metrics: Mapping[str, object],
    key: str,
    *,
    default: int | None = None,
) -> int:
    if key not in metrics:
        if default is None:
            raise KeyError(key)
        return default
    return int(_metric_float(metrics, key))


def _metric_optional_str(metrics: Mapping[str, object], key: str) -> str | None:
    value = metrics.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise InvalidTrainingMetricError.expected_optional_string(key)


def select_or_force_active_evaluator_name(
    *,
    evaluator_metrics: Mapping[str, MorpionEvaluatorMetrics],
    force_evaluator: str | None,
) -> str:
    """Return the forced evaluator when present, else the default auto-selection."""
    if force_evaluator is not None:
        if force_evaluator not in evaluator_metrics:
            raise UnknownForcedMorpionEvaluatorError(force_evaluator)
        return force_evaluator
    return select_active_evaluator_name(evaluator_metrics)


def restrict_evaluators_config(
    config: MorpionEvaluatorsConfig,
    evaluator_names: tuple[str, ...] | None,
) -> MorpionEvaluatorsConfig:
    """Return a config restricted to requested evaluator names, if any."""
    if not evaluator_names:
        return config
    missing_names = tuple(
        evaluator_name
        for evaluator_name in evaluator_names
        if evaluator_name not in config.evaluators
    )
    if missing_names:
        raise ValueError(
            "Unknown requested training evaluator names: " + ", ".join(missing_names)
        )
    return MorpionEvaluatorsConfig(
        evaluators={
            evaluator_name: config.evaluators[evaluator_name]
            for evaluator_name in evaluator_names
        }
    )


def diagnostic_rows_from_materialized_rows(
    rows: MorpionSupervisedRows,
    *,
    max_rows: int | None,
    source_format: str,
) -> MorpionSupervisedRows:
    """Return the bounded diagnostics view for materialized supervised rows."""
    if max_rows is None:
        return rows
    sample_rows = rows.rows[:max_rows]
    return replace(
        rows,
        rows=sample_rows,
        metadata={
            **rows.metadata,
            "diagnostic_sample_policy": DIAGNOSTIC_SAMPLE_POLICY,
            "diagnostic_sample_rows": len(sample_rows),
            "diagnostic_sample_max_rows": max_rows,
            "diagnostic_source_format": source_format,
        },
    )


def diagnostic_rows_from_streaming_rows(
    *,
    rows_path: Path,
    rows_source: MorpionSupervisedRowsSource,
    max_rows: int | None,
) -> MorpionSupervisedRows:
    """Return a bounded diagnostics sample from a streaming rows artifact."""
    from chipiron.environments.morpion.learning import (
        MorpionSupervisedRows,
        iter_morpion_supervised_rows_from_path,
    )

    effective_max_rows = (
        DEFAULT_STREAMING_DIAGNOSTIC_ROWS if max_rows is None else max_rows
    )
    sample_rows = tuple(
        iter_morpion_supervised_rows_from_path(
            rows_path,
            max_rows=effective_max_rows,
        )
    )
    return MorpionSupervisedRows(
        rows=sample_rows,
        metadata={
            **rows_source.metadata,
            "diagnostic_sample_policy": DIAGNOSTIC_SAMPLE_POLICY,
            "diagnostic_sample_rows": len(sample_rows),
            "diagnostic_sample_max_rows": effective_max_rows,
            "diagnostic_source_format": rows_source.format_kind,
        },
    )


def _log_diagnostic_sample(
    *,
    generation: int,
    evaluator_name: str,
    rows: MorpionSupervisedRows,
) -> None:
    if rows.metadata.get("diagnostic_sample_policy") != DIAGNOSTIC_SAMPLE_POLICY:
        return
    LOGGER.info(
        "[diagnostics] sampled generation=%s evaluator=%s rows=%s max_rows=%s "
        "policy=%s source_format=%s",
        generation,
        evaluator_name,
        rows.metadata.get("diagnostic_sample_rows", len(rows.rows)),
        rows.metadata.get("diagnostic_sample_max_rows"),
        rows.metadata.get("diagnostic_sample_policy"),
        rows.metadata.get("diagnostic_source_format"),
    )


def train_and_select_evaluators(
    *,
    args: MorpionBootstrapArgs,
    paths: MorpionBootstrapPaths,
    run_state: MorpionBootstrapRunState,
    rows: MorpionSupervisedRows,
    rows_path: Path,
    generation: int,
    timestamp_utc: str,
    resolved_evaluators_config: MorpionEvaluatorsConfig,
    resolved_control: MorpionBootstrapControl,
    memory: MemoryDiagnostics,
) -> BootstrapTrainingResult:
    """Train configured evaluators and select the active evaluator for search."""
    evaluator_metrics: dict[str, MorpionEvaluatorMetrics] = {}
    evaluator_results: dict[str, MorpionPipelineEvaluatorTrainingResult] = {}
    model_bundle_paths: dict[str, str] = {}
    training_started_at = time.perf_counter()
    memory.log("before_training")
    log_pipeline_memory(
        stage="training",
        generation=generation,
        event="start",
        rows_path=rows_path,
    )
    LOGGER.info(
        "[train] start evaluators=%s rows=%s",
        len(resolved_evaluators_config.evaluators),
        len(rows.rows),
    )
    diagnostic_rows = None
    if not args.skip_evaluator_diagnostics:
        diagnostic_rows = diagnostic_rows_from_materialized_rows(
            rows,
            max_rows=args.evaluator_diagnostics_max_rows,
            source_format="json",
        )
    for evaluator_name, spec in resolved_evaluators_config.evaluators.items():
        model_bundle_path = paths.model_bundle_path_for_generation(
            generation, evaluator_name
        )
        previous_model = None
        if not args.skip_evaluator_diagnostics:
            previous_model = load_previous_evaluator_for_diagnostics(
                resolve_previous_model_bundle_path(
                    paths=paths,
                    run_state=run_state,
                    evaluator_name=evaluator_name,
                )
            )
        LOGGER.info("[train] evaluator_start name=%s", evaluator_name)
        log_pipeline_memory(
            stage="training",
            generation=generation,
            event="before_evaluator",
            evaluator=evaluator_name,
        )
        evaluator_started_at = time.perf_counter()
        trained_model, metrics = train_morpion_regressor(
            morpion_training_args_from_evaluator_spec(
                spec=spec,
                dataset_file=rows_path,
                output_dir=model_bundle_path,
                shuffle=args.shuffle,
                validation_fraction=args.validation_fraction,
                validation_seed=args.validation_seed,
            )
        )
        memory.log("after_model_save")
        evaluator_elapsed_s = time.perf_counter() - evaluator_started_at
        learning_rate = _metric_optional_float(metrics, "learning_rate")
        evaluator_metrics[evaluator_name] = MorpionEvaluatorMetrics(
            final_loss=_metric_float(metrics, "final_loss"),
            train_loss=_metric_optional_float(metrics, "train_loss"),
            validation_loss=_metric_optional_float(metrics, "validation_loss"),
            train_mae=_metric_optional_float(metrics, "train_mae"),
            validation_mae=_metric_optional_float(metrics, "validation_mae"),
            num_epochs=_metric_int(metrics, "num_epochs"),
            num_samples=_metric_int(metrics, "num_samples"),
            num_train_samples=_metric_int(
                metrics,
                "num_train_samples",
                default=_metric_int(metrics, "num_samples"),
            ),
            num_validation_samples=_metric_int(
                metrics,
                "num_validation_samples",
                default=0,
            ),
            batch_size=_metric_int(metrics, "batch_size", default=spec.batch_size),
            learning_rate=(
                learning_rate if learning_rate is not None else spec.learning_rate
            ),
            loss_name=_metric_optional_str(metrics, "loss_name") or "mse",
        )
        model_bundle_paths[evaluator_name] = paths.relative_to_work_dir(
            model_bundle_path
        )
        evaluator_results[evaluator_name] = MorpionPipelineEvaluatorTrainingResult(
            final_loss=_required_metric_float(
                evaluator_metrics[evaluator_name].final_loss,
                key="final_loss",
            ),
            train_loss=evaluator_metrics[evaluator_name].train_loss,
            validation_loss=evaluator_metrics[evaluator_name].validation_loss,
            train_mae=evaluator_metrics[evaluator_name].train_mae,
            validation_mae=evaluator_metrics[evaluator_name].validation_mae,
            num_train_samples=evaluator_metrics[evaluator_name].num_train_samples,
            num_validation_samples=(
                evaluator_metrics[evaluator_name].num_validation_samples
            ),
            num_epochs=evaluator_metrics[evaluator_name].num_epochs,
            batch_size=evaluator_metrics[evaluator_name].batch_size,
            learning_rate=evaluator_metrics[evaluator_name].learning_rate,
            loss_name=evaluator_metrics[evaluator_name].loss_name,
            elapsed_s=evaluator_elapsed_s,
            model_bundle_path=model_bundle_paths[evaluator_name],
        )
        LOGGER.info(
            "[train] evaluator_done name=%s train_loss=%s "
            "validation_loss=%s final_loss=%s elapsed=%.3fs",
            evaluator_name,
            evaluator_metrics[evaluator_name].train_loss,
            evaluator_metrics[evaluator_name].validation_loss,
            evaluator_metrics[evaluator_name].final_loss,
            evaluator_elapsed_s,
        )
        log_pipeline_memory(
            stage="training",
            generation=generation,
            event="after_evaluator",
            evaluator=evaluator_name,
            final_loss=evaluator_metrics[evaluator_name].final_loss,
            validation_loss=evaluator_metrics[evaluator_name].validation_loss,
        )
        if args.skip_evaluator_diagnostics:
            LOGGER.info(
                "[diagnostics] skipped generation=%s evaluator=%s reason=skip_evaluator_diagnostics",
                generation,
                evaluator_name,
            )
        else:
            assert diagnostic_rows is not None
            _log_diagnostic_sample(
                generation=generation,
                evaluator_name=evaluator_name,
                rows=diagnostic_rows,
            )
            persist_evaluator_training_diagnostics(
                paths=paths,
                generation=generation,
                evaluator_name=evaluator_name,
                rows=diagnostic_rows,
                created_at=timestamp_utc,
                spec=spec,
                model_before=previous_model,
                model_after=trained_model,
                training_metrics=metrics,
            )
            memory.log("after_diagnostics")
        del previous_model
        del trained_model
        log_after_cycle_gc(memory, tag=f"after_evaluator:{evaluator_name}")

    LOGGER.info("[train] selection_start evaluators=%s", len(evaluator_metrics))
    selection_started_at = time.perf_counter()
    selected_evaluator_name: str | None = None
    selection_policy = (
        "forced_evaluator"
        if resolved_control.force_evaluator is not None
        else (
            "lowest_validation_loss"
            if _has_validation_loss(evaluator_metrics)
            else "lowest_final_loss"
        )
    )
    try:
        selected_evaluator_name = cast(
            "str | None",
            select_or_force_active_evaluator_name(
                evaluator_metrics=evaluator_metrics,
                force_evaluator=resolved_control.force_evaluator,
            ),
        )
    finally:
        LOGGER.info(
            "[train] selection_done elapsed=%.3fs selected=%s policy=%s",
            time.perf_counter() - selection_started_at,
            selected_evaluator_name,
            selection_policy,
        )
    if selected_evaluator_name is None:
        raise MissingBootstrapSelectedEvaluatorError

    training_duration_s = time.perf_counter() - training_started_at
    memory.log("after_training")
    log_pipeline_memory(
        stage="training",
        generation=generation,
        event="done",
        selected=selected_evaluator_name,
    )
    LOGGER.info("[train] done elapsed=%.3fs", training_duration_s)
    return BootstrapTrainingResult(
        generation=generation,
        evaluator_metrics=evaluator_metrics,
        evaluator_results=evaluator_results,
        model_bundle_paths=model_bundle_paths,
        selected_evaluator_name=selected_evaluator_name,
        selection_policy=selection_policy,
        training_duration_s=training_duration_s,
    )


def train_and_select_evaluators_streaming(
    *,
    args: MorpionBootstrapArgs,
    paths: MorpionBootstrapPaths,
    run_state: MorpionBootstrapRunState,
    rows_path: Path,
    rows_source: MorpionSupervisedRowsSource,
    generation: int,
    timestamp_utc: str,
    resolved_evaluators_config: MorpionEvaluatorsConfig,
    resolved_control: MorpionBootstrapControl,
    memory: MemoryDiagnostics,
    max_rows: int | None,
    chunk_size: int,
) -> BootstrapTrainingResult:
    """Train configured evaluators from JSONL chunks and select the active one."""
    evaluator_metrics: dict[str, MorpionEvaluatorMetrics] = {}
    evaluator_results: dict[str, MorpionPipelineEvaluatorTrainingResult] = {}
    model_bundle_paths: dict[str, str] = {}
    training_started_at = time.perf_counter()
    memory.log("before_training_streaming")
    log_pipeline_memory(
        stage="training",
        generation=generation,
        event="start_streaming",
        rows_path=rows_path,
        row_count=rows_source.row_count,
        chunk_size=chunk_size,
    )
    LOGGER.info(
        "[train-stream] start evaluators=%s rows=%s chunk_size=%s max_rows=%s",
        len(resolved_evaluators_config.evaluators),
        rows_source.row_count,
        chunk_size,
        max_rows,
    )
    diagnostic_rows: MorpionSupervisedRows | None = None
    if not args.skip_evaluator_diagnostics:
        diagnostic_rows = diagnostic_rows_from_streaming_rows(
            rows_path=rows_path,
            rows_source=rows_source,
            max_rows=args.evaluator_diagnostics_max_rows,
        )
    for evaluator_name, spec in resolved_evaluators_config.evaluators.items():
        model_bundle_path = paths.model_bundle_path_for_generation(
            generation, evaluator_name
        )
        previous_model = None
        if not args.skip_evaluator_diagnostics:
            previous_model = load_previous_evaluator_for_diagnostics(
                resolve_previous_model_bundle_path(
                    paths=paths,
                    run_state=run_state,
                    evaluator_name=evaluator_name,
                )
            )
        LOGGER.info("[train-stream] evaluator_start name=%s", evaluator_name)
        log_pipeline_memory(
            stage="training",
            generation=generation,
            event="before_evaluator_streaming",
            evaluator=evaluator_name,
        )
        evaluator_started_at = time.perf_counter()
        trained_model, metrics = train_morpion_regressor_streaming(
            MorpionStreamingTrainingArgs(
                training_args=morpion_training_args_from_evaluator_spec(
                    spec=spec,
                    dataset_file=rows_path,
                    output_dir=model_bundle_path,
                    shuffle=args.shuffle,
                    validation_fraction=args.validation_fraction,
                    validation_seed=args.validation_seed,
                ),
                row_chunk_size=chunk_size,
                max_rows=max_rows,
            )
        )
        memory.log("after_model_save")
        evaluator_elapsed_s = time.perf_counter() - evaluator_started_at
        learning_rate = _metric_optional_float(metrics, "learning_rate")
        evaluator_metrics[evaluator_name] = MorpionEvaluatorMetrics(
            final_loss=_metric_float(metrics, "final_loss"),
            train_loss=_metric_optional_float(metrics, "train_loss"),
            validation_loss=_metric_optional_float(metrics, "validation_loss"),
            train_mae=_metric_optional_float(metrics, "train_mae"),
            validation_mae=_metric_optional_float(metrics, "validation_mae"),
            num_epochs=_metric_int(metrics, "num_epochs"),
            num_samples=_metric_int(metrics, "num_samples"),
            num_train_samples=_metric_int(
                metrics,
                "num_train_samples",
                default=_metric_int(metrics, "num_samples"),
            ),
            num_validation_samples=_metric_int(
                metrics,
                "num_validation_samples",
                default=0,
            ),
            batch_size=_metric_int(metrics, "batch_size", default=spec.batch_size),
            learning_rate=(
                learning_rate if learning_rate is not None else spec.learning_rate
            ),
            loss_name=_metric_optional_str(metrics, "loss_name") or "mse",
        )
        model_bundle_paths[evaluator_name] = paths.relative_to_work_dir(
            model_bundle_path
        )
        evaluator_results[evaluator_name] = MorpionPipelineEvaluatorTrainingResult(
            final_loss=_required_metric_float(
                evaluator_metrics[evaluator_name].final_loss,
                key="final_loss",
            ),
            train_loss=evaluator_metrics[evaluator_name].train_loss,
            validation_loss=evaluator_metrics[evaluator_name].validation_loss,
            train_mae=evaluator_metrics[evaluator_name].train_mae,
            validation_mae=evaluator_metrics[evaluator_name].validation_mae,
            num_train_samples=evaluator_metrics[evaluator_name].num_train_samples,
            num_validation_samples=(
                evaluator_metrics[evaluator_name].num_validation_samples
            ),
            num_epochs=evaluator_metrics[evaluator_name].num_epochs,
            batch_size=evaluator_metrics[evaluator_name].batch_size,
            learning_rate=evaluator_metrics[evaluator_name].learning_rate,
            loss_name=evaluator_metrics[evaluator_name].loss_name,
            elapsed_s=evaluator_elapsed_s,
            model_bundle_path=model_bundle_paths[evaluator_name],
        )
        LOGGER.info(
            "[train-stream] evaluator_done name=%s train_loss=%s "
            "validation_loss=%s final_loss=%s elapsed=%.3fs",
            evaluator_name,
            evaluator_metrics[evaluator_name].train_loss,
            evaluator_metrics[evaluator_name].validation_loss,
            evaluator_metrics[evaluator_name].final_loss,
            evaluator_elapsed_s,
        )
        log_pipeline_memory(
            stage="training",
            generation=generation,
            event="after_evaluator_streaming",
            evaluator=evaluator_name,
            final_loss=evaluator_metrics[evaluator_name].final_loss,
            validation_loss=evaluator_metrics[evaluator_name].validation_loss,
        )
        if args.skip_evaluator_diagnostics:
            LOGGER.info(
                "[diagnostics] skipped generation=%s evaluator=%s reason=skip_evaluator_diagnostics",
                generation,
                evaluator_name,
            )
        elif diagnostic_rows is not None:
            _log_diagnostic_sample(
                generation=generation,
                evaluator_name=evaluator_name,
                rows=diagnostic_rows,
            )
            persist_evaluator_training_diagnostics(
                paths=paths,
                generation=generation,
                evaluator_name=evaluator_name,
                rows=diagnostic_rows,
                created_at=timestamp_utc,
                spec=spec,
                model_before=previous_model,
                model_after=trained_model,
                training_metrics=metrics,
            )
            memory.log("after_diagnostics")
        del previous_model
        del trained_model
        log_after_cycle_gc(memory, tag=f"after_evaluator:{evaluator_name}")

    LOGGER.info("[train-stream] selection_start evaluators=%s", len(evaluator_metrics))
    selection_started_at = time.perf_counter()
    selected_evaluator_name: str | None = None
    selection_policy = (
        "forced_evaluator"
        if resolved_control.force_evaluator is not None
        else (
            "lowest_validation_loss"
            if _has_validation_loss(evaluator_metrics)
            else "lowest_final_loss"
        )
    )
    try:
        selected_evaluator_name = cast(
            "str | None",
            select_or_force_active_evaluator_name(
                evaluator_metrics=evaluator_metrics,
                force_evaluator=resolved_control.force_evaluator,
            ),
        )
    finally:
        LOGGER.info(
            "[train-stream] selection_done elapsed=%.3fs selected=%s policy=%s",
            time.perf_counter() - selection_started_at,
            selected_evaluator_name,
            selection_policy,
        )
    if selected_evaluator_name is None:
        raise MissingBootstrapSelectedEvaluatorError

    training_duration_s = time.perf_counter() - training_started_at
    memory.log("after_training_streaming")
    log_pipeline_memory(
        stage="training",
        generation=generation,
        event="done_streaming",
        selected=selected_evaluator_name,
    )
    LOGGER.info("[train-stream] done elapsed=%.3fs", training_duration_s)
    return BootstrapTrainingResult(
        generation=generation,
        evaluator_metrics=evaluator_metrics,
        evaluator_results=evaluator_results,
        model_bundle_paths=model_bundle_paths,
        selected_evaluator_name=selected_evaluator_name,
        selection_policy=selection_policy,
        training_duration_s=training_duration_s,
    )


__all__ = [
    "BootstrapTrainingResult",
    "MorpionStreamingTrainingArgs",
    "MorpionTrainingArgs",
    "diagnostic_rows_from_materialized_rows",
    "diagnostic_rows_from_streaming_rows",
    "morpion_training_args_from_evaluator_spec",
    "persist_evaluator_training_diagnostics",
    "resolve_previous_model_bundle_path",
    "restrict_evaluators_config",
    "select_active_evaluator_name",
    "select_or_force_active_evaluator_name",
    "train_and_select_evaluators",
    "train_and_select_evaluators_streaming",
    "train_morpion_regressor",
    "train_morpion_regressor_streaming",
]
