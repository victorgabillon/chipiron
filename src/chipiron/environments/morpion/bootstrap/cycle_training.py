"""Training helpers shared by Morpion bootstrap workflows."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from chipiron.environments.morpion.players.evaluators.neural_networks.train import (
    MorpionTrainingArgs,
    train_morpion_regressor,
)

from .bootstrap_errors import (
    MissingBootstrapSelectedEvaluatorError,
    NoSelectableMorpionEvaluatorError,
    UnknownForcedMorpionEvaluatorError,
)
from .bootstrap_memory import log_after_cycle_gc
from .evaluator_diagnostics import (
    append_evaluator_training_diagnostics_history,
    build_evaluator_training_diagnostics,
    diagnostics_path,
    load_previous_evaluator_for_diagnostics,
    save_evaluator_training_diagnostics,
)
from .history import MorpionEvaluatorMetrics
from .pipeline_artifacts import MorpionPipelineEvaluatorTrainingResult

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from chipiron.environments.morpion.learning import MorpionSupervisedRows
    from chipiron.environments.morpion.players.evaluators.neural_networks.model import (
        MorpionRegressor,
    )

    from .bootstrap_args import MorpionBootstrapArgs
    from .bootstrap_paths import MorpionBootstrapPaths
    from .control import MorpionBootstrapControl
    from .evaluator_config import MorpionEvaluatorsConfig, MorpionEvaluatorSpec
    from .memory_diagnostics import MemoryDiagnostics
    from .run_state import MorpionBootstrapRunState

LOGGER = logging.getLogger(__name__)


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
    """Select the active evaluator using the lowest available final loss."""
    selectable_losses = {
        evaluator_name: metrics.final_loss
        for evaluator_name, metrics in evaluator_metrics.items()
        if metrics.final_loss is not None and math.isfinite(metrics.final_loss)
    }
    if not selectable_losses:
        raise NoSelectableMorpionEvaluatorError
    return min(
        selectable_losses,
        key=lambda evaluator_name: selectable_losses[evaluator_name],
    )


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
    LOGGER.info(
        "[train] start evaluators=%s rows=%s",
        len(resolved_evaluators_config.evaluators),
        len(rows.rows),
    )
    for evaluator_name, spec in resolved_evaluators_config.evaluators.items():
        model_bundle_path = paths.model_bundle_path_for_generation(
            generation, evaluator_name
        )
        previous_model = load_previous_evaluator_for_diagnostics(
            resolve_previous_model_bundle_path(
                paths=paths,
                run_state=run_state,
                evaluator_name=evaluator_name,
            )
        )
        LOGGER.info("[train] evaluator_start name=%s", evaluator_name)
        evaluator_started_at = time.perf_counter()
        trained_model, metrics = train_morpion_regressor(
            MorpionTrainingArgs(
                dataset_file=rows_path,
                output_dir=model_bundle_path,
                batch_size=spec.batch_size,
                num_epochs=spec.num_epochs,
                learning_rate=spec.learning_rate,
                shuffle=args.shuffle,
                model_kind=spec.model_type,
                feature_subset_name=spec.feature_subset_name,
                feature_names=spec.feature_names,
                hidden_sizes=spec.hidden_sizes,
            )
        )
        memory.log("after_model_save")
        evaluator_elapsed_s = time.perf_counter() - evaluator_started_at
        evaluator_metrics[evaluator_name] = MorpionEvaluatorMetrics(
            final_loss=float(metrics["final_loss"]),
            num_epochs=int(metrics["num_epochs"]),
            num_samples=int(metrics["num_samples"]),
        )
        model_bundle_paths[evaluator_name] = paths.relative_to_work_dir(
            model_bundle_path
        )
        evaluator_results[evaluator_name] = MorpionPipelineEvaluatorTrainingResult(
            final_loss=evaluator_metrics[evaluator_name].final_loss,
            elapsed_s=evaluator_elapsed_s,
            model_bundle_path=model_bundle_paths[evaluator_name],
        )
        LOGGER.info(
            "[train] evaluator_done name=%s final_loss=%s elapsed=%.3fs",
            evaluator_name,
            evaluator_metrics[evaluator_name].final_loss,
            evaluator_elapsed_s,
        )
        persist_evaluator_training_diagnostics(
            paths=paths,
            generation=generation,
            evaluator_name=evaluator_name,
            rows=rows,
            created_at=timestamp_utc,
            spec=spec,
            model_before=previous_model,
            model_after=trained_model,
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
        else "lowest_final_loss"
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


__all__ = [
    "BootstrapTrainingResult",
    "MorpionTrainingArgs",
    "persist_evaluator_training_diagnostics",
    "resolve_previous_model_bundle_path",
    "select_active_evaluator_name",
    "select_or_force_active_evaluator_name",
    "train_and_select_evaluators",
    "train_morpion_regressor",
]
