"""Fitted-backup sanity loop on one frozen Morpion bootstrap tree."""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import torch
from anemone.training_export import (
    TrainingNodeSnapshot,
    TrainingTreeSnapshot,
    load_training_tree_snapshot,
)

from chipiron.environments.morpion.learning import (
    MorpionSupervisedRow,
    MorpionSupervisedRows,
    decode_morpion_state_ref_payload,
    save_morpion_supervised_rows,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.bundle import (
    load_morpion_regressor_for_inference,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.model import (
    MorpionRegressor,
    MorpionRegressorArgs,
    build_morpion_regressor,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.state_to_tensor import (
    MorpionFeatureTensorConverter,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.train import (
    MorpionTrainingArgs,
    train_morpion_regressor,
)
from chipiron.environments.morpion.types import MorpionDynamics

from .bootstrap_loop import MorpionBootstrapPaths
from .evaluator_diagnostics import (
    build_evaluator_training_diagnostics,
    save_evaluator_training_diagnostics,
)
from .evaluator_family import (
    CANONICAL_MORPION_EVALUATOR_FAMILY_PRESET,
    morpion_evaluators_config_from_preset,
)
from .evaluator_sanity_check import (
    EmptyMorpionSanityDatasetError,
    MorpionSanityDatasetMode,
    build_backup_target_diagnostics,
    terminal_path_nodes,
    top_terminal_path_nodes,
)

if TYPE_CHECKING:
    from pathlib import Path

    from .bootstrap_loop import MorpionEvaluatorSpec

LOGGER = logging.getLogger(__name__)


class MissingFittedBackupTargetError(ValueError):
    """Raised when a snapshot node cannot receive a fitted backup target."""

    def __init__(self, node_id: str) -> None:
        """Initialize the missing fitted-backup target error."""
        super().__init__(f"No fitted backup target is available for node {node_id!r}.")


class UnknownFittedBackupDatasetModeError(ValueError):
    """Raised when the fitted-backup dataset mode is not supported."""

    def __init__(self, dataset_mode: str) -> None:
        """Initialize the unknown dataset mode error."""
        super().__init__(f"Unknown fitted-backup dataset mode: {dataset_mode!r}.")


class UnknownFittedBackupEvaluatorError(ValueError):
    """Raised when the requested evaluator is missing from the selected family."""

    def __init__(self, evaluator_name: str) -> None:
        """Initialize the unknown evaluator error."""
        super().__init__(f"Unknown Morpion fitted-backup evaluator: {evaluator_name!r}.")


@dataclass(frozen=True, slots=True)
class FittedBackupNodeValue:
    """Per-node values for one fitted backup iteration."""

    node_id: str
    depth: int
    is_exact: bool
    is_terminal: bool
    direct_value_before_backup: float
    backed_up_target: float
    target_source: str
    previous_backed_up_target: float | None = None
    abs_target_change: float | None = None


@dataclass(frozen=True, slots=True)
class MorpionFittedBackupSanityArgs:
    """Arguments for one fixed-tree fitted-backup sanity run."""

    work_dir: str | Path
    generation: int | None = None
    dataset_mode: MorpionSanityDatasetMode = "bootstrap_like"
    max_rows: int | None = None
    evaluator_name: str = "mlp_41"
    evaluator_family_preset: str = CANONICAL_MORPION_EVALUATOR_FAMILY_PRESET
    num_iterations: int = 20
    num_epochs: int = 100
    batch_size: int = 200
    learning_rate: float = 1e-3
    shuffle: bool = True
    run_name: str | None = None
    top_terminal_path_count: int = 5
    max_backup_nodes: int | None = None


def run_fitted_backup_sanity(
    args: MorpionFittedBackupSanityArgs,
) -> dict[str, object]:
    """Run fitted value iteration on one frozen Morpion tree snapshot."""
    paths = MorpionBootstrapPaths.from_work_dir(args.work_dir)
    snapshot_path = _resolve_snapshot_path(paths=paths, generation=args.generation)
    snapshot = load_training_tree_snapshot(snapshot_path)
    source_snapshot_nodes = len(snapshot.nodes)
    snapshot = _truncate_snapshot(snapshot, max_backup_nodes=args.max_backup_nodes)
    spec = _selected_evaluator_spec(args)
    run_dir = _run_output_dir(
        paths.work_dir,
        generation=args.generation,
        run_name=args.run_name,
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        "[fitted_backup] snapshot_loaded path=%s nodes=%s backup_nodes=%s max_backup_nodes=%s",
        str(snapshot_path),
        source_snapshot_nodes,
        len(snapshot.nodes),
        args.max_backup_nodes,
    )
    selected_node_ids = _selected_node_ids(
        snapshot=snapshot,
        dataset_mode=args.dataset_mode,
        max_rows=args.max_rows,
        top_terminal_path_count=args.top_terminal_path_count,
    )
    if not selected_node_ids:
        raise EmptyMorpionSanityDatasetError

    current_model = _initial_model(
        paths=paths,
        generation=args.generation,
        evaluator_name=args.evaluator_name,
        spec=spec,
    )
    previous_targets: dict[str, float] | None = None
    iteration_summaries: list[dict[str, object]] = []

    for iteration in range(args.num_iterations):
        LOGGER.info("[fitted_backup] iteration_start iteration=%s", iteration)
        iteration_dir = run_dir / f"iteration_{iteration:03d}"
        rows_path = iteration_dir / "rows.json"
        model_dir = iteration_dir / "models" / args.evaluator_name
        diagnostics_path = iteration_dir / "diagnostics" / f"{args.evaluator_name}.json"
        target_diagnostics_path = iteration_dir / "target_diagnostics.json"

        backup_started_at = time.perf_counter()
        node_values = fitted_backup_node_values(
            snapshot=snapshot,
            model=current_model,
            spec=spec,
            previous_targets=previous_targets,
        )
        LOGGER.info(
            "[fitted_backup] backup_node_values_done iteration=%s nodes=%s elapsed=%.3fs",
            iteration,
            len(node_values),
            time.perf_counter() - backup_started_at,
        )
        rows = _rows_from_fitted_values(
            snapshot=snapshot,
            node_values=node_values,
            selected_node_ids=selected_node_ids,
            iteration=iteration,
        )
        save_morpion_supervised_rows(rows, rows_path)

        created_at = datetime.now(UTC).isoformat()
        diagnostics_started_at = time.perf_counter()
        target_diagnostics = build_backup_target_diagnostics(
            snapshot=_snapshot_with_fitted_backups(snapshot, node_values),
            rows=rows,
            created_at=created_at,
        )
        target_diagnostics_path.write_text(
            json.dumps(target_diagnostics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        LOGGER.info(
            "[fitted_backup] target_diagnostics_done iteration=%s elapsed=%.3fs",
            iteration,
            time.perf_counter() - diagnostics_started_at,
        )

        previous_predictions = _predict_rows(current_model, rows=rows, spec=spec)
        train_started_at = time.perf_counter()
        trained_model, metrics = train_morpion_regressor(
            MorpionTrainingArgs(
                dataset_file=rows_path,
                output_dir=model_dir,
                batch_size=args.batch_size,
                num_epochs=args.num_epochs,
                learning_rate=args.learning_rate,
                shuffle=args.shuffle,
                model_kind=spec.model_type,
                feature_subset_name=spec.feature_subset_name,
                feature_names=spec.feature_names,
                hidden_sizes=spec.hidden_sizes,
            )
        )
        LOGGER.info(
            "[fitted_backup] train_done iteration=%s final_loss=%s elapsed=%.3fs",
            iteration,
            metrics.get("final_loss"),
            time.perf_counter() - train_started_at,
        )
        diagnostics_started_at = time.perf_counter()
        diagnostics = build_evaluator_training_diagnostics(
            generation=iteration,
            evaluator_name=args.evaluator_name,
            rows=rows,
            created_at=created_at,
            feature_subset_name=spec.feature_subset_name,
            feature_names=spec.feature_names,
            model_before=current_model,
            model_after=trained_model,
        )
        save_evaluator_training_diagnostics(diagnostics, diagnostics_path)
        LOGGER.info(
            "[fitted_backup] diagnostics_done iteration=%s elapsed=%.3fs",
            iteration,
            time.perf_counter() - diagnostics_started_at,
        )

        next_predictions = _predict_rows(trained_model, rows=rows, spec=spec)
        iteration_summary = _iteration_summary(
            iteration=iteration,
            rows=rows,
            node_values=node_values,
            selected_node_ids=selected_node_ids,
            final_loss=float(metrics["final_loss"]),
            mae_after=diagnostics.mae_after,
            max_abs_error_after=diagnostics.max_abs_error_after,
            previous_predictions=previous_predictions,
            next_predictions=next_predictions,
            rows_path=rows_path,
            model_dir=model_dir,
            diagnostics_path=diagnostics_path,
            target_diagnostics_path=target_diagnostics_path,
        )
        iteration_summaries.append(iteration_summary)
        previous_targets = {
            node_id: node_value.backed_up_target
            for node_id, node_value in node_values.items()
        }
        current_model = trained_model
        LOGGER.info(
            "[fitted_backup] iteration_done iteration=%s final_loss=%s mean_abs_target_change=%s",
            iteration,
            iteration_summary["final_loss"],
            iteration_summary["mean_abs_target_change"],
        )

    summary: dict[str, object] = {
        "created_at": datetime.now(UTC).isoformat(),
        "work_dir": str(paths.work_dir),
        "snapshot_path": str(snapshot_path),
        "source_snapshot_nodes": source_snapshot_nodes,
        "backup_nodes": len(snapshot.nodes),
        "max_backup_nodes": args.max_backup_nodes,
        "generation": args.generation,
        "run_name": run_dir.name,
        "dataset_mode": args.dataset_mode,
        "evaluator_name": args.evaluator_name,
        "num_iterations": args.num_iterations,
        "iterations": iteration_summaries,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("[fitted_backup] summary_saved path=%s", str(summary_path))
    return summary


def fitted_backup_node_values(
    *,
    snapshot: TrainingTreeSnapshot,
    model: MorpionRegressor,
    spec: MorpionEvaluatorSpec,
    previous_targets: dict[str, float] | None = None,
) -> dict[str, FittedBackupNodeValue]:
    """Evaluate non-exact leaves and max-backup values through a frozen tree."""
    nodes_by_id = {node.node_id: node for node in snapshot.nodes}
    direct_started_at = time.perf_counter()
    direct_values = _direct_values_before_backup(snapshot=snapshot, model=model, spec=spec)
    LOGGER.info(
        "[fitted_backup] direct_values_done nodes=%s elapsed=%.3fs",
        len(direct_values),
        time.perf_counter() - direct_started_at,
    )
    backed_targets: dict[str, float] = {}
    target_sources: dict[str, str] = {}

    for node in sorted(snapshot.nodes, key=lambda item: item.depth, reverse=True):
        ground_truth = _ground_truth_value(node)
        if ground_truth is not None:
            backed_targets[node.node_id] = ground_truth
            target_sources[node.node_id] = "ground_truth_exact_or_terminal"
            continue

        child_values = [
            backed_targets[child_id]
            for child_id in node.child_ids
            if child_id in backed_targets
        ]
        if child_values:
            backed_targets[node.node_id] = max(child_values)
            target_sources[node.node_id] = "child_backup"
            continue

        if node.node_id not in direct_values:
            raise MissingFittedBackupTargetError(node.node_id)
        backed_targets[node.node_id] = direct_values[node.node_id]
        target_sources[node.node_id] = "frontier_prediction"

    return {
        node_id: FittedBackupNodeValue(
            node_id=node_id,
            depth=nodes_by_id[node_id].depth,
            is_exact=nodes_by_id[node_id].is_exact,
            is_terminal=nodes_by_id[node_id].is_terminal,
            direct_value_before_backup=direct_values[node_id],
            backed_up_target=backed_targets[node_id],
            target_source=target_sources[node_id],
            previous_backed_up_target=None
            if previous_targets is None
            else previous_targets.get(node_id),
            abs_target_change=None
            if previous_targets is None or node_id not in previous_targets
            else abs(backed_targets[node_id] - previous_targets[node_id]),
        )
        for node_id in nodes_by_id
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for ``python -m ...evaluator_fitted_backup_sanity``."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_fitted_backup_sanity(_parse_args(argv))
    return 0


def _parse_args(argv: list[str] | None) -> MorpionFittedBackupSanityArgs:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run fitted backup sanity checks on a frozen Morpion tree."
    )
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--generation", type=int)
    parser.add_argument(
        "--dataset-mode",
        choices=("bootstrap_like", "terminal_path", "top_terminal_paths"),
        default="bootstrap_like",
    )
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--evaluator-name", default="mlp_41")
    parser.add_argument(
        "--evaluator-family-preset",
        default=CANONICAL_MORPION_EVALUATOR_FAMILY_PRESET,
    )
    parser.add_argument("--num-iterations", type=int, default=20)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--run-name")
    parser.add_argument("--top-terminal-path-count", type=int, default=5)
    parser.add_argument("--max-backup-nodes", type=int)
    namespace = parser.parse_args(argv)
    return MorpionFittedBackupSanityArgs(
        work_dir=namespace.work_dir,
        generation=namespace.generation,
        dataset_mode=namespace.dataset_mode,
        max_rows=namespace.max_rows,
        evaluator_name=namespace.evaluator_name,
        evaluator_family_preset=namespace.evaluator_family_preset,
        num_iterations=namespace.num_iterations,
        num_epochs=namespace.num_epochs,
        batch_size=namespace.batch_size,
        learning_rate=namespace.learning_rate,
        shuffle=not namespace.no_shuffle,
        run_name=namespace.run_name,
        top_terminal_path_count=namespace.top_terminal_path_count,
        max_backup_nodes=namespace.max_backup_nodes,
    )


def _resolve_snapshot_path(
    *,
    paths: MorpionBootstrapPaths,
    generation: int | None,
) -> Path:
    """Return an existing tree export path."""
    if generation is not None:
        path = paths.tree_snapshot_path_for_generation(generation)
        if path.is_file():
            return path
        raise FileNotFoundError(path)
    candidates = sorted(paths.tree_snapshot_dir.glob("generation_*.json"))
    if not candidates:
        raise FileNotFoundError(paths.tree_snapshot_dir)
    return candidates[-1]


def _truncate_snapshot(
    snapshot: TrainingTreeSnapshot,
    *,
    max_backup_nodes: int | None,
) -> TrainingTreeSnapshot:
    """Return a prefix-limited diagnostic snapshot with internal links filtered."""
    if max_backup_nodes is None or len(snapshot.nodes) <= max_backup_nodes:
        return snapshot

    kept_nodes = snapshot.nodes[:max_backup_nodes]
    kept_node_ids = {node.node_id for node in kept_nodes}
    filtered_nodes = tuple(
        TrainingNodeSnapshot(
            node_id=node.node_id,
            parent_ids=tuple(
                parent_id for parent_id in node.parent_ids if parent_id in kept_node_ids
            ),
            child_ids=tuple(
                child_id for child_id in node.child_ids if child_id in kept_node_ids
            ),
            depth=node.depth,
            state_ref_payload=node.state_ref_payload,
            direct_value_scalar=node.direct_value_scalar,
            backed_up_value_scalar=node.backed_up_value_scalar,
            is_terminal=node.is_terminal,
            is_exact=node.is_exact,
            over_event_label=node.over_event_label,
            visit_count=node.visit_count,
            metadata=dict(node.metadata),
        )
        for node in kept_nodes
    )
    return TrainingTreeSnapshot(
        root_node_id=snapshot.root_node_id,
        nodes=filtered_nodes,
        metadata={
            **dict(snapshot.metadata),
            "fitted_backup_source_node_count": len(snapshot.nodes),
            "fitted_backup_max_backup_nodes": max_backup_nodes,
        },
    )


def _run_output_dir(
    work_dir: Path,
    *,
    generation: int | None,
    run_name: str | None,
) -> Path:
    """Return the fitted-backup output directory."""
    if run_name is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        generation_text = "latest" if generation is None else f"generation_{generation:06d}"
        run_name = f"{generation_text}_{timestamp}"
    return work_dir / "evaluator_fitted_backup_sanity" / run_name


def _selected_evaluator_spec(
    args: MorpionFittedBackupSanityArgs,
) -> MorpionEvaluatorSpec:
    """Return the configured evaluator spec for this experiment."""
    config = morpion_evaluators_config_from_preset(args.evaluator_family_preset)
    if args.evaluator_name not in config.evaluators:
        raise UnknownFittedBackupEvaluatorError(args.evaluator_name)
    return config.evaluators[args.evaluator_name]


def _initial_model(
    *,
    paths: MorpionBootstrapPaths,
    generation: int | None,
    evaluator_name: str,
    spec: MorpionEvaluatorSpec,
) -> MorpionRegressor:
    """Load a bootstrap model when available, otherwise build an untrained one."""
    bundle_path = _previous_bootstrap_bundle_path(
        paths=paths,
        generation=generation,
        evaluator_name=evaluator_name,
    )
    if bundle_path is not None:
        return load_morpion_regressor_for_inference(bundle_path)
    model = build_morpion_regressor(
        MorpionRegressorArgs(
            model_kind=spec.model_type,
            feature_subset_name=spec.feature_subset_name,
            feature_names=spec.feature_names,
            hidden_sizes=spec.hidden_sizes,
        )
    )
    model.eval()
    return model


def _previous_bootstrap_bundle_path(
    *,
    paths: MorpionBootstrapPaths,
    generation: int | None,
    evaluator_name: str,
) -> Path | None:
    """Return a bootstrap model bundle to seed iteration zero."""
    if generation is not None:
        candidate = paths.model_bundle_path_for_generation(generation, evaluator_name)
        return candidate if candidate.is_dir() else None
    for generation_dir in reversed(sorted(paths.model_dir.glob("generation_*"))):
        candidate = generation_dir / evaluator_name
        if candidate.is_dir():
            return candidate
    return None


def _selected_node_ids(
    *,
    snapshot: TrainingTreeSnapshot,
    dataset_mode: MorpionSanityDatasetMode,
    max_rows: int | None,
    top_terminal_path_count: int,
) -> tuple[str, ...]:
    """Return node IDs included in fitted-backup training rows."""
    if dataset_mode == "bootstrap_like":
        selected = tuple(node.node_id for node in snapshot.nodes)
    elif dataset_mode == "terminal_path":
        selected = tuple(node.node_id for node in terminal_path_nodes(snapshot))
    elif dataset_mode == "top_terminal_paths":
        selected = tuple(
            node.node_id
            for node in top_terminal_path_nodes(
                snapshot,
                max_terminal_nodes=top_terminal_path_count,
            )
        )
    else:
        raise UnknownFittedBackupDatasetModeError(dataset_mode)
    if max_rows is not None:
        selected = selected[:max_rows]
    return selected


def _ground_truth_value(node: TrainingNodeSnapshot) -> float | None:
    """Return fixed exact/terminal ground truth when available."""
    if not (node.is_exact or node.is_terminal):
        return None
    if node.backed_up_value_scalar is not None:
        return node.backed_up_value_scalar
    return node.direct_value_scalar


def _direct_values_before_backup(
    *,
    snapshot: TrainingTreeSnapshot,
    model: MorpionRegressor,
    spec: MorpionEvaluatorSpec,
) -> dict[str, float]:
    """Return exact ground truth or evaluator predictions before backup."""
    predictions = _predict_snapshot_nodes(snapshot=snapshot, model=model, spec=spec)
    values: dict[str, float] = {}
    for node in snapshot.nodes:
        ground_truth = _ground_truth_value(node)
        if ground_truth is not None:
            values[node.node_id] = ground_truth
        elif node.node_id in predictions:
            values[node.node_id] = predictions[node.node_id]
        elif node.direct_value_scalar is not None:
            values[node.node_id] = node.direct_value_scalar
    return values


def _predict_snapshot_nodes(
    *,
    snapshot: TrainingTreeSnapshot,
    model: MorpionRegressor,
    spec: MorpionEvaluatorSpec,
) -> dict[str, float]:
    """Predict values for snapshot nodes that have Morpion state payloads."""
    started_at = time.perf_counter()
    dynamics = MorpionDynamics()
    converter = MorpionFeatureTensorConverter(
        dynamics=dynamics,
        feature_subset=spec.feature_subset,
    )
    node_ids: list[str] = []
    tensors: list[torch.Tensor] = []
    for node in snapshot.nodes:
        if node.state_ref_payload is None:
            continue
        state = dynamics.wrap_atomheart_state(
            decode_morpion_state_ref_payload(node.state_ref_payload)
        )
        node_ids.append(node.node_id)
        tensors.append(converter.state_to_tensor(state))
    if not tensors:
        LOGGER.info(
            "[fitted_backup] predict_snapshot_nodes_done nodes=%s predicted=0 elapsed=%.3fs",
            len(snapshot.nodes),
            time.perf_counter() - started_at,
        )
        return {}
    model.eval()
    with torch.no_grad():
        raw_predictions = model(torch.stack(tensors)).squeeze(-1).detach().cpu().tolist()
    if isinstance(raw_predictions, float):
        raw_predictions = [raw_predictions]
    predictions = {
        node_id: float(prediction)
        for node_id, prediction in zip(node_ids, raw_predictions, strict=True)
    }
    LOGGER.info(
        "[fitted_backup] predict_snapshot_nodes_done nodes=%s predicted=%s elapsed=%.3fs",
        len(snapshot.nodes),
        len(predictions),
        time.perf_counter() - started_at,
    )
    return predictions


def _rows_from_fitted_values(
    *,
    snapshot: TrainingTreeSnapshot,
    node_values: dict[str, FittedBackupNodeValue],
    selected_node_ids: tuple[str, ...],
    iteration: int,
) -> MorpionSupervisedRows:
    """Build supervised rows with fitted backed-up targets."""
    nodes_by_id = {node.node_id: node for node in snapshot.nodes}
    rows: list[MorpionSupervisedRow] = []
    for node_id in selected_node_ids:
        node = nodes_by_id[node_id]
        if node.state_ref_payload is None:
            continue
        node_value = node_values[node_id]
        rows.append(
            MorpionSupervisedRow(
                node_id=node.node_id,
                state_ref_payload=dict(node.state_ref_payload),
                target_value=node_value.backed_up_target,
                is_terminal=node.is_terminal,
                is_exact=node.is_exact,
                depth=node.depth,
                visit_count=node.visit_count,
                direct_value=node_value.direct_value_before_backup,
                over_event_label=node.over_event_label,
                metadata={
                    **dict(node.metadata),
                    "fitted_backup_iteration": iteration,
                    "target_source": node_value.target_source,
                    "previous_backed_up_target": node_value.previous_backed_up_target,
                    "abs_target_change": node_value.abs_target_change,
                },
            )
        )
    return MorpionSupervisedRows(
        rows=tuple(rows),
        metadata={
            "dataset_kind": "morpion_fitted_backup_sanity_rows",
            "iteration": iteration,
            "num_rows": len(rows),
        },
    )


def _snapshot_with_fitted_backups(
    snapshot: TrainingTreeSnapshot,
    node_values: dict[str, FittedBackupNodeValue],
) -> TrainingTreeSnapshot:
    """Return a diagnostic-only snapshot carrying fitted direct/backed values."""
    nodes = tuple(
        TrainingNodeSnapshot(
            node_id=node.node_id,
            parent_ids=node.parent_ids,
            child_ids=node.child_ids,
            depth=node.depth,
            state_ref_payload=node.state_ref_payload,
            direct_value_scalar=node_values[node.node_id].direct_value_before_backup,
            backed_up_value_scalar=node_values[node.node_id].backed_up_target,
            is_terminal=node.is_terminal,
            is_exact=node.is_exact,
            over_event_label=node.over_event_label,
            visit_count=node.visit_count,
            metadata=dict(node.metadata),
        )
        for node in snapshot.nodes
    )
    return TrainingTreeSnapshot(
        root_node_id=snapshot.root_node_id,
        nodes=nodes,
        metadata=dict(snapshot.metadata),
    )


def _predict_rows(
    model: MorpionRegressor,
    *,
    rows: MorpionSupervisedRows,
    spec: MorpionEvaluatorSpec,
) -> list[float]:
    """Predict values for supervised rows."""
    snapshot = TrainingTreeSnapshot(
        root_node_id="rows",
        nodes=tuple(
            TrainingNodeSnapshot(
                node_id=row.node_id,
                parent_ids=(),
                child_ids=(),
                depth=row.depth,
                state_ref_payload=row.state_ref_payload,
                direct_value_scalar=row.direct_value,
                backed_up_value_scalar=row.target_value,
                is_terminal=row.is_terminal,
                is_exact=row.is_exact,
                over_event_label=row.over_event_label,
                visit_count=row.visit_count,
                metadata=dict(row.metadata),
            )
            for row in rows.rows
        ),
        metadata={},
    )
    predictions_by_id = _predict_snapshot_nodes(snapshot=snapshot, model=model, spec=spec)
    return [predictions_by_id[row.node_id] for row in rows.rows]


def _iteration_summary(
    *,
    iteration: int,
    rows: MorpionSupervisedRows,
    node_values: dict[str, FittedBackupNodeValue],
    selected_node_ids: tuple[str, ...],
    final_loss: float,
    mae_after: float | None,
    max_abs_error_after: float | None,
    previous_predictions: list[float],
    next_predictions: list[float],
    rows_path: Path,
    model_dir: Path,
    diagnostics_path: Path,
    target_diagnostics_path: Path,
) -> dict[str, object]:
    """Build one per-iteration summary."""
    selected_values = [node_values[node_id] for node_id in selected_node_ids]
    targets = [value.backed_up_target for value in selected_values]
    target_changes = [
        value.abs_target_change
        for value in selected_values
        if value.abs_target_change is not None
    ]
    prediction_changes = [
        abs(after - before)
        for before, after in zip(previous_predictions, next_predictions, strict=True)
    ]
    return {
        "iteration": iteration,
        "num_rows": len(rows.rows),
        "exact_or_terminal_count": sum(
            1 for value in selected_values if value.target_source == "ground_truth_exact_or_terminal"
        ),
        "frontier_prediction_count": sum(
            1 for value in selected_values if value.target_source == "frontier_prediction"
        ),
        "child_backup_count": sum(
            1 for value in selected_values if value.target_source == "child_backup"
        ),
        "final_loss": final_loss,
        "mae_after": mae_after,
        "max_abs_error_after": max_abs_error_after,
        "mean_abs_target_change": _mean(target_changes),
        "max_abs_target_change": max(target_changes, default=None),
        "mean_prediction_change": _mean(prediction_changes),
        "mean_backed_up_target": _mean(targets),
        "min_backed_up_target": min(targets, default=None),
        "max_backed_up_target": max(targets, default=None),
        "target_std": _std(targets),
        "rows_path": str(rows_path),
        "model_dir": str(model_dir),
        "diagnostics_path": str(diagnostics_path),
        "target_diagnostics_path": str(target_diagnostics_path),
    }


def _mean(values: list[float]) -> float | None:
    """Return the arithmetic mean."""
    if not values:
        return None
    return sum(values) / float(len(values))


def _std(values: list[float]) -> float | None:
    """Return the population standard deviation."""
    if not values:
        return None
    mean = sum(values) / float(len(values))
    return (sum((value - mean) ** 2 for value in values) / float(len(values))) ** 0.5


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FittedBackupNodeValue",
    "MorpionFittedBackupSanityArgs",
    "fitted_backup_node_values",
    "main",
    "run_fitted_backup_sanity",
]
