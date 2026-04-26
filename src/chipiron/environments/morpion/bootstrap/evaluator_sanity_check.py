"""Standalone sanity checks for Morpion bootstrap evaluator learning."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from anemone.training_export import (
    TrainingNodeSnapshot,
    TrainingTreeSnapshot,
    load_training_tree_snapshot,
)

from chipiron.environments.morpion.learning import (
    MorpionSupervisedRow,
    MorpionSupervisedRows,
    save_morpion_supervised_rows,
    training_node_to_morpion_supervised_row,
    training_tree_snapshot_to_morpion_supervised_rows,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.train import (
    MorpionTrainingArgs,
    train_morpion_regressor,
)

from .bootstrap_loop import MorpionBootstrapPaths
from .evaluator_diagnostics import (
    build_evaluator_training_diagnostics,
    load_previous_evaluator_for_diagnostics,
    save_evaluator_training_diagnostics,
)
from .evaluator_family import (
    CANONICAL_MORPION_EVALUATOR_FAMILY_PRESET,
    morpion_evaluators_config_from_preset,
)

if TYPE_CHECKING:
    from pathlib import Path

    from .bootstrap_loop import MorpionEvaluatorSpec

LOGGER = logging.getLogger(__name__)

MorpionSanityDatasetMode = Literal[
    "bootstrap_like",
    "terminal_path",
    "top_terminal_paths",
]


class MissingMorpionSanitySnapshotError(FileNotFoundError):
    """Raised when no usable tree export can be found."""

    def __init__(self, work_dir: Path, generation: int | None) -> None:
        """Initialize the missing-snapshot error."""
        generation_text = "latest" if generation is None else str(generation)
        super().__init__(
            "No Morpion bootstrap tree export found in "
            f"{work_dir}/tree_exports for generation {generation_text}."
        )


class MissingMorpionSanityTargetError(ValueError):
    """Raised when a selected snapshot node has no numeric supervised target."""

    def __init__(self, node_id: str) -> None:
        """Initialize the missing-target error."""
        super().__init__(
            "Morpion evaluator sanity selected node "
            f"{node_id!r}, but it has neither backed_up_value_scalar nor "
            "direct_value_scalar."
        )


class MissingMorpionSanityStatePayloadError(ValueError):
    """Raised when a selected snapshot node has no Morpion state payload."""

    def __init__(self, node_id: str) -> None:
        """Initialize the missing-state-payload error."""
        super().__init__(
            "Morpion evaluator sanity selected node "
            f"{node_id!r}, but it has no state_ref_payload."
        )


class NoMorpionSanityTerminalNodeError(ValueError):
    """Raised when a path dataset cannot find any terminal/exact node."""

    def __init__(self) -> None:
        """Initialize the missing-terminal-node error."""
        super().__init__(
            "Morpion evaluator sanity path dataset requires at least one terminal "
            "or exact node in the snapshot."
        )


class EmptyMorpionSanityDatasetError(ValueError):
    """Raised when extraction produces no supervised sanity rows."""

    def __init__(self) -> None:
        """Initialize the empty-dataset error."""
        super().__init__("Morpion evaluator sanity dataset is empty.")


class UnknownMorpionSanityDatasetModeError(ValueError):
    """Raised when the requested dataset mode is not supported."""

    def __init__(self, dataset_mode: str) -> None:
        """Initialize the unknown-dataset-mode error."""
        super().__init__(f"Unknown Morpion sanity dataset mode: {dataset_mode!r}.")


class UnknownMorpionSanityEvaluatorError(ValueError):
    """Raised when the requested evaluator is absent from the configured family."""

    def __init__(self, evaluator_name: str) -> None:
        """Initialize the unknown-evaluator error."""
        super().__init__(
            f"Morpion evaluator sanity requested unknown evaluator {evaluator_name!r}."
        )


@dataclass(frozen=True, slots=True)
class MorpionEvaluatorSanityArgs:
    """Arguments for one standalone evaluator sanity-check run."""

    work_dir: str | Path
    generation: int | None = None
    dataset_mode: MorpionSanityDatasetMode = "terminal_path"
    max_rows: int | None = None
    num_epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 1e-3
    evaluator_name: str | None = None
    evaluator_family_preset: str = CANONICAL_MORPION_EVALUATOR_FAMILY_PRESET
    run_name: str | None = None
    top_terminal_path_count: int = 5
    require_exact_or_terminal: bool = False
    min_depth: int | None = None
    min_visit_count: int | None = None
    use_backed_up_value: bool = True
    shuffle: bool = True


def run_evaluator_sanity_check(args: MorpionEvaluatorSanityArgs) -> dict[str, object]:
    """Run the standalone sanity-check pipeline and return the summary payload."""
    paths = MorpionBootstrapPaths.from_work_dir(args.work_dir)
    output_dir = _sanity_output_dir(
        paths.work_dir,
        generation=args.generation,
        run_name=args.run_name,
    )
    rows_path = output_dir / "rows.json"
    models_dir = output_dir / "models"
    diagnostics_dir = output_dir / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("[sanity] snapshot_load_start")
    snapshot_path = _resolve_snapshot_path(
        paths=paths,
        generation=args.generation,
    )
    snapshot = load_training_tree_snapshot(snapshot_path)
    LOGGER.info(
        "[sanity] snapshot_load_done path=%s nodes=%s",
        str(snapshot_path),
        len(snapshot.nodes),
    )

    LOGGER.info("[sanity] dataset_extract_start mode=%s", args.dataset_mode)
    rows = build_sanity_dataset_rows(
        snapshot=snapshot,
        dataset_mode=args.dataset_mode,
        max_rows=args.max_rows,
        top_terminal_path_count=args.top_terminal_path_count,
        require_exact_or_terminal=args.require_exact_or_terminal,
        min_depth=args.min_depth,
        min_visit_count=args.min_visit_count,
        use_backed_up_value=args.use_backed_up_value,
        metadata={
            "sanity_generation": args.generation,
            "sanity_snapshot_path": str(snapshot_path),
        },
    )
    save_morpion_supervised_rows(rows, rows_path)
    LOGGER.info(
        "[sanity] dataset_extract_done rows=%s path=%s",
        len(rows.rows),
        str(rows_path),
    )
    if not rows.rows:
        raise EmptyMorpionSanityDatasetError

    created_at = datetime.now(UTC).isoformat()
    evaluator_summaries: dict[str, dict[str, object]] = {}
    for evaluator_name, spec in _selected_evaluator_specs(args).items():
        model_bundle_path = models_dir / evaluator_name
        diagnostics_path = diagnostics_dir / f"{evaluator_name}.json"
        previous_model = load_previous_evaluator_for_diagnostics(
            _previous_model_bundle_path(
                paths=paths,
                generation=args.generation,
                evaluator_name=evaluator_name,
            )
        )

        LOGGER.info("[sanity] train_start evaluator=%s", evaluator_name)
        trained_model, metrics = train_morpion_regressor(
            MorpionTrainingArgs(
                dataset_file=rows_path,
                output_dir=model_bundle_path,
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
            "[sanity] train_done evaluator=%s final_loss=%s",
            evaluator_name,
            metrics.get("final_loss"),
        )

        diagnostics = build_evaluator_training_diagnostics(
            generation=args.generation or 0,
            evaluator_name=evaluator_name,
            rows=rows,
            created_at=created_at,
            feature_subset_name=spec.feature_subset_name,
            feature_names=spec.feature_names,
            model_before=previous_model,
            model_after=trained_model,
        )
        save_evaluator_training_diagnostics(diagnostics, diagnostics_path)
        LOGGER.info(
            "[sanity] diagnostics_saved evaluator=%s path=%s",
            evaluator_name,
            str(diagnostics_path),
        )

        evaluator_summaries[evaluator_name] = {
            "final_loss": float(metrics["final_loss"]),
            "num_epochs": int(metrics["num_epochs"]),
            "num_samples": int(metrics["num_samples"]),
            "mae_before": diagnostics.mae_before,
            "mae_after": diagnostics.mae_after,
            "max_abs_error_before": diagnostics.max_abs_error_before,
            "max_abs_error_after": diagnostics.max_abs_error_after,
            "model_bundle_path": str(model_bundle_path),
            "diagnostics_path": str(diagnostics_path),
        }

    summary: dict[str, object] = {
        "created_at": created_at,
        "work_dir": str(paths.work_dir),
        "generation": args.generation,
        "run_name": output_dir.name,
        "dataset_mode": args.dataset_mode,
        "snapshot_path": str(snapshot_path),
        "rows_path": str(rows_path),
        "num_rows": len(rows.rows),
        "evaluators": evaluator_summaries,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("[sanity] summary_saved path=%s", str(summary_path))
    return summary


def build_sanity_dataset_rows(
    *,
    snapshot: TrainingTreeSnapshot,
    dataset_mode: MorpionSanityDatasetMode,
    max_rows: int | None = None,
    top_terminal_path_count: int = 5,
    require_exact_or_terminal: bool = False,
    min_depth: int | None = None,
    min_visit_count: int | None = None,
    use_backed_up_value: bool = True,
    metadata: dict[str, object] | None = None,
) -> MorpionSupervisedRows:
    """Build fixed supervised rows from one snapshot using a sanity dataset mode."""
    if dataset_mode == "bootstrap_like":
        return training_tree_snapshot_to_morpion_supervised_rows(
            snapshot,
            require_exact_or_terminal=require_exact_or_terminal,
            min_depth=min_depth,
            min_visit_count=min_visit_count,
            max_rows=max_rows,
            use_backed_up_value=use_backed_up_value,
            metadata={"sanity_dataset_mode": dataset_mode, **(metadata or {})},
        )
    if dataset_mode == "terminal_path":
        selected_nodes = terminal_path_nodes(snapshot)
    elif dataset_mode == "top_terminal_paths":
        selected_nodes = top_terminal_path_nodes(
            snapshot,
            max_terminal_nodes=top_terminal_path_count,
        )
    else:
        raise UnknownMorpionSanityDatasetModeError(dataset_mode)

    if max_rows is not None:
        selected_nodes = selected_nodes[:max_rows]
    rows = tuple(
        _path_node_to_row(
            node,
            dataset_mode=dataset_mode,
            use_backed_up_value=use_backed_up_value,
        )
        for node in selected_nodes
    )
    return MorpionSupervisedRows(
        rows=rows,
        metadata={
            "sanity_dataset_mode": dataset_mode,
            "source_root_node_id": snapshot.root_node_id,
            "max_rows": max_rows,
            "top_terminal_path_count": top_terminal_path_count,
            "num_rows": len(rows),
            **(metadata or {}),
        },
    )


def terminal_path_nodes(snapshot: TrainingTreeSnapshot) -> tuple[TrainingNodeSnapshot, ...]:
    """Return the root-to-node chain for the deepest terminal/exact snapshot node."""
    terminal_nodes = _terminal_or_exact_nodes(snapshot)
    if not terminal_nodes:
        raise NoMorpionSanityTerminalNodeError
    selected = sorted(terminal_nodes, key=lambda node: (-node.depth, node.node_id))[0]
    return _ancestor_chain(snapshot, selected)


def top_terminal_path_nodes(
    snapshot: TrainingTreeSnapshot,
    *,
    max_terminal_nodes: int,
) -> tuple[TrainingNodeSnapshot, ...]:
    """Return deduplicated ancestors for the deepest terminal/exact nodes."""
    terminal_nodes = sorted(
        _terminal_or_exact_nodes(snapshot),
        key=lambda node: (-node.depth, node.node_id),
    )
    if not terminal_nodes:
        raise NoMorpionSanityTerminalNodeError

    selected_nodes: list[TrainingNodeSnapshot] = []
    seen_node_ids: set[str] = set()
    for terminal_node in terminal_nodes[:max_terminal_nodes]:
        for node in _ancestor_chain(snapshot, terminal_node):
            if node.node_id in seen_node_ids:
                continue
            selected_nodes.append(node)
            seen_node_ids.add(node.node_id)
    return tuple(selected_nodes)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for ``python -m ...evaluator_sanity_check``."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parsed = _parse_args(argv)
    run_evaluator_sanity_check(parsed)
    return 0


def _parse_args(argv: list[str] | None) -> MorpionEvaluatorSanityArgs:
    """Parse CLI arguments into a typed sanity-check argument object."""
    parser = argparse.ArgumentParser(
        description="Train Morpion bootstrap evaluators on a fixed sanity dataset."
    )
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--generation", type=int)
    parser.add_argument(
        "--dataset-mode",
        choices=("terminal_path", "top_terminal_paths", "bootstrap_like"),
        default="terminal_path",
    )
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--evaluator-name")
    parser.add_argument(
        "--evaluator-family-preset",
        default=CANONICAL_MORPION_EVALUATOR_FAMILY_PRESET,
    )
    parser.add_argument("--run-name")
    parser.add_argument("--top-terminal-path-count", type=int, default=5)
    parser.add_argument("--require-exact-or-terminal", action="store_true")
    parser.add_argument("--min-depth", type=int)
    parser.add_argument("--min-visit-count", type=int)
    parser.add_argument("--use-direct-value", action="store_true")
    parser.add_argument("--no-shuffle", action="store_true")
    namespace = parser.parse_args(argv)
    return MorpionEvaluatorSanityArgs(
        work_dir=namespace.work_dir,
        generation=namespace.generation,
        dataset_mode=namespace.dataset_mode,
        max_rows=namespace.max_rows,
        num_epochs=namespace.num_epochs,
        batch_size=namespace.batch_size,
        learning_rate=namespace.learning_rate,
        evaluator_name=namespace.evaluator_name,
        evaluator_family_preset=namespace.evaluator_family_preset,
        run_name=namespace.run_name,
        top_terminal_path_count=namespace.top_terminal_path_count,
        require_exact_or_terminal=namespace.require_exact_or_terminal,
        min_depth=namespace.min_depth,
        min_visit_count=namespace.min_visit_count,
        use_backed_up_value=not namespace.use_direct_value,
        shuffle=not namespace.no_shuffle,
    )


def _sanity_output_dir(
    work_dir: Path,
    *,
    generation: int | None,
    run_name: str | None,
) -> Path:
    """Return the artifact directory for one sanity run."""
    if run_name is not None:
        return work_dir / "evaluator_sanity" / run_name
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    if generation is None:
        return work_dir / "evaluator_sanity" / timestamp
    return work_dir / "evaluator_sanity" / f"generation_{generation:06d}_{timestamp}"


def _resolve_snapshot_path(
    *,
    paths: MorpionBootstrapPaths,
    generation: int | None,
) -> Path:
    """Return an existing tree-export path for one generation."""
    tree_snapshot_path = _generation_artifact_path(
        directory=paths.tree_snapshot_dir,
        generation=generation,
        suffix=".json",
    )
    if tree_snapshot_path is not None:
        return tree_snapshot_path

    raise MissingMorpionSanitySnapshotError(paths.work_dir, generation)


def _generation_artifact_path(
    *,
    directory: Path,
    generation: int | None,
    suffix: str,
) -> Path | None:
    """Return the requested or latest generation artifact in one directory."""
    if generation is not None:
        path = directory / f"generation_{generation:06d}{suffix}"
        return path if path.is_file() else None
    candidates = [
        path
        for path in directory.glob(f"generation_*{suffix}")
        if _parse_generation_index(path, suffix=suffix) is not None
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda path: int(_parse_generation_index(path, suffix=suffix) or 0),
    )


def _parse_generation_index(path: Path, *, suffix: str) -> int | None:
    """Parse a ``generation_XXXXXX`` artifact path."""
    if path.suffix != suffix:
        return None
    stem = path.stem
    prefix = "generation_"
    if not stem.startswith(prefix):
        return None
    generation_text = stem.removeprefix(prefix)
    if not generation_text.isdigit():
        return None
    return int(generation_text)


def _selected_evaluator_specs(
    args: MorpionEvaluatorSanityArgs,
) -> dict[str, MorpionEvaluatorSpec]:
    """Return evaluator specs selected by CLI arguments."""
    config = morpion_evaluators_config_from_preset(args.evaluator_family_preset)
    if args.evaluator_name is None:
        return dict(config.evaluators)
    if args.evaluator_name not in config.evaluators:
        raise UnknownMorpionSanityEvaluatorError(args.evaluator_name)
    return {args.evaluator_name: config.evaluators[args.evaluator_name]}


def _previous_model_bundle_path(
    *,
    paths: MorpionBootstrapPaths,
    generation: int | None,
    evaluator_name: str,
) -> Path | None:
    """Return the best available bootstrap model bundle for before diagnostics."""
    if generation is not None:
        candidate = paths.model_bundle_path_for_generation(generation, evaluator_name)
        return candidate if candidate.is_dir() else None

    generation_dirs = sorted(
        path
        for path in paths.model_dir.glob("generation_*")
        if path.is_dir()
        and _parse_generation_index(path.with_suffix(".json"), suffix=".json")
        is not None
    )
    for generation_dir in reversed(generation_dirs):
        candidate = generation_dir / evaluator_name
        if candidate.is_dir():
            return candidate
    return None


def _terminal_or_exact_nodes(
    snapshot: TrainingTreeSnapshot,
) -> tuple[TrainingNodeSnapshot, ...]:
    """Return all nodes that have terminal or exact values."""
    return tuple(node for node in snapshot.nodes if node.is_terminal or node.is_exact)


def _ancestor_chain(
    snapshot: TrainingTreeSnapshot,
    terminal_node: TrainingNodeSnapshot,
) -> tuple[TrainingNodeSnapshot, ...]:
    """Return one parent chain in root-to-terminal order."""
    nodes_by_id = {node.node_id: node for node in snapshot.nodes}
    chain: list[TrainingNodeSnapshot] = [terminal_node]
    seen_node_ids = {terminal_node.node_id}
    current = terminal_node
    while current.parent_ids:
        parent_candidates = [
            nodes_by_id[parent_id]
            for parent_id in current.parent_ids
            if parent_id in nodes_by_id
        ]
        if not parent_candidates:
            break
        parent = sorted(
            parent_candidates,
            key=lambda node: (-node.depth, node.node_id),
        )[0]
        if parent.node_id in seen_node_ids:
            break
        chain.append(parent)
        seen_node_ids.add(parent.node_id)
        current = parent
    chain.reverse()
    return tuple(chain)


def _path_node_to_row(
    node: TrainingNodeSnapshot,
    *,
    dataset_mode: MorpionSanityDatasetMode,
    use_backed_up_value: bool,
) -> MorpionSupervisedRow:
    """Convert one path-selected snapshot node into a supervised row."""
    if node.state_ref_payload is None:
        raise MissingMorpionSanityStatePayloadError(node.node_id)
    if node.backed_up_value_scalar is None and node.direct_value_scalar is None:
        raise MissingMorpionSanityTargetError(node.node_id)
    resolved_use_backed_up_value = (
        node.backed_up_value_scalar is not None
        if use_backed_up_value
        else node.direct_value_scalar is None
    )

    row = training_node_to_morpion_supervised_row(
        node,
        use_backed_up_value=resolved_use_backed_up_value,
    )
    if row is None:
        raise MissingMorpionSanityTargetError(node.node_id)

    target_source = (
        "backed_up_value_scalar"
        if resolved_use_backed_up_value
        else "direct_value_scalar"
    )
    return MorpionSupervisedRow(
        node_id=row.node_id,
        state_ref_payload=row.state_ref_payload,
        target_value=row.target_value,
        is_terminal=row.is_terminal,
        is_exact=row.is_exact,
        depth=row.depth,
        visit_count=row.visit_count,
        direct_value=row.direct_value,
        over_event_label=row.over_event_label,
        metadata={
            **dict(row.metadata),
            "sanity_dataset_mode": dataset_mode,
            "target_source": target_source,
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EmptyMorpionSanityDatasetError",
    "MissingMorpionSanitySnapshotError",
    "MissingMorpionSanityStatePayloadError",
    "MissingMorpionSanityTargetError",
    "MorpionEvaluatorSanityArgs",
    "MorpionSanityDatasetMode",
    "NoMorpionSanityTerminalNodeError",
    "UnknownMorpionSanityDatasetModeError",
    "UnknownMorpionSanityEvaluatorError",
    "build_sanity_dataset_rows",
    "main",
    "run_evaluator_sanity_check",
    "terminal_path_nodes",
    "top_terminal_path_nodes",
]
