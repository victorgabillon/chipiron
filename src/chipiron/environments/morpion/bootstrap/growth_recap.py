"""Human-readable Morpion growth recap for cluster operator terminals."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .bootstrap_paths import MorpionBootstrapPaths
from .operator_console import render_operator_panel
from .pipeline_artifacts import MorpionPipelineActiveModel, load_pipeline_active_model
from .run_state import MorpionBootstrapRunState, load_bootstrap_run_state


@dataclass(frozen=True, slots=True)
class MorpionGrowthRecap:
    """Dependency-light growth status snapshot for operator terminals."""

    latest_generation: int | None
    cycle_index: int | None
    active_evaluator_name: str | None
    active_model_generation: int | None
    active_model_source: str | None
    latest_checkpoint_path: str | None
    tree_nodes: int | None
    branch_count: int | None
    last_restore_s: float | None
    last_growth_s: float | None
    last_save_s: float | None
    worker_max_cycles: str | None
    source_paths: tuple[Path, ...]


def collect_growth_recap(
    work_dir: Path,
    *,
    worker_max_cycles: str | None = None,
) -> MorpionGrowthRecap:
    """Collect a tolerant growth recap from persisted bootstrap artifacts."""
    paths = MorpionBootstrapPaths.from_work_dir(work_dir)
    run_state = _load_run_state(paths.run_state_path)
    latest_status = _read_json_mapping(paths.latest_status_path)
    active_model = _load_active_model(paths.pipeline_active_model_path)

    active_evaluator_name = (
        None if active_model is None else active_model.evaluator_name
    ) or (None if run_state is None else run_state.active_evaluator_name)
    latest_checkpoint_path = _latest_checkpoint_path(
        run_state=run_state,
        latest_status=latest_status,
    )
    metadata = {} if run_state is None else run_state.metadata
    tree_metadata = _mapping(metadata.get("tree"))
    checkpoint_metadata = _mapping(metadata.get("checkpoint"))
    restore_metadata = _mapping(metadata.get("restore"))

    source_paths = tuple(
        path
        for path in (
            paths.run_state_path if paths.run_state_path.is_file() else None,
            paths.latest_status_path if paths.latest_status_path.is_file() else None,
            paths.pipeline_active_model_path
            if paths.pipeline_active_model_path.is_file()
            else None,
        )
        if path is not None
    )
    return MorpionGrowthRecap(
        latest_generation=_first_int(
            None if run_state is None else run_state.generation,
            _int_from_mapping(latest_status, "latest_generation"),
        ),
        cycle_index=_first_int(
            None if run_state is None else run_state.cycle_index,
            _int_from_mapping(latest_status, "latest_cycle_index"),
        ),
        active_evaluator_name=active_evaluator_name,
        active_model_generation=None
        if active_model is None
        else active_model.generation,
        active_model_source=None if active_model is None else active_model.source,
        latest_checkpoint_path=latest_checkpoint_path,
        tree_nodes=_first_int(
            None if run_state is None else run_state.tree_size_at_last_save,
            _int_from_mapping(tree_metadata, "node_count"),
        ),
        branch_count=_int_from_mapping(tree_metadata, "branch_count"),
        last_restore_s=_float_from_mapping(restore_metadata, "total_s"),
        last_growth_s=_float_from_mapping(tree_metadata, "growth_elapsed_s"),
        last_save_s=_float_from_mapping(checkpoint_metadata, "total_s"),
        worker_max_cycles=worker_max_cycles,
        source_paths=source_paths,
    )


def render_growth_recap(
    recap: MorpionGrowthRecap,
    *,
    force_plain: bool = False,
    force_rich: bool = False,
) -> str:
    """Render one growth recap for an operator terminal."""
    return render_operator_panel(
        "GROWTH-RECAP",
        _growth_recap_lines(recap),
        style="green",
        force_plain=force_plain,
        force_rich=force_rich,
    )


def _growth_recap_lines(recap: MorpionGrowthRecap) -> list[str]:
    active_generation = _format_value(recap.active_model_generation)
    active_source = _format_value(recap.active_model_source)
    return [
        (
            "ACTIVE EVALUATOR: "
            f"{_format_value(recap.active_evaluator_name)} "
            f"(generation {active_generation}, source {active_source})"
        ),
        (
            "latest_generation="
            f"{_format_value(recap.latest_generation)} "
            f"cycle={_format_value(recap.cycle_index)}"
        ),
        f"latest_checkpoint={_format_value(recap.latest_checkpoint_path)}",
        (
            "tree_nodes="
            f"{_format_value(recap.tree_nodes)} "
            f"branches={_format_value(recap.branch_count)}"
        ),
        (
            "last_restore_s="
            f"{_format_float(recap.last_restore_s)} "
            f"last_growth_s={_format_float(recap.last_growth_s)} "
            f"last_save_s={_format_float(recap.last_save_s)}"
        ),
        f"worker_max_cycles={_format_value(recap.worker_max_cycles)}",
    ]


def _load_run_state(path: Path) -> MorpionBootstrapRunState | None:
    if not path.is_file():
        return None
    try:
        return load_bootstrap_run_state(path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _load_active_model(path: Path) -> MorpionPipelineActiveModel | None:
    if not path.is_file():
        return None
    try:
        return load_pipeline_active_model(path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _read_json_mapping(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, Mapping):
        return {}
    return dict(cast("Mapping[str, object]", payload))


def _latest_checkpoint_path(
    *,
    run_state: MorpionBootstrapRunState | None,
    latest_status: Mapping[str, object],
) -> str | None:
    if run_state is not None and run_state.latest_runtime_checkpoint_path is not None:
        return run_state.latest_runtime_checkpoint_path
    latest_event = _mapping(latest_status.get("latest_event"))
    artifacts = _mapping(latest_event.get("artifacts"))
    checkpoint_path = artifacts.get("runtime_checkpoint_path")
    return checkpoint_path if isinstance(checkpoint_path, str) else None


def _mapping(value: object) -> dict[str, object]:
    return (
        dict(cast("Mapping[str, object]", value)) if isinstance(value, Mapping) else {}
    )


def _first_int(*values: int | None) -> int | None:
    return next((value for value in values if value is not None), None)


def _int_from_mapping(payload: Mapping[str, object], key: str) -> int | None:
    value = payload.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _float_from_mapping(payload: Mapping[str, object], key: str) -> float | None:
    value = payload.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _format_value(value: object | None) -> str:
    return "n/a" if value is None else str(value)


def _format_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def main(argv: list[str] | None = None) -> int:
    """Run the growth recap CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument(
        "--worker-max-cycles",
        default=os.environ.get("MORPION_GROWTH_WORKER_MAX_CYCLES"),
    )
    parser.add_argument("--rich", action="store_true", dest="force_rich")
    parser.add_argument("--no-rich", action="store_true", dest="force_plain")
    args = parser.parse_args(argv)

    recap = collect_growth_recap(
        args.work_dir,
        worker_max_cycles=args.worker_max_cycles,
    )
    print(
        render_growth_recap(
            recap,
            force_plain=args.force_plain,
            force_rich=args.force_rich,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
