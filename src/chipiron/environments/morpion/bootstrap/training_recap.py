"""Compact Morpion training recap for cluster worker terminals."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .operator_console import render_operator_panel, use_rich_output
from .pipeline.training_recovery import (
    DEFAULT_STALE_GRACE_SECONDS,
    StaleTrainingState,
    inspect_training_state_for_recovery,
)

_GENERATION_DIR_RE = re.compile(r"generation_(\d+)")
_TRAINING_LOG_KEY_RE = re.compile(
    r"(?P<key>generation|evaluator|name|train_loss|validation_loss)=(?P<value>\S+)"
)


@dataclass(frozen=True)
class MorpionTrainingRecap:
    """Latest dependency-light Morpion training status snapshot."""

    generation: int | None
    evaluator_name: str | None
    status: str
    train_loss: float | None
    validation_loss: float | None
    validation_quality_r2_vs_mean_baseline: float | None
    validation_quality_pearson_correlation: float | None
    graph_output_tanh: bool | None
    graph_token_cache_used: bool | None
    cached_global_shuffle: bool | None
    model_bundle_path: Path | None
    source_path: Path | None
    updated_at: str | None


@dataclass(frozen=True)
class MorpionEvaluatorTrainingRecap:
    """One evaluator row in a Morpion training recap table."""

    evaluator_name: str
    status: str | None
    train_loss: float | None
    validation_loss: float | None
    validation_quality_r2_vs_mean_baseline: float | None
    validation_quality_pearson_correlation: float | None
    graph_output_tanh: bool | None
    graph_token_cache_used: bool | None
    cached_global_shuffle: bool | None
    model_bundle_path: Path | None


@dataclass(frozen=True)
class MorpionTrainingRecapTable:
    """Latest dependency-light Morpion training recap for all evaluators."""

    generation: int | None
    status: str
    selected_evaluator_name: str | None
    updated_at: str | None
    source_path: Path | None
    rows: tuple[MorpionEvaluatorTrainingRecap, ...]


def collect_latest_training_recap(work_dir: Path) -> MorpionTrainingRecap | None:
    """Return the best available compact recap for one Morpion work directory."""
    table = collect_latest_training_recap_table(work_dir)
    if table is not None:
        return _single_recap_from_table(table)

    work_dir = work_dir.expanduser()
    return _recap_from_training_log(work_dir / "logs" / "training.log", work_dir)


def collect_latest_training_recap_table(
    work_dir: Path,
) -> MorpionTrainingRecapTable | None:
    """Return the best available all-evaluator recap for one work directory."""
    work_dir = work_dir.expanduser()
    for status_path in _latest_training_status_paths(work_dir):
        table = _recap_table_from_training_status(work_dir, status_path)
        if table is not None:
            return table

    for manifest_path in _latest_model_manifest_paths(work_dir):
        table = _recap_table_from_model_manifest(work_dir, manifest_path)
        if table is not None:
            return table

    for manifest_path in _latest_debug_training_manifest_paths(work_dir):
        table = _recap_table_from_debug_training_manifest(work_dir, manifest_path)
        if table is not None:
            return table

    log_recap = _recap_from_training_log(work_dir / "logs" / "training.log", work_dir)
    if log_recap is None:
        return None
    return _recap_table_from_single(log_recap)


def collect_latest_stale_training_state(
    work_dir: Path,
    *,
    stale_grace_seconds: int = DEFAULT_STALE_GRACE_SECONDS,
) -> StaleTrainingState | None:
    """Return the latest stale training state, if a generation is blocked."""
    now_utc = datetime.now(UTC)
    for generation_dir in _latest_generation_dirs(work_dir.expanduser()):
        state = inspect_training_state_for_recovery(
            generation_dir,
            now_utc=now_utc,
            stale_grace_seconds=stale_grace_seconds,
        )
        if state.is_stale:
            return state
    return None


def render_training_recap(recap: MorpionTrainingRecap | None) -> str:
    """Render a compact training recap for a worker terminal."""
    if recap is None:
        return "[TRAINING-RECAP] no training recap available"

    header = (
        "[TRAINING-RECAP] "
        f"generation={_format_value(recap.generation)} "
        f"evaluator={_format_value(recap.evaluator_name)} "
        f"status={_format_value(recap.status)}"
    )
    if recap.updated_at is not None:
        header += f" updated={recap.updated_at}"

    lines = [
        header,
        "  "
        f"train_loss={_format_float(recap.train_loss)} "
        f"validation_loss={_format_float(recap.validation_loss)} "
        "validation_r2="
        f"{_format_float(recap.validation_quality_r2_vs_mean_baseline)} "
        "validation_pearson="
        f"{_format_float(recap.validation_quality_pearson_correlation)}",
    ]
    flag_parts = [
        _format_flag("graph_output_tanh", recap.graph_output_tanh),
        _format_flag("graph_token_cache_used", recap.graph_token_cache_used),
        _format_flag("cached_global_shuffle", recap.cached_global_shuffle),
    ]
    rendered_flags = [part for part in flag_parts if part is not None]
    if rendered_flags:
        lines.append("  " + " ".join(rendered_flags))
    if recap.model_bundle_path is not None:
        lines.append(f"  model={recap.model_bundle_path}")
    return "\n".join(lines)


def render_training_recap_table(table: MorpionTrainingRecapTable | None) -> str:
    """Render a compact all-evaluator recap table for a terminal."""
    if table is None:
        return "[TRAINING-RECAP] no training recap available"

    header = (
        "[TRAINING-RECAP] "
        f"generation={_format_value(table.generation)} "
        f"status={_format_value(table.status)} "
        f"selected={_format_value(table.selected_evaluator_name)}"
    )
    if table.updated_at is not None:
        header += f" updated={table.updated_at}"

    if not table.rows:
        return "\n".join([header, "  no evaluator loss rows available"])

    lines = [
        header,
        "  evaluator                         train_loss  val_loss  "
        "r2      pearson  tanh   graph_cache  shuffle",
    ]
    for row in table.rows:
        marker = "*" if row.evaluator_name == table.selected_evaluator_name else " "
        lines.append(
            f"{marker} {row.evaluator_name:<32.32} "
            f"{_format_float(row.train_loss):>10} "
            f"{_format_float(row.validation_loss):>9} "
            f"{_format_float(row.validation_quality_r2_vs_mean_baseline):>7} "
            f"{_format_float(row.validation_quality_pearson_correlation):>8} "
            f"{_format_bool(row.graph_output_tanh):>6} "
            f"{_format_bool(row.graph_token_cache_used):>11} "
            f"{_format_bool(row.cached_global_shuffle):>8}"
        )

    selected_row = _selected_row(table)
    if selected_row is not None and selected_row.model_bundle_path is not None:
        lines.append(f"  selected_model={selected_row.model_bundle_path}")
    if table.source_path is not None:
        lines.append(f"  source={table.source_path}")
    return "\n".join(lines)


def render_training_recap_table_operator(
    table: MorpionTrainingRecapTable | None,
    *,
    force_plain: bool = False,
    force_rich: bool = False,
) -> str:
    """Render the all-evaluator recap with operator-facing Rich support."""
    if table is None:
        return render_operator_panel(
            "Morpion Training Recap",
            ["no training recap available"],
            style="magenta",
            force_plain=force_plain,
            force_rich=force_rich,
        )
    lines = [
        (
            "generation "
            f"{_format_value(table.generation)} | "
            f"selected {_format_value(table.selected_evaluator_name)} | "
            f"status {_format_value(table.status)}"
        ),
        "evaluator                         val_loss  train_loss  r2      pearson  flags",
    ]
    for row in table.rows:
        marker = "*" if row.evaluator_name == table.selected_evaluator_name else " "
        flags = " ".join(
            part
            for part in (
                f"tanh={_format_bool(row.graph_output_tanh)}",
                f"cache={_format_bool(row.graph_token_cache_used)}",
                f"shuffle={_format_bool(row.cached_global_shuffle)}",
            )
            if part is not None
        )
        lines.append(
            f"{marker} {row.evaluator_name:<32.32} "
            f"{_format_float(row.validation_loss):>8} "
            f"{_format_float(row.train_loss):>10} "
            f"{_format_float(row.validation_quality_r2_vs_mean_baseline):>7} "
            f"{_format_float(row.validation_quality_pearson_correlation):>8} "
            f"{flags}"
        )
    return render_operator_panel(
        "Morpion Training Recap",
        lines,
        style="magenta",
        force_plain=force_plain,
        force_rich=force_rich,
    )


def render_stale_training_warning(
    state: StaleTrainingState,
    *,
    work_dir: Path,
) -> str:
    """Render a terminal warning for stale training state."""
    claim_state = (
        "expired"
        if state.claim_expired
        else ("missing" if state.claim_path is None else "active")
    )
    return "\n".join((
        "TRAINING BLOCKED / STALE",
        f"generation={state.generation}",
        f"dataset_status={_format_value(state.dataset_status)}",
        (f"manifest_training_status={_format_value(state.manifest_training_status)}"),
        f"claim={claim_state}",
        f"evaluator_results={_format_value(state.evaluator_results_count)}",
        f"reason={state.reason}",
        (
            "suggested: python -m "
            "chipiron.environments.morpion.bootstrap.training_recovery "
            f"--work-dir {work_dir} --generation {state.generation} --recover"
        ),
    ))


def recap_to_dict(recap: MorpionTrainingRecap | None) -> dict[str, object] | None:
    """Convert one recap to JSON-friendly data."""
    if recap is None:
        return None
    return {
        "generation": recap.generation,
        "evaluator_name": recap.evaluator_name,
        "status": recap.status,
        "train_loss": recap.train_loss,
        "validation_loss": recap.validation_loss,
        "validation_quality_r2_vs_mean_baseline": (
            recap.validation_quality_r2_vs_mean_baseline
        ),
        "validation_quality_pearson_correlation": (
            recap.validation_quality_pearson_correlation
        ),
        "graph_output_tanh": recap.graph_output_tanh,
        "graph_token_cache_used": recap.graph_token_cache_used,
        "cached_global_shuffle": recap.cached_global_shuffle,
        "model_bundle_path": _path_to_str(recap.model_bundle_path),
        "source_path": _path_to_str(recap.source_path),
        "updated_at": recap.updated_at,
    }


def recap_table_to_dict(
    table: MorpionTrainingRecapTable | None,
) -> dict[str, object] | None:
    """Convert one all-evaluator recap table to JSON-friendly data."""
    if table is None:
        return None
    return {
        "generation": table.generation,
        "status": table.status,
        "selected_evaluator_name": table.selected_evaluator_name,
        "updated_at": table.updated_at,
        "source_path": _path_to_str(table.source_path),
        "rows": [
            {
                "evaluator_name": row.evaluator_name,
                "status": row.status,
                "train_loss": row.train_loss,
                "validation_loss": row.validation_loss,
                "validation_quality_r2_vs_mean_baseline": (
                    row.validation_quality_r2_vs_mean_baseline
                ),
                "validation_quality_pearson_correlation": (
                    row.validation_quality_pearson_correlation
                ),
                "graph_output_tanh": row.graph_output_tanh,
                "graph_token_cache_used": row.graph_token_cache_used,
                "cached_global_shuffle": row.cached_global_shuffle,
                "model_bundle_path": _path_to_str(row.model_bundle_path),
            }
            for row in table.rows
        ],
    }


def find_key(obj: object, key: str) -> object | None:
    """Recursively return the first value for ``key`` in JSON-like data."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = find_key(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_key(value, key)
            if found is not None:
                return found
    return None


def main(argv: list[str] | None = None) -> int:
    """Run the training recap CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--watch", type=float, default=None)
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--json", action="store_true", dest="emit_json")
    rich_mode = parser.add_mutually_exclusive_group()
    rich_mode.add_argument("--rich", action="store_true", dest="force_rich")
    rich_mode.add_argument("--no-rich", action="store_true", dest="force_plain")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--all-evaluators", action="store_false", dest="single")
    mode.add_argument("--single", action="store_true")
    parser.set_defaults(single=False, force_rich=False, force_plain=False)
    args = parser.parse_args(argv)

    try:
        while True:
            if args.clear:
                print("\033[2J\033[H", end="")
            if args.single:
                recap = collect_latest_training_recap(args.work_dir)
                output = recap_to_dict(recap) if args.emit_json else None
                rendered = render_training_recap(recap)
            else:
                table = collect_latest_training_recap_table(args.work_dir)
                output = recap_table_to_dict(table) if args.emit_json else None
                stale_state = (
                    None
                    if args.emit_json
                    else collect_latest_stale_training_state(args.work_dir)
                )
                if stale_state is not None:
                    completed_table = _latest_completed_training_recap_table(
                        args.work_dir
                    )
                    rendered = render_stale_training_warning(
                        stale_state,
                        work_dir=args.work_dir,
                    )
                    if completed_table is not None:
                        rendered += (
                            "\n\nLast completed training:\n"
                            + render_training_recap_table(completed_table)
                        )
                else:
                    rendered = (
                        render_training_recap_table_operator(
                            table,
                            force_plain=args.force_plain,
                            force_rich=args.force_rich,
                        )
                        if use_rich_output(
                            force_plain=args.force_plain,
                            force_rich=args.force_rich,
                        )
                        else render_training_recap_table(table)
                    )
            if args.emit_json:
                print(json.dumps(output, sort_keys=True))
            else:
                print(rendered)
            if args.watch is None:
                return 0
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 130


def _latest_training_status_paths(work_dir: Path) -> list[Path]:
    status_paths = list(
        (work_dir / "pipeline").glob("generation_*/training_status.json")
    )
    return sorted(
        status_paths,
        key=lambda path: (_generation_from_path(path) or -1, _path_mtime_ns(path)),
        reverse=True,
    )


def _latest_generation_dirs(work_dir: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in (work_dir / "pipeline").glob("generation_*")
            if path.is_dir()
        ),
        key=lambda path: (_generation_from_path(path) or -1, _path_mtime_ns(path)),
        reverse=True,
    )


def _latest_completed_training_recap_table(
    work_dir: Path,
) -> MorpionTrainingRecapTable | None:
    work_dir = work_dir.expanduser()
    for status_path in _latest_training_status_paths(work_dir):
        table = _recap_table_from_training_status(work_dir, status_path)
        if table is not None and table.status == "done" and table.rows:
            return table
    return None


def _latest_model_manifest_paths(work_dir: Path) -> list[Path]:
    manifest_paths = list(
        (work_dir / "models").glob("generation_*/*/morpion_manifest.json")
    )
    return sorted(
        manifest_paths,
        key=lambda path: (_generation_from_path(path) or -1, _path_mtime_ns(path)),
        reverse=True,
    )


def _latest_debug_training_manifest_paths(work_dir: Path) -> list[Path]:
    manifest_paths: list[Path] = []
    for base_path in (work_dir / "debug", work_dir / "training"):
        if base_path.is_dir():
            manifest_paths.extend(base_path.glob("**/*training*manifest*.json"))
            manifest_paths.extend(base_path.glob("**/*training*metadata*.json"))
    return sorted(
        manifest_paths,
        key=lambda path: (_generation_from_path(path) or -1, _path_mtime_ns(path)),
        reverse=True,
    )


def _recap_table_from_training_status(
    work_dir: Path,
    status_path: Path,
) -> MorpionTrainingRecapTable | None:
    payload = _read_json_file(status_path)
    if not isinstance(payload, dict):
        return None

    evaluator_results = payload.get("evaluator_results")
    if not isinstance(evaluator_results, dict):
        return None

    generation = _coerce_int(payload.get("generation")) or _generation_from_path(
        status_path
    )
    status = _coerce_str(payload.get("status")) or _coerce_str(
        payload.get("training_status")
    )
    rows = tuple(
        sorted(
            (
                _row_from_evaluator_result(
                    work_dir=work_dir,
                    evaluator_name=str(evaluator_name),
                    result_payload=result_payload,
                    status=status,
                    fallback_payload=payload,
                )
                for evaluator_name, result_payload in evaluator_results.items()
            ),
            key=_row_sort_key,
        )
    )
    return MorpionTrainingRecapTable(
        generation=generation,
        status=status or "unknown",
        selected_evaluator_name=_coerce_str(payload.get("selected_evaluator_name")),
        updated_at=_coerce_str(payload.get("updated_at_utc"))
        or _coerce_str(payload.get("updated_at")),
        source_path=_relative_to_work_dir(status_path, work_dir),
        rows=rows,
    )


def _recap_table_from_model_manifest(
    work_dir: Path,
    manifest_path: Path,
) -> MorpionTrainingRecapTable | None:
    recap = _recap_from_model_manifest(work_dir, manifest_path)
    if recap is None:
        return None
    return _recap_table_from_single(recap)


def _recap_table_from_debug_training_manifest(
    work_dir: Path,
    manifest_path: Path,
) -> MorpionTrainingRecapTable | None:
    recap = _recap_from_debug_training_manifest(work_dir, manifest_path)
    if recap is None:
        return None
    return _recap_table_from_single(recap)


def _recap_table_from_single(
    recap: MorpionTrainingRecap,
) -> MorpionTrainingRecapTable:
    row = MorpionEvaluatorTrainingRecap(
        evaluator_name=recap.evaluator_name or "n/a",
        status=recap.status,
        train_loss=recap.train_loss,
        validation_loss=recap.validation_loss,
        validation_quality_r2_vs_mean_baseline=(
            recap.validation_quality_r2_vs_mean_baseline
        ),
        validation_quality_pearson_correlation=(
            recap.validation_quality_pearson_correlation
        ),
        graph_output_tanh=recap.graph_output_tanh,
        graph_token_cache_used=recap.graph_token_cache_used,
        cached_global_shuffle=recap.cached_global_shuffle,
        model_bundle_path=recap.model_bundle_path,
    )
    return MorpionTrainingRecapTable(
        generation=recap.generation,
        status=recap.status,
        selected_evaluator_name=recap.evaluator_name,
        updated_at=recap.updated_at,
        source_path=recap.source_path,
        rows=(row,),
    )


def _single_recap_from_table(table: MorpionTrainingRecapTable) -> MorpionTrainingRecap:
    selected_row = _selected_row(table)
    if selected_row is None and table.rows:
        selected_row = table.rows[0]
    if selected_row is None:
        return MorpionTrainingRecap(
            generation=table.generation,
            evaluator_name=table.selected_evaluator_name,
            status=table.status,
            train_loss=None,
            validation_loss=None,
            validation_quality_r2_vs_mean_baseline=None,
            validation_quality_pearson_correlation=None,
            graph_output_tanh=None,
            graph_token_cache_used=None,
            cached_global_shuffle=None,
            model_bundle_path=None,
            source_path=table.source_path,
            updated_at=table.updated_at,
        )
    return MorpionTrainingRecap(
        generation=table.generation,
        evaluator_name=selected_row.evaluator_name,
        status=table.status,
        train_loss=selected_row.train_loss,
        validation_loss=selected_row.validation_loss,
        validation_quality_r2_vs_mean_baseline=(
            selected_row.validation_quality_r2_vs_mean_baseline
        ),
        validation_quality_pearson_correlation=(
            selected_row.validation_quality_pearson_correlation
        ),
        graph_output_tanh=selected_row.graph_output_tanh,
        graph_token_cache_used=selected_row.graph_token_cache_used,
        cached_global_shuffle=selected_row.cached_global_shuffle,
        model_bundle_path=selected_row.model_bundle_path,
        source_path=table.source_path,
        updated_at=table.updated_at,
    )


def _row_from_evaluator_result(
    *,
    work_dir: Path,
    evaluator_name: str,
    result_payload: object,
    status: str | None,
    fallback_payload: object,
) -> MorpionEvaluatorTrainingRecap:
    model_path = _coerce_model_path(
        _find_first([result_payload, fallback_payload], "model_bundle_path"),
        work_dir=work_dir,
    )
    manifest_payload, args_payload = _load_model_metadata(work_dir, model_path)
    sources: list[object] = [
        result_payload,
        manifest_payload,
        args_payload,
        fallback_payload,
    ]
    validation_loss = _coerce_float(_find_first(sources, "validation_loss"))
    if validation_loss is None:
        validation_loss = _coerce_float(_find_first(sources, "final_loss"))
    return MorpionEvaluatorTrainingRecap(
        evaluator_name=evaluator_name,
        status=status,
        train_loss=_coerce_float(_find_first(sources, "train_loss")),
        validation_loss=validation_loss,
        validation_quality_r2_vs_mean_baseline=_validation_quality_metric(
            sources,
            "r2_vs_mean_baseline",
            "validation_quality_r2_vs_mean_baseline",
        ),
        validation_quality_pearson_correlation=_validation_quality_metric(
            sources,
            "pearson_correlation",
            "validation_quality_pearson_correlation",
        ),
        graph_output_tanh=_coerce_bool(_find_first(sources, "graph_output_tanh")),
        graph_token_cache_used=_coerce_bool(
            _find_first(sources, "graph_token_cache_used")
        ),
        cached_global_shuffle=_coerce_bool(
            _find_first(sources, "cached_global_shuffle")
        ),
        model_bundle_path=model_path,
    )


def _row_sort_key(row: MorpionEvaluatorTrainingRecap) -> tuple[bool, float, str]:
    return (
        row.validation_loss is None,
        math.inf if row.validation_loss is None else row.validation_loss,
        row.evaluator_name,
    )


def _selected_row(
    table: MorpionTrainingRecapTable,
) -> MorpionEvaluatorTrainingRecap | None:
    if table.selected_evaluator_name is None:
        return None
    for row in table.rows:
        if row.evaluator_name == table.selected_evaluator_name:
            return row
    return None


def _recap_from_training_status(
    work_dir: Path,
    status_path: Path,
) -> MorpionTrainingRecap | None:
    payload = _read_json_file(status_path)
    if not isinstance(payload, dict):
        return None

    generation = _coerce_int(payload.get("generation")) or _generation_from_path(
        status_path
    )
    status = _coerce_str(payload.get("status")) or _coerce_str(
        payload.get("training_status")
    )
    evaluator_name = _coerce_str(payload.get("selected_evaluator_name"))
    evaluator_results = payload.get("evaluator_results")
    selected_result = _select_evaluator_result(evaluator_results, evaluator_name)
    if evaluator_name is None and selected_result is not None:
        evaluator_name = selected_result[0]
    result_payload = selected_result[1] if selected_result is not None else {}

    model_path = _coerce_model_path(
        _find_first([result_payload, payload], "model_bundle_path"),
        work_dir=work_dir,
    )
    manifest_payload, args_payload = _load_model_metadata(work_dir, model_path)
    sources: list[object] = [result_payload, manifest_payload, args_payload, payload]

    return MorpionTrainingRecap(
        generation=generation,
        evaluator_name=evaluator_name
        or _coerce_str(find_key(payload, "evaluator_name")),
        status=status or "unknown",
        train_loss=_coerce_float(_find_first(sources, "train_loss")),
        validation_loss=_coerce_float(_find_first(sources, "validation_loss")),
        validation_quality_r2_vs_mean_baseline=_validation_quality_metric(
            sources,
            "r2_vs_mean_baseline",
            "validation_quality_r2_vs_mean_baseline",
        ),
        validation_quality_pearson_correlation=_validation_quality_metric(
            sources,
            "pearson_correlation",
            "validation_quality_pearson_correlation",
        ),
        graph_output_tanh=_coerce_bool(_find_first(sources, "graph_output_tanh")),
        graph_token_cache_used=_coerce_bool(
            _find_first(sources, "graph_token_cache_used")
        ),
        cached_global_shuffle=_coerce_bool(
            _find_first(sources, "cached_global_shuffle")
        ),
        model_bundle_path=model_path,
        source_path=_relative_to_work_dir(status_path, work_dir),
        updated_at=_coerce_str(payload.get("updated_at_utc"))
        or _coerce_str(payload.get("updated_at")),
    )


def _recap_from_model_manifest(
    work_dir: Path,
    manifest_path: Path,
) -> MorpionTrainingRecap | None:
    manifest_payload = _read_json_file(manifest_path)
    if not isinstance(manifest_payload, dict):
        return None
    model_dir = manifest_path.parent
    model_path = _relative_to_work_dir(model_dir, work_dir)
    args_payload = _read_json_file(model_dir / "morpion_regressor_args.json")
    if not isinstance(args_payload, dict):
        args_payload = {}
    sources: list[object] = [manifest_payload, args_payload]
    return MorpionTrainingRecap(
        generation=_generation_from_path(manifest_path),
        evaluator_name=model_dir.name,
        status="model_manifest",
        train_loss=_coerce_float(_find_first(sources, "train_loss")),
        validation_loss=_coerce_float(_find_first(sources, "validation_loss")),
        validation_quality_r2_vs_mean_baseline=_validation_quality_metric(
            sources,
            "r2_vs_mean_baseline",
            "validation_quality_r2_vs_mean_baseline",
        ),
        validation_quality_pearson_correlation=_validation_quality_metric(
            sources,
            "pearson_correlation",
            "validation_quality_pearson_correlation",
        ),
        graph_output_tanh=_coerce_bool(_find_first(sources, "graph_output_tanh")),
        graph_token_cache_used=_coerce_bool(
            _find_first(sources, "graph_token_cache_used")
        ),
        cached_global_shuffle=_coerce_bool(
            _find_first(sources, "cached_global_shuffle")
        ),
        model_bundle_path=model_path,
        source_path=_relative_to_work_dir(manifest_path, work_dir),
        updated_at=_coerce_str(_find_first(sources, "updated_at_utc"))
        or _coerce_str(_find_first(sources, "updated_at")),
    )


def _recap_from_debug_training_manifest(
    work_dir: Path,
    manifest_path: Path,
) -> MorpionTrainingRecap | None:
    payload = _read_json_file(manifest_path)
    if not isinstance(payload, dict):
        return None
    model_path = _coerce_model_path(
        find_key(payload, "model_bundle_path"), work_dir=work_dir
    )
    sources: list[object] = [payload]
    return MorpionTrainingRecap(
        generation=_coerce_int(find_key(payload, "generation"))
        or _generation_from_path(manifest_path),
        evaluator_name=_coerce_str(find_key(payload, "evaluator_name")),
        status=_coerce_str(find_key(payload, "status")) or "training_manifest",
        train_loss=_coerce_float(find_key(payload, "train_loss")),
        validation_loss=_coerce_float(find_key(payload, "validation_loss")),
        validation_quality_r2_vs_mean_baseline=_validation_quality_metric(
            sources,
            "r2_vs_mean_baseline",
            "validation_quality_r2_vs_mean_baseline",
        ),
        validation_quality_pearson_correlation=_validation_quality_metric(
            sources,
            "pearson_correlation",
            "validation_quality_pearson_correlation",
        ),
        graph_output_tanh=_coerce_bool(find_key(payload, "graph_output_tanh")),
        graph_token_cache_used=_coerce_bool(
            find_key(payload, "graph_token_cache_used")
        ),
        cached_global_shuffle=_coerce_bool(find_key(payload, "cached_global_shuffle")),
        model_bundle_path=model_path,
        source_path=_relative_to_work_dir(manifest_path, work_dir),
        updated_at=_coerce_str(find_key(payload, "updated_at_utc"))
        or _coerce_str(find_key(payload, "updated_at")),
    )


def _recap_from_training_log(
    training_log_path: Path,
    work_dir: Path,
) -> MorpionTrainingRecap | None:
    try:
        lines = training_log_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
    except OSError:
        return None
    for line in reversed(lines[-2000:]):
        if "train_loss=" not in line and "validation_loss=" not in line:
            continue
        values = {
            match.group("key"): match.group("value")
            for match in _TRAINING_LOG_KEY_RE.finditer(line)
        }
        if not values:
            continue
        return MorpionTrainingRecap(
            generation=_coerce_int(values.get("generation")),
            evaluator_name=values.get("evaluator") or values.get("name"),
            status="log",
            train_loss=_coerce_float(values.get("train_loss")),
            validation_loss=_coerce_float(values.get("validation_loss")),
            validation_quality_r2_vs_mean_baseline=None,
            validation_quality_pearson_correlation=None,
            graph_output_tanh=None,
            graph_token_cache_used=None,
            cached_global_shuffle=None,
            model_bundle_path=None,
            source_path=_relative_to_work_dir(training_log_path, work_dir),
            updated_at=None,
        )
    return None


def _load_model_metadata(
    work_dir: Path,
    model_path: Path | None,
) -> tuple[dict[str, object], dict[str, object]]:
    if model_path is None:
        return {}, {}
    model_dir = model_path if model_path.is_absolute() else work_dir / model_path
    manifest_payload = _read_json_file(model_dir / "morpion_manifest.json")
    args_payload = _read_json_file(model_dir / "morpion_regressor_args.json")
    return (
        manifest_payload if isinstance(manifest_payload, dict) else {},
        args_payload if isinstance(args_payload, dict) else {},
    )


def _select_evaluator_result(
    evaluator_results: object,
    evaluator_name: str | None,
) -> tuple[str, object] | None:
    if not isinstance(evaluator_results, dict):
        return None
    if evaluator_name is not None:
        result = evaluator_results.get(evaluator_name)
        if result is not None:
            return evaluator_name, result

    scored_results: list[tuple[float, str, object]] = []
    unscored_results: list[tuple[str, object]] = []
    for name, result in evaluator_results.items():
        name_text = str(name)
        score = _coerce_float(_find_first([result], "validation_loss"))
        if score is None:
            score = _coerce_float(_find_first([result], "final_loss"))
        if score is None:
            unscored_results.append((name_text, result))
        else:
            scored_results.append((score, name_text, result))
    if scored_results:
        _, name, result = min(scored_results, key=lambda item: item[0])
        return name, result
    if unscored_results:
        return sorted(unscored_results, key=lambda item: item[0])[0]
    return None


def _validation_quality_metric(
    sources: list[object],
    nested_key: str,
    direct_key: str,
) -> float | None:
    direct = _coerce_float(_find_first(sources, direct_key))
    if direct is not None:
        return direct
    for source in sources:
        regression_quality = find_key(source, "regression_quality")
        if not isinstance(regression_quality, dict):
            continue
        validation = regression_quality.get("validation")
        if isinstance(validation, dict):
            metric = _coerce_float(validation.get(nested_key))
            if metric is not None:
                return metric
    return None


def _find_first(sources: list[object], key: str) -> object | None:
    for source in sources:
        found = find_key(source, key)
        if found is not None:
            return found
    return None


def _read_json_file(path: Path) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _coerce_model_path(value: object | None, *, work_dir: Path) -> Path | None:
    text = _coerce_str(value)
    if text is None:
        return None
    return _relative_to_work_dir(Path(text), work_dir)


def _relative_to_work_dir(path: Path, work_dir: Path) -> Path:
    try:
        return path.resolve().relative_to(work_dir.resolve())
    except (OSError, ValueError):
        return path


def _generation_from_path(path: Path) -> int | None:
    for part in path.parts:
        match = _GENERATION_DIR_RE.fullmatch(part)
        if match is not None:
            return int(match.group(1))
    return None


def _path_mtime_ns(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


def _coerce_int(value: object | None) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _coerce_float(value: object | None) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        result = float(value)
    elif isinstance(value, str):
        try:
            result = float(value)
        except ValueError:
            return None
    else:
        return None
    return result if math.isfinite(result) else None


def _coerce_bool(value: object | None) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y"}:
            return True
        if normalized in {"0", "false", "no", "n"}:
            return False
    return None


def _coerce_str(value: object | None) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _format_value(value: object | None) -> str:
    return "n/a" if value is None else str(value)


def _format_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4g}"


def _format_flag(name: str, value: bool | None) -> str | None:
    if value is None:
        return None
    return f"{name}={str(value).lower()}"


def _format_bool(value: bool | None) -> str:
    return "n/a" if value is None else str(value).lower()


def _path_to_str(path: Path | None) -> str | None:
    return None if path is None else str(path)


if __name__ == "__main__":
    raise SystemExit(main())
