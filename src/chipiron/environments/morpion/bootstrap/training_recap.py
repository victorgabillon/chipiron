"""Compact Morpion training recap for cluster worker terminals."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

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


def collect_latest_training_recap(work_dir: Path) -> MorpionTrainingRecap | None:
    """Return the best available compact recap for one Morpion work directory."""
    work_dir = work_dir.expanduser()
    for status_path in _latest_training_status_paths(work_dir):
        recap = _recap_from_training_status(work_dir, status_path)
        if recap is not None:
            return recap

    for manifest_path in _latest_model_manifest_paths(work_dir):
        recap = _recap_from_model_manifest(work_dir, manifest_path)
        if recap is not None:
            return recap

    for manifest_path in _latest_debug_training_manifest_paths(work_dir):
        recap = _recap_from_debug_training_manifest(work_dir, manifest_path)
        if recap is not None:
            return recap

    return _recap_from_training_log(work_dir / "logs" / "training.log", work_dir)


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
    args = parser.parse_args(argv)

    try:
        while True:
            if args.clear:
                print("\033[2J\033[H", end="")
            recap = collect_latest_training_recap(args.work_dir)
            if args.emit_json:
                print(json.dumps(recap_to_dict(recap), sort_keys=True))
            else:
                print(render_training_recap(recap))
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


def _path_to_str(path: Path | None) -> str | None:
    return None if path is None else str(path)


if __name__ == "__main__":
    raise SystemExit(main())
