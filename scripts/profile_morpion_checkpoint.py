#!/usr/bin/env python3
"""Standalone Morpion checkpoint profiling outside the bootstrap pipeline."""

from __future__ import annotations

import argparse
import cProfile
import json
import logging
import pstats
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


DEFAULT_TARGET_NODES = 50_000
DEFAULT_GROWTH_STEPS_PER_BATCH = 100
DEFAULT_TOP_FUNCTIONS = 80

_T = TypeVar("_T")


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for Morpion checkpoint profiling."""
    parser = argparse.ArgumentParser(
        description=(
            "Profile Morpion Anemone checkpoint payload build and optional save "
            "outside the full bootstrap pipeline."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("load", "grow"),
        required=True,
        help="Load an existing runtime checkpoint or grow a fresh runtime.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="Bootstrap work directory used to resolve checkpoints and artifacts.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint JSON path for load mode. Defaults to the latest runtime checkpoint in work-dir.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Required only when dump-json or dump-zstd is enabled.",
    )
    parser.add_argument(
        "--profile-output",
        type=Path,
        default=Path("morpion_checkpoint_profile.prof"),
        help="Path to the generated cProfile .prof file.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP_FUNCTIONS,
        help="Number of top functions to print for cumulative and tottime stats.",
    )
    parser.add_argument(
        "--target-nodes",
        type=int,
        default=DEFAULT_TARGET_NODES,
        help="Target node count for grow mode.",
    )
    parser.add_argument(
        "--growth-steps-per-batch",
        type=int,
        default=DEFAULT_GROWTH_STEPS_PER_BATCH,
        help="Number of growth steps to execute before printing progress in grow mode.",
    )
    parser.add_argument(
        "--dump-json",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to run dataclasses.asdict and JSON dump timing.",
    )
    parser.add_argument(
        "--dump-zstd",
        action="store_true",
        help="Optionally write a zstd-compressed JSON payload next to --output.",
    )
    profile_mode_group = parser.add_mutually_exclusive_group()
    profile_mode_group.add_argument(
        "--profile-build-only",
        dest="profile_mode",
        action="store_const",
        const="build_only",
        default="build_only",
        help="Profile only build_search_checkpoint_payload(...).",
    )
    profile_mode_group.add_argument(
        "--profile-full-save",
        dest="profile_mode",
        action="store_const",
        const="full_save",
        help="Profile payload build plus optional asdict and dump phases.",
    )
    return parser


def _time_call(function: Callable[[], _T]) -> tuple[_T, float]:
    """Return the function result plus wall-clock elapsed seconds."""
    started_at = perf_counter()
    result = function()
    return result, perf_counter() - started_at


def _resolve_latest_runtime_checkpoint(runtime_checkpoint_dir: Path) -> Path:
    """Return the latest generation checkpoint from one runtime checkpoint dir."""
    latest_generation: int | None = None
    latest_path: Path | None = None
    for path in runtime_checkpoint_dir.glob("generation_*.json"):
        generation = _parse_generation_json_name(path)
        if generation is None:
            continue
        if latest_generation is None or generation > latest_generation:
            latest_generation = generation
            latest_path = path
    if latest_path is None:
        raise FileNotFoundError(
            f"No runtime checkpoint found in {runtime_checkpoint_dir!s}."
        )
    return latest_path


def _parse_generation_json_name(path: Path) -> int | None:
    """Parse generation_XXXXXX.json names into an integer generation index."""
    if path.suffix != ".json":
        return None
    stem = path.stem
    if not stem.startswith("generation_"):
        return None
    generation_text = stem.removeprefix("generation_")
    if not generation_text.isdigit():
        return None
    return int(generation_text)


def _configure_logging() -> None:
    """Enable plain INFO logs when the caller has not configured logging."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def _load_runtime_modules() -> dict[str, Any]:
    """Import runtime modules lazily so parser tests stay lightweight."""
    import anemone
    import atomheart
    import chipiron
    from anemone.checkpoints import build_search_checkpoint_payload
    from chipiron.environments.morpion.bootstrap import (
        AnemoneMorpionSearchRunner,
        MorpionBootstrapPaths,
    )
    import chipiron.environments.morpion.bootstrap.anemone_runner as runner_module

    return {
        "anemone": anemone,
        "atomheart": atomheart,
        "build_search_checkpoint_payload": build_search_checkpoint_payload,
        "chipiron": chipiron,
        "MorpionBootstrapPaths": MorpionBootstrapPaths,
        "AnemoneMorpionSearchRunner": AnemoneMorpionSearchRunner,
        "runner_module": runner_module,
    }


def _print_module_paths(modules: dict[str, Any]) -> None:
    """Print module file paths to confirm the script is profiling the right code."""
    print(f"anemone: {getattr(modules['anemone'], '__file__', None)}")
    print(f"chipiron: {getattr(modules['chipiron'], '__file__', None)}")
    print(f"atomheart: {getattr(modules['atomheart'], '__file__', None)}")


def _require_output_path(args: argparse.Namespace) -> Path:
    """Return the required output path for dump modes or fail clearly."""
    if args.output is None:
        raise ValueError("--output is required when --dump-json or --dump-zstd is used.")
    return Path(args.output)


def _load_runner_for_mode(
    args: argparse.Namespace,
    modules: dict[str, Any],
) -> tuple[Any, Path | None]:
    """Return a runner initialized in load or grow mode."""
    runner = modules["AnemoneMorpionSearchRunner"]()
    bootstrap_paths = modules["MorpionBootstrapPaths"].from_work_dir(args.work_dir)

    if args.mode == "load":
        if args.checkpoint is None:
            checkpoint_path = _resolve_latest_runtime_checkpoint(
                bootstrap_paths.runtime_checkpoint_dir
            )
        else:
            checkpoint_path = Path(args.checkpoint).resolve()
        print(f"[profile] mode=load checkpoint={checkpoint_path}")
        runner.load_or_create(checkpoint_path, None)
        return runner, checkpoint_path

    if args.checkpoint is not None:
        raise ValueError("--checkpoint is only valid in --mode load.")

    bootstrap_paths.ensure_directories()
    print(
        "[profile] mode=grow work_dir=%s target_nodes=%s growth_steps_per_batch=%s"
        % (
            bootstrap_paths.work_dir,
            args.target_nodes,
            args.growth_steps_per_batch,
        )
    )
    runner.load_or_create(None, None)
    _grow_runner_until_target(
        runner=runner,
        target_nodes=args.target_nodes,
        growth_steps_per_batch=args.growth_steps_per_batch,
    )
    return runner, None


def _grow_runner_until_target(
    *,
    runner: Any,
    target_nodes: int,
    growth_steps_per_batch: int,
) -> None:
    """Grow the live runtime in batches until reaching the target node count."""
    if target_nodes < 1:
        raise ValueError("--target-nodes must be >= 1.")
    if growth_steps_per_batch < 1:
        raise ValueError("--growth-steps-per-batch must be >= 1.")

    started_at = perf_counter()
    previous_node_count = runner.current_tree_size()
    while previous_node_count < target_nodes:
        runner.grow(growth_steps_per_batch)
        current_node_count = runner.current_tree_size()
        runtime = runner._require_runtime()
        tree = getattr(runtime, "tree", None)
        branch_count = getattr(tree, "branch_count", None)
        print(
            "[grow] nodes=%s branches=%s elapsed_s=%.6f"
            % (
                current_node_count,
                branch_count if isinstance(branch_count, int) else "unknown",
                perf_counter() - started_at,
            )
        )
        if current_node_count <= previous_node_count:
            raise RuntimeError(
                "Grow mode stopped increasing node count before reaching target_nodes."
            )
        previous_node_count = current_node_count


def _profile_checkpoint_save(
    *,
    args: argparse.Namespace,
    modules: dict[str, Any],
    runner: Any,
) -> None:
    """Profile checkpoint payload build and optional serialization phases."""
    runner_module = modules["runner_module"]
    build_search_checkpoint_payload = modules["build_search_checkpoint_payload"]

    runtime = runner._require_runtime()
    state_codec = runner._state_codec
    runtime_tree = getattr(runtime, "tree", None)
    runtime_branch_count = getattr(runtime_tree, "branch_count", None)
    print(
        "[profile] runtime nodes=%s branches=%s"
        % (
            runner.current_tree_size(),
            runtime_branch_count if isinstance(runtime_branch_count, int) else "unknown",
        )
    )

    if args.dump_json or args.dump_zstd:
        output_path = _require_output_path(args)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None

    profile_output = Path(args.profile_output)
    profile_output.parent.mkdir(parents=True, exist_ok=True)
    profiler = cProfile.Profile()
    payload_dict: dict[str, Any] | None = None
    payload: Any
    payload_build_s: float
    asdict_s: float | None = None
    json_dump_s: float | None = None
    json_dump_bytes: int | None = None
    zstd_dump_s: float | None = None
    zstd_dump_bytes: int | None = None

    rss_before_mb = runner_module._current_rss_mb()
    total_started_at = perf_counter()

    if args.profile_mode == "full_save":
        profiler.enable()
    if args.profile_mode == "build_only":
        profiler.enable()
    payload, payload_build_s = _time_call(
        lambda: build_search_checkpoint_payload(runtime, state_codec=state_codec)
    )
    if args.profile_mode == "build_only":
        profiler.disable()
    rss_after_payload_build_mb = runner_module._current_rss_mb()
    print(f"[profile] phase=payload_build elapsed_s={payload_build_s:.6f}")

    needs_payload_dict = bool(args.dump_json or args.dump_zstd)
    if needs_payload_dict:
        payload_dict, asdict_s = _time_call(lambda: asdict(payload))
        rss_after_asdict_mb = runner_module._current_rss_mb()
        print(f"[profile] phase=asdict elapsed_s={asdict_s:.6f}")
    else:
        rss_after_asdict_mb = None
        print("[profile] phase=asdict skipped=true")

    if args.dump_json:
        assert output_path is not None
        json_dump_bytes, json_dump_s = _time_call(
            lambda: _dump_json_payload(payload_dict, output_path)
        )
        print(
            "[profile] phase=json_dump elapsed_s=%.6f bytes=%s output=%s"
            % (json_dump_s, json_dump_bytes, output_path)
        )
    else:
        print("[profile] phase=json_dump skipped=true")

    rss_after_json_dump_mb = runner_module._current_rss_mb()

    if args.dump_zstd:
        assert output_path is not None
        zstd_result, zstd_dump_s = _time_call(
            lambda: _dump_zstd_payload(payload_dict, output_path)
        )
        if zstd_result is None:
            print("[profile] phase=zstd_dump skipped=true reason=zstandard_not_installed")
        else:
            zstd_path, zstd_dump_bytes = zstd_result
            print(
                "[profile] phase=zstd_dump elapsed_s=%.6f bytes=%s output=%s"
                % (zstd_dump_s, zstd_dump_bytes, zstd_path)
            )
    else:
        print("[profile] phase=zstd_dump skipped=true")

    if args.profile_mode == "full_save":
        profiler.disable()

    total_s = perf_counter() - total_started_at
    rss_after_total_mb = runner_module._current_rss_mb()
    print(f"[profile] phase=total elapsed_s={total_s:.6f}")
    print(
        "[profile-memory] rss_before_mb=%s rss_after_payload_build_mb=%s rss_after_asdict_mb=%s rss_after_json_dump_mb=%s rss_after_total_mb=%s"
        % (
            _format_optional_number(rss_before_mb),
            _format_optional_number(rss_after_payload_build_mb),
            _format_optional_number(rss_after_asdict_mb),
            _format_optional_number(rss_after_json_dump_mb),
            _format_optional_number(rss_after_total_mb),
        )
    )

    profiler.dump_stats(str(profile_output))
    print(f"[profile] phase=cprofile_dump output={profile_output}")
    _print_cprofile_stats(profiler, args.top)

    node_count, anchor_count, delta_count = runner_module._checkpoint_node_counts(payload)
    runner_module._log_checkpoint_metrics(
        "profile",
        runner_module.CheckpointIoMetrics(
            path=str(output_path) if output_path is not None else "none",
            bytes=json_dump_bytes,
            payload_build_s=payload_build_s,
            asdict_s=asdict_s,
            json_dump_s=json_dump_s,
            total_s=total_s,
            rss_before_mb=rss_before_mb,
            rss_after_mb=rss_after_total_mb,
            node_count=node_count,
            anchor_count=anchor_count,
            delta_count=delta_count,
        ),
    )
    selector_fields = runner_module._checkpoint_selector_state_fields(
        payload,
        prefix="checkpoint",
    )
    print(
        "[profile-checkpoint] nodes=%s branches=%s anchors=%s deltas=%s checkpoint_selector_state_present=%s checkpoint_selector_state_type=%s checkpoint_selector_state_version=%s"
        % (
            node_count,
            runtime_branch_count if isinstance(runtime_branch_count, int) else "unknown",
            anchor_count,
            delta_count,
            selector_fields["checkpoint_selector_state_present"],
            selector_fields["checkpoint_selector_state_type"],
            selector_fields["checkpoint_selector_state_version"],
        )
    )


def _dump_json_payload(payload_dict: dict[str, Any] | None, output_path: Path) -> int:
    """Write the payload dict as JSON and return the number of bytes written."""
    if payload_dict is None:
        raise ValueError("Payload dict is required before JSON dumping.")
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload_dict, handle, indent=2, sort_keys=True)
    return output_path.stat().st_size


def _dump_zstd_payload(
    payload_dict: dict[str, Any] | None,
    output_path: Path,
) -> tuple[Path, int] | None:
    """Write the payload dict as zstd-compressed JSON when zstandard is available."""
    if payload_dict is None:
        raise ValueError("Payload dict is required before zstd dumping.")
    try:
        import zstandard
    except ImportError:
        return None

    json_bytes = json.dumps(payload_dict, indent=2, sort_keys=True).encode("utf-8")
    zstd_path = _zstd_output_path(output_path)
    compressor = zstandard.ZstdCompressor()
    with zstd_path.open("wb") as handle:
        handle.write(compressor.compress(json_bytes))
    return zstd_path, zstd_path.stat().st_size


def _zstd_output_path(output_path: Path) -> Path:
    """Return the zstd output path derived from the optional JSON output path."""
    if output_path.suffix:
        return output_path.with_suffix(f"{output_path.suffix}.zst")
    return output_path.with_name(f"{output_path.name}.zst")


def _print_cprofile_stats(profiler: cProfile.Profile, top: int) -> None:
    """Print the top cumulative and total-time cProfile entries."""
    print(f"[profile] stats sort=cumulative top={top}")
    pstats.Stats(profiler).sort_stats("cumulative").print_stats(top)
    print(f"[profile] stats sort=tottime top={top}")
    pstats.Stats(profiler).sort_stats("tottime").print_stats(top)


def _format_optional_number(value: object) -> str:
    """Format optional numeric values for stable profiling output."""
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the standalone Morpion checkpoint profiler."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging()
    modules = _load_runtime_modules()
    _print_module_paths(modules)
    runner, _checkpoint_path = _load_runner_for_mode(args, modules)
    _profile_checkpoint_save(args=args, modules=modules, runner=runner)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())