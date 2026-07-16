#!/usr/bin/env python3
"""Standalone Morpion checkpoint profiling outside the bootstrap pipeline."""

from __future__ import annotations

import argparse
import cProfile
import logging
import pstats
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, TypeVar

from anemone.checkpoints import (
    DEFAULT_CHECKPOINT_FILE_FORMAT,
    checkpoint_cli_name,
    checkpoint_format_from_cli_name,
    checkpoint_output_path,
    resolve_latest_generation_checkpoint_path,
    write_checkpoint_json_payload,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


DEFAULT_TARGET_NODES = 50_000
DEFAULT_GROWTH_STEPS_PER_BATCH = 100
DEFAULT_TOP_FUNCTIONS = 80
DEFAULT_CHECKPOINT_FORMAT = checkpoint_cli_name(DEFAULT_CHECKPOINT_FILE_FORMAT)

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
        help="Checkpoint path for load mode. Defaults to the latest runtime checkpoint in work-dir.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional checkpoint output path. Required only when --dump-json is enabled.",
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
        help="Whether to serialize and write a checkpoint payload to --output.",
    )
    parser.add_argument(
        "--dump-zstd",
        action="store_true",
        help="Deprecated alias for --checkpoint-format json-zst plus --dump-json.",
    )
    parser.add_argument(
        "--checkpoint-format",
        choices=("json", "json-gz", "json-zst"),
        default=DEFAULT_CHECKPOINT_FORMAT,
        help="Output format used when --dump-json is enabled.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Python logging level for the standalone profiler process.",
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
        help="Profile payload build plus optional payload-to-JSON and write phases.",
    )
    return parser


def _time_call[T](function: Callable[[], _T]) -> tuple[_T, float]:
    """Return the function result plus wall-clock elapsed seconds."""
    started_at = perf_counter()
    result = function()
    return result, perf_counter() - started_at


def _resolve_latest_runtime_checkpoint(runtime_checkpoint_dir: Path) -> Path:
    """Return the latest generation checkpoint from one runtime checkpoint dir."""
    return resolve_latest_generation_checkpoint_path(runtime_checkpoint_dir)


def _configure_logging(log_level: str) -> None:
    """Enable plain INFO logs when the caller has not configured logging."""
    logging.basicConfig(level=getattr(logging, log_level), format="%(message)s")


def _load_runtime_modules() -> dict[str, Any]:
    """Import runtime modules lazily so parser tests stay lightweight."""
    import anemone
    import atomheart
    from anemone.checkpoints import build_search_checkpoint_payload

    import chipiron
    import chipiron.environments.morpion.bootstrap.runtime.checkpoint_io as checkpoint_io_module
    import chipiron.environments.morpion.bootstrap.runtime.restore_memory_logging as restore_memory_logging_module
    import chipiron.environments.morpion.bootstrap.runtime.runner as runner_module
    import chipiron.environments.morpion.bootstrap.runtime.selection_logging as selection_logging_module
    from chipiron.environments.morpion.bootstrap import (
        AnemoneMorpionSearchRunner,
        MorpionBootstrapPaths,
    )

    return {
        "anemone": anemone,
        "atomheart": atomheart,
        "build_search_checkpoint_payload": build_search_checkpoint_payload,
        "chipiron": chipiron,
        "MorpionBootstrapPaths": MorpionBootstrapPaths,
        "AnemoneMorpionSearchRunner": AnemoneMorpionSearchRunner,
        "checkpoint_io_module": checkpoint_io_module,
        "restore_memory_logging_module": restore_memory_logging_module,
        "runner_module": runner_module,
        "selection_logging_module": selection_logging_module,
    }


def _print_module_paths(modules: dict[str, Any]) -> None:
    """Print module file paths to confirm the script is profiling the right code."""
    print(f"anemone: {getattr(modules['anemone'], '__file__', None)}")
    print(f"chipiron: {getattr(modules['chipiron'], '__file__', None)}")
    print(f"atomheart: {getattr(modules['atomheart'], '__file__', None)}")


def _require_output_path(args: argparse.Namespace) -> Path:
    """Return the required output path for dump modes or fail clearly."""
    if args.output is None:
        raise ValueError("--output is required when --dump-json is enabled.")
    return checkpoint_output_path(
        args.output,
        file_format=checkpoint_format_from_cli_name(args.checkpoint_format),
    )


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
        runner.load_or_create(
            tree_snapshot_path=checkpoint_path,
            model_bundle_path=None,
        )
        return runner, checkpoint_path

    if args.checkpoint is not None:
        raise ValueError("--checkpoint is only valid in --mode load.")

    bootstrap_paths.ensure_directories()
    print(
        f"[profile] mode=grow work_dir={bootstrap_paths.work_dir} target_nodes={args.target_nodes} growth_steps_per_batch={args.growth_steps_per_batch}"
    )
    runner.load_or_create(
        tree_snapshot_path=None,
        model_bundle_path=None,
    )
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
            "[grow] nodes={} branches={} elapsed_s={:.6f}".format(
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
    checkpoint_io_module = modules["checkpoint_io_module"]
    restore_memory_logging_module = modules["restore_memory_logging_module"]
    selection_logging_module = modules["selection_logging_module"]
    build_search_checkpoint_payload = modules["build_search_checkpoint_payload"]

    runtime = runner._require_runtime()
    state_codec = runner._state_codec
    runtime_tree = getattr(runtime, "tree", None)
    runtime_branch_count = getattr(runtime_tree, "branch_count", None)
    print(
        "[profile] runtime nodes={} branches={}".format(
            runner.current_tree_size(),
            runtime_branch_count
            if isinstance(runtime_branch_count, int)
            else "unknown",
        )
    )

    if args.dump_zstd:
        args.dump_json = True
        args.checkpoint_format = "json-zst"

    if args.dump_json:
        output_path = _require_output_path(args)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None

    profile_output = Path(args.profile_output)
    profile_output.parent.mkdir(parents=True, exist_ok=True)
    profiler = cProfile.Profile()
    payload: Any
    payload_build_s: float
    jsonable_s: float | None = None
    output_bytes: int | None = None
    json_encode_s: float | None = None
    compress_s: float | None = None
    write_s: float | None = None
    uncompressed_bytes: int | None = None
    compression_ratio: float | None = None
    output_format: str | None = None
    output_encoder: str | None = None

    rss_before_mb = restore_memory_logging_module.current_rss_mb()
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
    rss_after_payload_build_mb = restore_memory_logging_module.current_rss_mb()
    print(f"[profile] phase=payload_build elapsed_s={payload_build_s:.6f}")

    if args.dump_json:
        assert output_path is not None
        write_stats, write_elapsed_s = _time_call(
            lambda: write_checkpoint_json_payload(payload, output_path)
        )
        output_bytes = write_stats.compressed_bytes
        output_encoder = write_stats.encoder
        jsonable_s = write_stats.jsonable_s
        json_encode_s = write_stats.json_encode_s
        compress_s = write_stats.compress_s
        write_s = write_stats.write_s
        uncompressed_bytes = write_stats.uncompressed_bytes
        compression_ratio = write_stats.compression_ratio
        output_format = write_stats.file_format
        rss_after_asdict_mb = restore_memory_logging_module.current_rss_mb()
        if jsonable_s is None:
            print("[profile] phase=payload_to_jsonable skipped=true")
        else:
            print(f"[profile] phase=payload_to_jsonable elapsed_s={jsonable_s:.6f}")
        print(
            f"[profile] phase=checkpoint_write elapsed_s={write_elapsed_s:.6f} format={write_stats.file_format} encoder={write_stats.encoder} json_encode_s={write_stats.json_encode_s:.6f} compress_s={_format_optional_number(write_stats.compress_s)} write_s={_format_optional_number(write_stats.write_s)} bytes={write_stats.compressed_bytes} uncompressed_bytes={write_stats.uncompressed_bytes} compression_ratio={_format_optional_number(write_stats.compression_ratio)} output={write_stats.output_path}"
        )
    else:
        rss_after_asdict_mb = None
        print("[profile] phase=payload_to_jsonable skipped=true")
        print("[profile] phase=checkpoint_write skipped=true")

    rss_after_json_dump_mb = restore_memory_logging_module.current_rss_mb()

    if args.profile_mode == "full_save":
        profiler.disable()

    total_s = perf_counter() - total_started_at
    rss_after_total_mb = restore_memory_logging_module.current_rss_mb()
    print(f"[profile] phase=total elapsed_s={total_s:.6f}")
    print(
        f"[profile-memory] rss_before_mb={_format_optional_number(rss_before_mb)} rss_after_payload_build_mb={_format_optional_number(rss_after_payload_build_mb)} rss_after_asdict_mb={_format_optional_number(rss_after_asdict_mb)} rss_after_json_dump_mb={_format_optional_number(rss_after_json_dump_mb)} rss_after_total_mb={_format_optional_number(rss_after_total_mb)}"
    )

    profiler.dump_stats(str(profile_output))
    print(f"[profile] phase=cprofile_dump output={profile_output}")
    _print_cprofile_stats(profiler, args.top)

    node_count, anchor_count, delta_count = (
        checkpoint_io_module._checkpoint_node_counts(payload)
    )
    checkpoint_io_module._log_checkpoint_metrics(
        "profile",
        checkpoint_io_module.CheckpointIoMetrics(
            path=str(output_path) if output_path is not None else "none",
            bytes=output_bytes,
            file_format=output_format,
            encoder=output_encoder,
            payload_build_s=payload_build_s,
            jsonable_s=jsonable_s,
            json_encode_s=json_encode_s,
            compress_s=compress_s,
            write_s=write_s,
            total_s=total_s,
            uncompressed_bytes=uncompressed_bytes,
            compression_ratio=compression_ratio,
            rss_before_mb=rss_before_mb,
            rss_after_mb=rss_after_total_mb,
            node_count=node_count,
            anchor_count=anchor_count,
            delta_count=delta_count,
        ),
    )
    selector_fields = selection_logging_module.checkpoint_selector_state_fields(
        payload,
        prefix="checkpoint",
    )
    print(
        "[profile-checkpoint] nodes={} branches={} anchors={} deltas={} checkpoint_selector_state_present={} checkpoint_selector_state_type={} checkpoint_selector_state_version={}".format(
            node_count,
            runtime_branch_count
            if isinstance(runtime_branch_count, int)
            else "unknown",
            anchor_count,
            delta_count,
            selector_fields["checkpoint_selector_state_present"],
            selector_fields["checkpoint_selector_state_type"],
            selector_fields["checkpoint_selector_state_version"],
        )
    )


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
    _configure_logging(args.log_level)
    modules = _load_runtime_modules()
    _print_module_paths(modules)
    runner, _checkpoint_path = _load_runner_for_mode(args, modules)
    _profile_checkpoint_save(args=args, modules=modules, runner=runner)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
