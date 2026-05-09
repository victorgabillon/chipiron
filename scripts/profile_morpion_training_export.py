#!/usr/bin/env python3
"""Standalone Morpion training export profiling outside the bootstrap pipeline."""

from __future__ import annotations

import argparse
import cProfile
import json
import logging
import pstats
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, TypeVar

from anemone.checkpoints import resolve_latest_generation_checkpoint_path

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


DEFAULT_TOP_FUNCTIONS = 80

_T = TypeVar("_T")


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for Morpion training export profiling."""
    parser = argparse.ArgumentParser(
        description=(
            "Profile Morpion training/tree export payload build and optional JSON "
            "serialization outside the full bootstrap pipeline."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("load",),
        required=True,
        help="Load an existing runtime checkpoint before profiling training export.",
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
        help="Optional JSON output path used only when --dump-json is enabled.",
    )
    parser.add_argument(
        "--profile-output",
        type=Path,
        default=Path("morpion_training_export.prof"),
        help="Path to the generated cProfile .prof file.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP_FUNCTIONS,
        help="Number of top functions to print for cumulative and tottime stats.",
    )
    parser.add_argument(
        "--dump-json",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to write the dumped export JSON to --output after profiling.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Python logging level for the standalone profiler process.",
    )
    profile_mode_group = parser.add_mutually_exclusive_group()
    profile_mode_group.add_argument(
        "--build-only",
        dest="profile_mode",
        action="store_const",
        const="build_only",
        default="build_only",
        help="Profile only training export payload build.",
    )
    profile_mode_group.add_argument(
        "--full-save",
        dest="profile_mode",
        action="store_const",
        const="full_save",
        help="Profile payload build plus payload_to_dict/json_dump/json_write phases.",
    )
    return parser


def _time_call(function: Callable[[], _T]) -> tuple[_T, float]:
    """Return the function result plus wall-clock elapsed seconds."""
    started_at = perf_counter()
    result = function()
    return result, perf_counter() - started_at


def _resolve_latest_runtime_checkpoint(runtime_checkpoint_dir: Path) -> Path:
    """Return the latest generation checkpoint from one runtime checkpoint dir."""
    return resolve_latest_generation_checkpoint_path(runtime_checkpoint_dir)


def _configure_logging(log_level: str) -> None:
    """Enable plain logs when the caller has not configured logging."""
    logging.basicConfig(level=getattr(logging, log_level), format="%(message)s")


def _load_runtime_modules() -> dict[str, Any]:
    """Import runtime modules lazily so parser tests stay lightweight."""
    import anemone
    import atomheart
    import chipiron
    from anemone.training_export import training_tree_snapshot_to_dict
    from chipiron.environments.morpion.bootstrap import (
        AnemoneMorpionSearchRunner,
        MorpionBootstrapPaths,
    )

    return {
        "anemone": anemone,
        "atomheart": atomheart,
        "chipiron": chipiron,
        "training_tree_snapshot_to_dict": training_tree_snapshot_to_dict,
        "MorpionBootstrapPaths": MorpionBootstrapPaths,
        "AnemoneMorpionSearchRunner": AnemoneMorpionSearchRunner,
    }


def _print_module_paths(modules: dict[str, Any]) -> None:
    """Print module file paths to confirm the script is profiling the right code."""
    print(f"anemone: {getattr(modules['anemone'], '__file__', None)}")
    print(f"chipiron: {getattr(modules['chipiron'], '__file__', None)}")
    print(f"atomheart: {getattr(modules['atomheart'], '__file__', None)}")


def _require_output_path(args: argparse.Namespace) -> Path:
    """Return the required output path for JSON write mode or fail clearly."""
    if args.output is None:
        raise ValueError("--output is required when --dump-json is enabled.")
    return Path(args.output)


def _load_runner_for_mode(
    args: argparse.Namespace,
    modules: dict[str, Any],
) -> tuple[Any, Path]:
    """Return a runner initialized from an existing runtime checkpoint."""
    runner = modules["AnemoneMorpionSearchRunner"]()
    bootstrap_paths = modules["MorpionBootstrapPaths"].from_work_dir(args.work_dir)
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


def _profile_training_export(
    *,
    args: argparse.Namespace,
    modules: dict[str, Any],
    runner: Any,
) -> None:
    """Profile training export payload build and optional serialization phases."""
    training_tree_snapshot_to_dict = modules["training_tree_snapshot_to_dict"]

    runtime = runner._require_runtime()
    runtime_tree = getattr(runtime, "tree", None)
    runtime_branch_count = getattr(runtime_tree, "branch_count", None)
    print(
        "[profile] runtime nodes=%s branches=%s"
        % (
            runner.current_tree_size(),
            runtime_branch_count if isinstance(runtime_branch_count, int) else "unknown",
        )
    )

    if args.dump_json:
        output_path = _require_output_path(args)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None

    profile_output = Path(args.profile_output)
    profile_output.parent.mkdir(parents=True, exist_ok=True)
    profiler = cProfile.Profile()
    payload_dict: dict[str, Any] | None = None
    dumped_json: str | None = None

    total_started_at = perf_counter()
    if args.profile_mode in {"build_only", "full_save"}:
        profiler.enable()
    build_result, payload_build_s = _time_call(
        runner.build_training_tree_snapshot_payload
    )
    snapshot, _export_profile = build_result
    if args.profile_mode == "build_only":
        profiler.disable()
    print(f"[profile] phase=payload_build elapsed_s={payload_build_s:.6f}")

    payload_to_dict_s: float | None = None
    json_dump_s: float | None = None
    json_write_s: float | None = None
    json_write_bytes: int | None = None

    if args.profile_mode == "full_save":
        payload_dict, payload_to_dict_s = _time_call(
            lambda: training_tree_snapshot_to_dict(snapshot)
        )
        print(f"[profile] phase=payload_to_dict elapsed_s={payload_to_dict_s:.6f}")
        dumped_json, json_dump_s = _time_call(
            lambda: json.dumps(payload_dict, indent=2) + "\n"
        )
        print(f"[profile] phase=json_dump elapsed_s={json_dump_s:.6f}")
        if args.dump_json:
            assert output_path is not None
            json_write_bytes, json_write_s = _time_call(
                lambda: _write_json_dump(dumped_json, output_path)
            )
            print(
                "[profile] phase=json_write elapsed_s=%.6f bytes=%s output=%s"
                % (json_write_s, json_write_bytes, output_path)
            )
        else:
            print("[profile] phase=json_write skipped=true")
        profiler.disable()
    else:
        print("[profile] phase=payload_to_dict skipped=true")
        print("[profile] phase=json_dump skipped=true")
        print("[profile] phase=json_write skipped=true")

    total_s = perf_counter() - total_started_at
    print(f"[profile] phase=total elapsed_s={total_s:.6f}")

    profiler.dump_stats(str(profile_output))
    print(f"[profile] phase=cprofile_dump output={profile_output}")
    _print_cprofile_stats(profiler, args.top)


def _write_json_dump(dumped_json: str | None, output_path: Path) -> int:
    """Write the dumped JSON text and return the byte count."""
    if dumped_json is None:
        raise ValueError("Dumped JSON is required before json_write.")
    encoded_json = dumped_json.encode("utf-8")
    output_path.write_bytes(encoded_json)
    return len(encoded_json)


def _print_cprofile_stats(profiler: cProfile.Profile, top: int) -> None:
    """Print the top cumulative and total-time cProfile entries."""
    print(f"[profile] stats sort=cumulative top={top}")
    pstats.Stats(profiler).sort_stats("cumulative").print_stats(top)
    print(f"[profile] stats sort=tottime top={top}")
    pstats.Stats(profiler).sort_stats("tottime").print_stats(top)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the standalone Morpion training export profiler."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)
    modules = _load_runtime_modules()
    _print_module_paths(modules)
    runner, _checkpoint_path = _load_runner_for_mode(args, modules)
    _profile_training_export(args=args, modules=modules, runner=runner)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())