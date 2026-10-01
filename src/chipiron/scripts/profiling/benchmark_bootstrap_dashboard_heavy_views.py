"""Benchmark the expensive Morpion bootstrap dashboard read paths.

This command is intentionally read-only. It places all derived dashboard caches in a
temporary directory so historical/scientific workspaces are never modified.

Example:
    python -m chipiron.scripts.profiling.benchmark_bootstrap_dashboard_heavy_views \
        /path/to/generic_linoo_fresh_with_bigrun_models_v1 \
        /path/to/big_run_01
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    MorpionBootstrapPaths,
)
from chipiron.environments.morpion.bootstrap.dashboard.checkpoint_reader import (
    read_inspection_checkpoint,
)
from chipiron.environments.morpion.bootstrap.dashboard.history_view import (
    _board_view_from_cached_record,
    _load_resolved_training_tree_snapshot,
    _resolve_latest_tree_snapshot_reference,
    build_current_certified_record_board_view,
    load_morpion_bootstrap_run_view,
)
from chipiron.environments.morpion.bootstrap.dashboard.record_view_cache import (
    CachedCertifiedRecord,
)
from chipiron.environments.morpion.bootstrap.dashboard.tree_inspector import (
    _INDEXED_CHECKPOINT_TREE_CACHE,
    _build_selected_node_snapshot_parts,
    _index_checkpoint_payload,
    build_morpion_bootstrap_tree_inspector_snapshot,
    resolve_latest_runtime_checkpoint,
)
from chipiron.environments.morpion.bootstrap.record_status import (
    select_best_certified_record_candidate_from_training_tree_snapshot,
)


@dataclass(frozen=True, slots=True)
class Timing:
    """One named wall-clock benchmark measurement."""

    name: str
    seconds: float


def _timed[T](name: str, operation: Callable[[], T]) -> tuple[T, Timing]:
    started_at = time.perf_counter()
    result = operation()
    return result, Timing(name=name, seconds=time.perf_counter() - started_at)


def _clear_tree_process_caches() -> None:
    """Simulate a fresh dashboard Python process while retaining persistent cache."""
    _INDEXED_CHECKPOINT_TREE_CACHE.clear()
    _build_selected_node_snapshot_parts.cache_clear()
    gc.collect()


def _benchmark_record(work_dir: Path) -> list[Timing]:
    """Measure legacy O(N), first fast-path, and persistent Record reads."""
    timings: list[Timing] = []
    run_view = load_morpion_bootstrap_run_view(work_dir)
    resolved = _resolve_latest_tree_snapshot_reference(run_view)
    if resolved.snapshot_path is not None:
        snapshot, timing = _timed(
            "record.baseline.training_snapshot_load",
            lambda: _load_resolved_training_tree_snapshot(resolved),
        )
        timings.append(timing)
        if snapshot is not None:
            candidate, timing = _timed(
                "record.baseline.certified_candidate_scan",
                lambda: (
                    select_best_certified_record_candidate_from_training_tree_snapshot(
                        snapshot
                    )
                ),
            )
            timings.append(timing)
            if candidate is not None:
                record = CachedCertifiedRecord(
                    variant=candidate.variant,
                    moves_since_start=candidate.moves_since_start,
                    total_points=candidate.total_points,
                    is_exact=True,
                    is_terminal=True,
                    source="certified_terminal_leaf",
                    state_ref_payload=candidate.state_ref_payload,
                    node_id=candidate.node_id,
                )
                _view, timing = _timed(
                    "record.baseline.state_decode_and_svg",
                    lambda: _board_view_from_cached_record(
                        record,
                        source_context="benchmark baseline",
                    ),
                )
                timings.append(timing)
            del snapshot
            gc.collect()

    _view, timing = _timed(
        "record.optimized.first_read",
        lambda: build_current_certified_record_board_view(work_dir),
    )
    timings.append(timing)
    _view, timing = _timed(
        "record.optimized.persistent_read",
        lambda: build_current_certified_record_board_view(work_dir),
    )
    timings.append(timing)
    return timings


def _benchmark_tree(work_dir: Path) -> list[Timing]:
    """Measure legacy full checkpoint work and persistent-index inspector reads."""
    timings: list[Timing] = []
    paths = MorpionBootstrapPaths.from_work_dir(work_dir)
    resolved = resolve_latest_runtime_checkpoint(paths)
    if resolved.checkpoint_path is None:
        return timings

    checkpoint_payload, timing = _timed(
        "tree.baseline.checkpoint_deserialize",
        lambda: read_inspection_checkpoint(resolved.checkpoint_path),
    )
    timings.append(timing)
    _index, timing = _timed(
        "tree.baseline.in_memory_index_build",
        lambda: _index_checkpoint_payload(checkpoint_payload),
    )
    timings.append(timing)
    del checkpoint_payload
    del _index
    gc.collect()

    _clear_tree_process_caches()
    first, timing = _timed(
        "tree.optimized.first_index_and_root",
        lambda: build_morpion_bootstrap_tree_inspector_snapshot(work_dir),
    )
    timings.append(timing)

    _clear_tree_process_caches()
    reopened, timing = _timed(
        "tree.optimized.persisted_root_after_restart",
        lambda: build_morpion_bootstrap_tree_inspector_snapshot(work_dir),
    )
    timings.append(timing)

    expanded_child_id = next(
        (
            child.child_node_id
            for child in reopened.child_summaries
            if child.child_node_id is not None
        ),
        None,
    )
    if expanded_child_id is not None:
        _child, timing = _timed(
            "tree.optimized.selected_child_navigation",
            lambda: build_morpion_bootstrap_tree_inspector_snapshot(
                work_dir,
                selected_node_id=expanded_child_id,
            ),
        )
        timings.append(timing)

    del first
    del reopened
    gc.collect()
    return timings


def benchmark_work_dir(work_dir: Path) -> dict[str, object]:
    """Benchmark one run without writing into its scientific workspace."""
    resolved = work_dir.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)

    with tempfile.TemporaryDirectory(prefix="chipiron-dashboard-benchmark-") as cache:
        previous = os.environ.get("CHIPIRON_DASHBOARD_CACHE_DIR")
        os.environ["CHIPIRON_DASHBOARD_CACHE_DIR"] = cache
        try:
            record_timings = _benchmark_record(resolved)
            tree_timings = _benchmark_tree(resolved)
        finally:
            if previous is None:
                os.environ.pop("CHIPIRON_DASHBOARD_CACHE_DIR", None)
            else:
                os.environ["CHIPIRON_DASHBOARD_CACHE_DIR"] = previous

    return {
        "work_dir": str(resolved),
        "record": {timing.name: round(timing.seconds, 6) for timing in record_timings},
        "tree": {timing.name: round(timing.seconds, 6) for timing in tree_timings},
    }


def main() -> None:
    """Run read-only benchmarks and print machine-readable JSON."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare legacy full-tree dashboard reads with persistent Record/Tree "
            "cache paths. This can be I/O and RAM intensive on first/baseline reads."
        )
    )
    parser.add_argument(
        "work_dir",
        nargs="+",
        type=Path,
        help="Morpion bootstrap work directory to benchmark.",
    )
    args = parser.parse_args()
    results = [benchmark_work_dir(path) for path in args.work_dir]
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
