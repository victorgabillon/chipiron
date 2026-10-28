"""Fresh-process synthetic v2 reader measurements and full-reference parity.

Run prepare, cold, reuse and parity separately and serially. Measurement modes
forbid the full reader. No bootstrap/distributed worker or experiment is used.
"""

import argparse
import cProfile
import hashlib
import json
import logging
import pstats
import resource
import sqlite3
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, get_args
from unittest.mock import patch

import chipiron.environments.morpion.bootstrap.sharded_training_reader as indexed_reader
import chipiron.environments.morpion.bootstrap.sharded_training_v2 as scientific
from chipiron.environments.morpion.bootstrap.bootstrap_args import MorpionBootstrapArgs
from chipiron.environments.morpion.bootstrap.cycle_dataset import (
    extract_rows_from_training_snapshot,
)
from chipiron.environments.morpion.bootstrap.performance import StageMeasurement
from chipiron.environments.morpion.bootstrap.pv_family_targets import (
    PvFamilyTargetPolicy,
)
from chipiron.environments.morpion.bootstrap.reevaluation_worker import (
    MorpionActiveModelNodeReevaluationEvaluator,
    select_reevaluation_node_window,
)
from chipiron.environments.morpion.bootstrap.sharded_training_export import (
    load_morpion_sharded_training_tree_snapshot,
)
from chipiron.environments.morpion.bootstrap.sharded_training_reader import (
    ShardedTrainingGenerationView,
    iter_shard_rows,
)
from chipiron.environments.morpion.bootstrap.streaming_dataset import (
    IndexedDatasetSource,
    disk_dataset_context,
)
from chipiron.environments.morpion.learning import (
    save_morpion_supervised_rows_streaming,
)
from tests.environments.morpion_scientific_reader_helpers import (
    generated_scientific_source,
)
from tests.environments.test_morpion_scientific_training_reader import StateScore

if TYPE_CHECKING:
    from collections.abc import Iterator

PINS = {
    "reader": "79693ce2dc59c6d6ccb951951c9ba75fa69057f3",
    "dataset": "5c8906ba0d520c63d821e14a2d52530f8bed2307",
    "scientific": "bfd8f2ab05f017ffda5e14af5dbf6e1f1afa3fde",
}


def encode(value: object) -> bytes:
    """Encode complete scientific payloads deterministically, retaining all fields."""
    return json.dumps(
        value, sort_keys=True, ensure_ascii=True, allow_nan=False
    ).encode()


def sha(path: Path) -> str:
    """Hash actual file contents using bounded buffers."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def io() -> dict[str, int]:
    """Read actual process logical and physical I/O counters on Linux."""
    return {
        key: int(value)
        for key, value in (
            line.split(":", 1)
            for line in Path("/proc/self/io").read_text().splitlines()
        )
    }


def costs() -> tuple[float, float, dict[str, int]]:
    """Capture CPU user/system time and kernel I/O counters."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime, usage.ru_stime, io()


def delta(before: tuple[float, float, dict[str, int]]) -> dict[str, Any]:
    """Return measured costs, distinguishing cached reads from physical I/O."""
    user, system, current = costs()
    return {
        "cpu_user_s": user - before[0],
        "cpu_system_s": system - before[1],
        "io": {key: value - before[2][key] for key, value in current.items()},
    }


@contextmanager
def profile_operations() -> Iterator[tuple[dict[str, dict[str, float]], list[bool]]]:
    """Profile-only hooks, following the existing writer benchmark's mock pattern.

    cProfile/pstats remain the profiler. These bounded counters distinguish SQL
    literals and suspension-aware shard parsing that cProfile combines. No SQL
    parameters or node populations are retained. Enabled only for the requested
    phase (cProfile uses monitoring rather than sys.getprofile on Python 3.13).
    """
    observations: dict[str, dict[str, float]] = {}
    active = [False]

    def record(label: str, started: float) -> None:
        entry = observations.setdefault(label, {"calls": 0, "elapsed_s": 0.0})
        entry["calls"] += 1
        entry["elapsed_s"] += perf_counter() - started

    class ObservedConnection(sqlite3.Connection):
        def execute(self, sql: str, parameters: Any = (), /) -> sqlite3.Cursor:
            if not active[0]:
                return super().execute(sql, parameters)
            started = perf_counter()
            try:
                return super().execute(sql, parameters)
            finally:
                record("SQL.execute " + " ".join(sql.split()), started)

        def executemany(self, sql: str, parameters: Any, /) -> sqlite3.Cursor:
            started = perf_counter()
            try:
                return super().executemany(sql, parameters)
            finally:
                if active[0]:
                    record("SQL.executemany " + " ".join(sql.split()), started)

        def executescript(self, sql: str, /) -> sqlite3.Cursor:
            started = perf_counter()
            try:
                return super().executescript(sql)
            finally:
                if active[0]:
                    record("SQL.executescript " + " ".join(sql.split()), started)

        def commit(self) -> None:
            started = perf_counter()
            try:
                return super().commit()
            finally:
                if active[0]:
                    record("SQL.commit", started)

    connect = sqlite3.connect
    iterator = indexed_reader.iter_shard_rows

    def observed_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        return connect(*args, **kwargs, factory=ObservedConnection)

    def observed_rows(
        path: Path, field: str, **kwargs: Any
    ) -> Iterator[dict[str, object]]:
        rows = iterator(path, field, **kwargs)
        try:
            while True:
                started = perf_counter()
                try:
                    row = next(rows)
                except StopIteration:
                    return
                finally:
                    if active[0]:
                        record(
                            f"parser.{field}.generation-{kwargs.get('scientific_generation', 'v1')}",
                            started,
                        )
                yield row
        finally:
            rows.close()

    with ExitStack() as stack:
        stack.enter_context(patch.object(sqlite3, "connect", observed_connect))
        stack.enter_context(
            patch.object(indexed_reader, "iter_shard_rows", observed_rows)
        )
        yield observations, active


def measure(
    root: Path,
    path: Path,
    size: int,
    mode: str,
    *,
    profile: bool = False,
    cache_dir: Path | None = None,
    capture: list[bool] | None = None,
) -> dict[str, Any]:
    """Measure bounded initialization, a wrapped window and an exact full scan."""

    def forbidden(*_args: object, **_kwargs: object) -> Any:
        message = "full v2 oracle called by indexed measurement"
        raise AssertionError(message)

    scientific.load_scientific_training_generation = forbidden
    cache = cache_dir or root / "cache"
    if mode == "reuse" and not any(cache.glob("v*.sqlite3")):
        message = "cold/reuse cache precondition failed"
        raise ValueError(message)
    stage = StageMeasurement()
    before = costs()
    profiler = cProfile.Profile() if profile else None
    if capture is not None:
        capture[0] = True
    if profiler is not None:
        profiler.enable()
    view = ShardedTrainingGenerationView(
        path, cache_dir=cache, allow_scientific_v2=True
    )
    if profiler is not None:
        profiler.disable()
        profiler.dump_stats(str(root / f"{mode}.prof"))
    if capture is not None:
        capture[0] = False
    with view:
        assert view.index_reused == (mode == "reuse")
        initialization = stage.finish(
            costs=delta(before), reader=view.initialization_observation
        )
        stage = StageMeasurement()
        before = costs()
        selection = view.select_sorted_node_window(f"node-{size - 5:06d}-é", 64)
        nodes = view.load_nodes(selection[0])
        contracts = (
            view.load_value_contracts(selection[0])
            if view.format_version == 2
            else None
        )
        adapter = MorpionActiveModelNodeReevaluationEvaluator(
            Path("/synthetic/fixed-model")
        )
        adapter._evaluator = StateScore()
        patches = adapter.evaluate_node_window(nodes)
        selected_digest = hashlib.sha256(
            encode({
                "nodes": [asdict(n) for n in nodes],
                "contracts": contracts,
                "selection": selection,
                "provenance": view.provenance,
                "patch_rows": [asdict(row) for row in patches],
            })
        ).hexdigest()
        window = stage.finish(
            costs=delta(before), size=len(nodes), scientific_sha256=selected_digest
        )
        stage = StageMeasurement()
        before = costs()
        digest = hashlib.sha256()
        count = 0
        for node in view.iter_nodes_in_export_order():
            digest.update(
                encode({
                    "node": asdict(node),
                    "contract": view.load_value_contracts((node.node_id,))[0]
                    if view.format_version == 2
                    else None,
                })
            )
            digest.update(b"\n")
            count += 1
        scan = stage.finish(
            costs=delta(before), node_count=count, scientific_sha256=digest.hexdigest()
        )
        view.validate_source_identity()
        return {
            "mode": mode,
            "node_count": size,
            "initialization": initialization,
            "window": window,
            "scan": scan,
            "index_bytes": view.index_bytes,
            "index_build_peak_disk_bytes": view.index_build_peak_disk_bytes,
            "index_build_temp_logical_bytes": view.index_build_temp_logical_bytes,
            "source_bytes": sum(p.stat().st_size for p in view._source_paths),
        }


def parity(root: Path, path: Path, *, cache_dir: Path | None = None) -> dict[str, Any]:
    """Use the pinned full oracle only in this separate, unmeasured process."""
    oracle = scientific.load_scientific_training_generation(path)
    cold = json.loads((root / "cold.json").read_text())
    reused = json.loads((root / "reuse.json").read_text())
    ids = tuple(sorted(n.node_id for n in oracle.snapshot.nodes))
    selected = select_reevaluation_node_window(
        ids, start_cursor=f"node-{len(ids) - 5:06d}-é", max_nodes=64
    )
    selected_ids = set(selected[0])
    by_id = {
        node.node_id: node
        for node in oracle.snapshot.nodes
        if node.node_id in selected_ids
    }
    nodes = tuple(by_id[node_id] for node_id in selected[0])
    adapter = MorpionActiveModelNodeReevaluationEvaluator(
        Path("/synthetic/fixed-model")
    )
    adapter._evaluator = StateScore()
    patches = adapter.evaluate_patch_rows(oracle.snapshot, selected[0])
    expected_window = hashlib.sha256(
        encode({
            "nodes": [asdict(n) for n in nodes],
            "contracts": tuple(oracle.value_contracts[i] for i in selected[0]),
            "selection": selected,
            "provenance": oracle.provenance,
            "patch_rows": [asdict(row) for row in patches],
        })
    ).hexdigest()
    digest = hashlib.sha256()
    for node in oracle.snapshot.nodes:
        digest.update(
            encode({
                "node": asdict(node),
                "contract": oracle.value_contracts[node.node_id],
            })
        )
        digest.update(b"\n")
    for measurement in (cold, reused):
        assert measurement["window"]["scientific_sha256"] == expected_window
        assert measurement["scan"]["scientific_sha256"] == digest.hexdigest()
    rows = []
    with ShardedTrainingGenerationView(
        path, cache_dir=cache_dir or root / "cache", allow_scientific_v2=True
    ) as view:
        assert view.provenance == oracle.provenance
        for actual, expected_node in zip(
            view.iter_nodes_in_export_order(), oracle.snapshot.nodes, strict=True
        ):
            assert actual == expected_node
        for policy in get_args(PvFamilyTargetPolicy):
            args = MorpionBootstrapArgs(
                work_dir=root, dataset_family_target_policy=policy
            )
            expected = extract_rows_from_training_snapshot(
                args=args, snapshot=oracle.snapshot, generation=view.generation
            )
            expected_path = root / f"{policy}-reference.jsonl"
            actual_path = root / f"{policy}-indexed.jsonl"
            save_morpion_supervised_rows_streaming(
                rows=expected.rows, metadata=expected.metadata, path=expected_path
            )
            with disk_dataset_context(IndexedDatasetSource(view), args) as context:
                stream = context.stream(view.generation)
                assert stream.metadata == expected.metadata
                save_morpion_supervised_rows_streaming(
                    rows=stream.rows, metadata=stream.metadata, path=actual_path
                )
            with (
                expected_path.open("rb") as reference,
                actual_path.open("rb") as indexed,
            ):
                while chunk := reference.read(64 * 1024):
                    assert chunk == indexed.read(len(chunk))
                assert indexed.read(1) == b""
            assert sha(expected_path) == sha(actual_path)
            rows.append({
                "policy": policy,
                "row_count": len(expected.rows),
                "bytes": actual_path.stat().st_size,
                "jsonl_sha256": sha(actual_path),
                "all_fields_metadata_and_bytes_equal": True,
            })
            print(
                json.dumps({"nodes": len(ids), "policy": policy, "rows_equal": True}),
                flush=True,
            )
    return {
        "all_scientific_fields_equal": True,
        "all_value_contracts_equal": True,
        "all_provenance_equal": True,
        "window_and_wrap_equal": True,
        "reevaluation_rows_and_metadata_equal": True,
        "export_sha256": digest.hexdigest(),
        "window_sha256": expected_window,
        "policies": rows,
        "incompatible_backup_certainty_nodes": sum(
            not c["backup_exactness_compatible"]
            for c in oracle.value_contracts.values()
        ),
    }


def measure_dataset(
    root: Path,
    path: Path,
    *,
    expected: Path | None = None,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Measure a complete global-family Dataset pass, forbidding the full oracle."""

    def forbidden(*_args: object, **_kwargs: object) -> Any:
        message = "full v2 oracle called by indexed Dataset measurement"
        raise AssertionError(message)

    scientific.load_scientific_training_generation = forbidden
    policy = "pv_exact_then_blend_mean_prediction"
    args = MorpionBootstrapArgs(work_dir=root, dataset_family_target_policy=policy)
    output = root / "measured-dataset.jsonl"
    if output.exists():
        message = "refusing to overwrite Dataset measurement"
        raise ValueError(message)
    stage = StageMeasurement()
    before = costs()
    with ShardedTrainingGenerationView(
        path, cache_dir=cache_dir or root / "cache", allow_scientific_v2=True
    ) as view:
        assert view.index_reused
        with disk_dataset_context(IndexedDatasetSource(view), args) as context:
            stream = context.stream(view.generation)
            stats = save_morpion_supervised_rows_streaming(
                rows=stream.rows, metadata=stream.metadata, path=output
            )
            scratch_bytes = context.scratch_bytes()
        view.validate_source_identity()
        observation = stage.finish(
            costs=delta(before),
            policy=policy,
            node_count=view.node_count(),
            stats={**asdict(stats), "path": str(stats.path)},
            temp_logical_bytes=scratch_bytes,
        )
    expected = expected or root / f"{policy}-reference.jsonl"
    assert sha(output) == sha(expected)
    return {"dataset": observation, "jsonl_sha256": sha(output)}


def measure_reference(
    root: Path, path: Path, *, legacy: bool, generation: int
) -> dict[str, Any]:
    """Measure the full oracle only in its own separate reference process."""
    stage, before = StageMeasurement(), costs()
    loaded = (
        load_morpion_sharded_training_tree_snapshot(path)
        if legacy
        else scientific.load_scientific_training_generation(path)
    )
    snapshot = loaded if legacy else loaded.snapshot
    initialization = stage.finish(costs=delta(before), node_count=len(snapshot.nodes))
    stage, before = StageMeasurement(), costs()
    policy = "pv_exact_then_blend_mean_prediction"
    rows = extract_rows_from_training_snapshot(
        args=MorpionBootstrapArgs(work_dir=root, dataset_family_target_policy=policy),
        snapshot=snapshot,
        generation=generation,
    )
    output = root / f"{policy}-reference.jsonl"
    stats = save_morpion_supervised_rows_streaming(
        rows=rows.rows, metadata=rows.metadata, path=output
    )
    dataset = stage.finish(
        costs=delta(before), stats={**asdict(stats), "path": str(stats.path)}
    )
    return {
        "initialization": initialization,
        "dataset": dataset,
        "jsonl_sha256": sha(output),
    }


def audit_matched_inputs(root: Path) -> dict[str, Any]:
    """Compare actual immutable refs and update order with the existing parser."""
    legacy_manifest = json.loads((root / "v1/generation_000001.json").read_text())
    first_records = root / "source/generation_000001/records.jsonl"
    fingerprint = hashlib.sha256()
    counts: dict[int, int] = {}
    states: dict[int, str] = {}
    count = 0
    for old, new in zip(
        iter_shard_rows(root / "v1" / legacy_manifest["node_shard_path"], "nodes"),
        iter_shard_rows(first_records, "records", scientific_generation=1),
        strict=True,
    ):
        assert old["node_id"] == new["node_id"]
        assert encode(old["state_ref_payload"]) == encode(new["state_ref_payload"])
        ref = new["state_ref_payload"]
        moves = len(ref[1])
        state_sha = hashlib.sha256(encode(ref)).hexdigest()
        assert states.setdefault(moves, state_sha) == state_sha
        counts[moves] = counts.get(moves, 0) + 1
        fingerprint.update(encode([new["node_id"], ref]))
        fingerprint.update(b"\n")
        count += 1
    orders = []
    for generation in (1, 2):
        manifest = json.loads(
            (root / f"v1/generation_{generation:06d}.json").read_text()
        )
        order = hashlib.sha256()
        for old, new in zip(
            iter_shard_rows(root / "v1" / manifest["update_shard_path"], "updates"),
            iter_shard_rows(
                root / f"source/generation_{generation:06d}/updates.jsonl",
                "updates",
                scientific_generation=generation,
            ),
            strict=True,
        ):
            assert old["node_id"] == new["node_id"]
            assert old["child_ids"] == new["child_ids"]
            order.update(encode(old["node_id"]))
            order.update(b"\n")
        orders.append(order.hexdigest())
    return {
        "node_count": count,
        "ordered_node_and_reference_sha256": fingerprint.hexdigest(),
        "v1_v2_immutable_reference_bytes_equal": True,
        "current_child_ids_and_update_order_equal": True,
        "update_order_sha256": orders,
        "moves_histogram": counts,
        "distinct_references": len(states),
        "known_format_difference": "v1 generation 2 incoming parents remain creation-time; v2 preserves current links and richer scientific values/provenance",
        "sources": [
            {
                "path": str(p.relative_to(root)),
                "bytes": p.stat().st_size,
                "sha256": sha(p),
            }
            for directory in (root / "source", root / "v1")
            for p in sorted(directory.rglob("*"))
            if p.is_file()
        ],
    }


def main() -> None:
    """Refuse historical inputs and overwrites; persist new scratch evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        choices=(
            "prepare",
            "cold",
            "reuse",
            "parity",
            "dataset",
            "reference",
            "audit-inputs",
        ),
        required=True,
    )
    parser.add_argument("--size", type=int, choices=(1000, 5000, 10000), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--matched-v1", action="store_true")
    parser.add_argument("--format", choices=("v1", "v2"), default="v2")
    parser.add_argument("--generation", type=int, choices=(1, 2), default=2)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument(
        "--operation-counts",
        action="store_true",
        help="SQL/parser phase timers without cProfile overhead",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Existing cProfile/pstats tools; overhead-bearing samples",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(Path("/tmp")):
        parser.error("only new synthetic /tmp fixtures are permitted")
    logging.disable(logging.CRITICAL)
    if args.action == "prepare":
        root.mkdir(parents=True, exist_ok=False)
        path = generated_scientific_source(
            root / "source",
            args.size,
            padding=4096,
            generations=2,
            legacy_root=root / "v1" if args.matched_v1 else None,
        )
        (root / "fixture.json").write_text(
            json.dumps(
                {
                    "synthetic": True,
                    "size": args.size,
                    "manifest": path.relative_to(root).as_posix(),
                    "pins": PINS,
                    "oracle_source_sha256": sha(Path(scientific.__file__)),
                },
                indent=2,
            )
            + "\n"
        )
        return
    fixture = json.loads((root / "fixture.json").read_text())
    if (
        fixture["synthetic"] is not True
        or fixture["size"] != args.size
        or fixture["pins"] != PINS
        or fixture["oracle_source_sha256"] != sha(Path(scientific.__file__))
    ):
        parser.error("synthetic fixture identity mismatch")
    path = (
        root / "v1" / f"generation_{args.generation:06d}.json"
        if args.format == "v1"
        else root / "source" / f"generation_{args.generation:06d}" / "generation.json"
    )
    output_dir = args.output_dir.resolve() if args.output_dir else root
    if not output_dir.is_relative_to(root):
        parser.error("output-dir must be inside the synthetic fixture root")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.cache_dir and not args.cache_dir.resolve().is_relative_to(root):
        parser.error("cache-dir must be inside the synthetic fixture root")
    observers = ExitStack()
    operations, capture = (
        observers.enter_context(profile_operations())
        if args.profile or args.operation_counts
        else (None, None)
    )
    profiler = (
        cProfile.Profile()
        if args.profile and args.action not in ("cold", "reuse")
        else None
    )
    if profiler is not None:
        profiler.enable()
    if capture is not None and args.action not in ("cold", "reuse"):
        capture[0] = True
    if args.action == "audit-inputs":
        result = audit_matched_inputs(root)
    elif args.action == "parity":
        result = parity(output_dir, path, cache_dir=args.cache_dir)
    elif args.action == "dataset":
        expected = (
            (
                args.reference_dir.resolve()
                if args.reference_dir
                else root / f"reference-{args.format}-g{args.generation}"
            )
            / "pv_exact_then_blend_mean_prediction-reference.jsonl"
            if args.output_dir
            else None
        )
        if expected is not None and (
            not expected.is_relative_to(root) or not expected.is_file()
        ):
            parser.error(
                "reference JSONL must already exist within this synthetic root"
            )
        result = measure_dataset(
            output_dir, path, expected=expected, cache_dir=args.cache_dir
        )
    elif args.action == "reference":
        result = measure_reference(
            output_dir, path, legacy=args.format == "v1", generation=args.generation
        )
    else:
        result = measure(
            output_dir,
            path,
            args.size,
            args.action,
            profile=args.profile,
            cache_dir=args.cache_dir,
            capture=capture,
        )
    if profiler is not None:
        profiler.disable()
        profiler.dump_stats(str(output_dir / f"{args.action}.prof"))
    if capture is not None:
        capture[0] = False
    observers.close()
    if operations is not None:
        result["profile_operations"] = operations
    if args.profile:
        with (output_dir / f"{args.action}-profile.txt").open("w") as stream:
            pstats.Stats(
                str(output_dir / f"{args.action}.prof"), stream=stream
            ).sort_stats("cumulative").print_stats(60)
            pstats.Stats(
                str(output_dir / f"{args.action}.prof"), stream=stream
            ).sort_stats("tottime").print_stats(60)
    output = output_dir / f"{args.action}.json"
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps({"output": str(output), "mode": args.action, "size": args.size}),
        flush=True,
    )


if __name__ == "__main__":
    main()
