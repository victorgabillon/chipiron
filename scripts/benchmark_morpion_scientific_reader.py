"""Fresh-process synthetic v2 reader measurements and full-reference parity.

Run prepare, cold, reuse and parity separately and serially. Measurement modes
forbid the full reader. No bootstrap/distributed worker or experiment is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import resource
from dataclasses import asdict
from pathlib import Path
from typing import Any, get_args

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
from chipiron.environments.morpion.bootstrap.sharded_training_reader import (
    ShardedTrainingGenerationView,
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


def measure(root: Path, path: Path, size: int, mode: str) -> dict[str, Any]:
    """Measure bounded initialization, a wrapped window and an exact full scan."""

    def forbidden(*_args: object, **_kwargs: object) -> Any:
        message = "full v2 oracle called by indexed measurement"
        raise AssertionError(message)

    scientific.load_scientific_training_generation = forbidden
    cache = root / "cache"
    if bool(list(cache.glob("v*.sqlite3"))) != (mode == "reuse"):
        message = "cold/reuse cache precondition failed"
        raise ValueError(message)
    stage = StageMeasurement()
    before = costs()
    with ShardedTrainingGenerationView(
        path, cache_dir=cache, allow_scientific_v2=True
    ) as view:
        assert view.index_reused == (mode == "reuse")
        initialization = stage.finish(
            costs=delta(before), reader=view.initialization_observation
        )
        stage = StageMeasurement()
        before = costs()
        selection = view.select_sorted_node_window(f"node-{size - 5:06d}-é", 64)
        nodes = view.load_nodes(selection[0])
        contracts = view.load_value_contracts(selection[0])
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
                    "contract": view.load_value_contracts((node.node_id,))[0],
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


def parity(root: Path, path: Path) -> dict[str, Any]:
    """Use the pinned full oracle only in this separate, unmeasured process."""
    oracle = scientific.load_scientific_training_generation(path)
    cold = json.loads((root / "cold.json").read_text())
    reused = json.loads((root / "reuse.json").read_text())
    ids = tuple(sorted(n.node_id for n in oracle.snapshot.nodes))
    selected = select_reevaluation_node_window(
        ids, start_cursor=f"node-{len(ids) - 5:06d}-é", max_nodes=64
    )
    by_id = {node.node_id: node for node in oracle.snapshot.nodes}
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
        path, cache_dir=root / "cache", allow_scientific_v2=True
    ) as view:
        assert view.provenance == oracle.provenance
        assert tuple(view.iter_nodes_in_export_order()) == oracle.snapshot.nodes
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


def measure_dataset(root: Path, path: Path) -> dict[str, Any]:
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
        path, cache_dir=root / "cache", allow_scientific_v2=True
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
    expected = root / f"{policy}-reference.jsonl"
    assert sha(output) == sha(expected)
    return {"dataset": observation, "jsonl_sha256": sha(output)}


def main() -> None:
    """Refuse historical inputs and overwrites; persist new scratch evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        choices=("prepare", "cold", "reuse", "parity", "dataset"),
        required=True,
    )
    parser.add_argument("--size", type=int, choices=(1000, 5000, 10000), required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(Path("/tmp")):
        parser.error("only new synthetic /tmp fixtures are permitted")
    logging.disable(logging.CRITICAL)
    if args.action == "prepare":
        root.mkdir(parents=True, exist_ok=False)
        path = generated_scientific_source(
            root / "source", args.size, padding=4096, generations=2
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
    ):
        parser.error("synthetic fixture identity mismatch")
    path = root / fixture["manifest"]
    if args.action == "parity":
        result = parity(root, path)
    elif args.action == "dataset":
        result = measure_dataset(root, path)
    else:
        result = measure(root, path, args.size, args.action)
    output = root / f"{args.action}.json"
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps({"output": str(output), "mode": args.action, "size": args.size}),
        flush=True,
    )


if __name__ == "__main__":
    main()
