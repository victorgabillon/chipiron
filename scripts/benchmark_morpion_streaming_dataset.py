"""Measure generated-only Dataset workers serially in fresh processes.

No existing experiment fixture is accepted. Optional legacy-source selects a
read-only code oracle, never scientific artifacts. All output is scratch data.
"""
# ruff: noqa: TRY003

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args

if TYPE_CHECKING:
    from chipiron.environments.morpion.bootstrap.pv_family_targets import (
        PvFamilyTargetPolicy,
    )


def generated_dataset(
    root: Path,
    count: int,
    *,
    generation: int = 1,
    padding: int = 0,
    chain: bool = False,
    escape_unicode: bool = False,
) -> Path:
    """Write an incremental synthetic forest with conflicting exact family anchors."""
    root.mkdir(parents=True, exist_ok=True)
    ids = [f"node-{i:06d}" for i in range(count)]
    if count > 3:
        ids[1:4] = ["é", "😀", "a\u0000b"]
    for field, name in (("nodes", "nodes"), ("updates", f"updates-{generation}")):
        with (root / f"{name}.json").open("w") as stream:
            stream.write('{"' + field + '":[')
            for i, node_id in enumerate(ids):
                if i:
                    stream.write(",")
                if field == "nodes":
                    row = {
                        "node_id": node_id,
                        "depth": i % 13,
                        "creation_generation": 1,
                        "parent_ids": [],
                        "state_ref_payload": None if i % 41 == 40 else [0, []],
                    }
                else:
                    # Different from numeric and lexical ID order, including stable ties.
                    row = {
                        "node_id": node_id,
                        "order_index": (count - i) // 2,
                        "child_ids": ids[i + 1 : min(i + 3, (i // 13 + 1) * 13, count)],
                        "direct_value_scalar": None
                        if i % 7 == 0
                        else (i % 17 - 8) / 10,
                        "backed_up_value_scalar": None
                        if i % 5 == 0
                        else (i % 11 - 5) / 3 + generation,
                        "is_terminal": i % 13 == 12,
                        "is_exact": i % 7 == 1,
                        "visit_count": None if i % 17 == 0 else i % 19,
                        "over_event_label": "win" if i % 13 == 12 else None,
                        "metadata": {
                            "synthetic_padding": "x" * padding,
                            "fraction": 0.1,
                        },
                    }
                    # Keep every node's raw family value available; direct-only frontier
                    # nodes still exercise the extraction skipped-no-target counter.
                    if (
                        row["direct_value_scalar"] is None
                        and row["backed_up_value_scalar"] is None
                    ):
                        row["direct_value_scalar"] = -0.0
                if field == "updates" and chain:
                    row["child_ids"] = ids[i + 1 : i + 2]
                json.dump(row, stream, ensure_ascii=escape_unicode)
            stream.write("]}")
    path = root / f"generation_{generation:06d}.json"
    path.write_text(
        json.dumps({
            "generation": generation,
            "node_count": count,
            "new_node_count": count if generation == 1 else 0,
            "root_node_id": ids[0] if ids else "empty",
            "node_shard_path": "nodes.json",
            "update_shard_path": f"updates-{generation}.json",
        })
    )
    (root / "manifest.json").write_text(
        json.dumps({
            "latest_generation": generation,
            "generation_manifests": {
                str(i): f"generation_{i:06d}.json" for i in range(1, generation + 1)
            },
        })
    )
    if generation > 1:
        # G2 contains no new records; the original immutable shard remains selected.
        (root / "empty.json").write_text('{"nodes":[]}')
        payload = json.loads(path.read_text())
        payload["node_shard_path"] = "empty.json"
        path.write_text(json.dumps(payload))
    return path


def child(
    root: Path,
    mode: str,
    policy: PvFamilyTargetPolicy = "pv_exact_then_blend_mean_prediction",
) -> dict[str, Any]:
    """Measure a full Dataset worker, including record/frontier/JSONL/manifests."""
    from chipiron.environments.morpion.bootstrap.bootstrap_args import (
        MorpionBootstrapArgs,
    )
    from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
        MorpionBootstrapPaths,
    )
    from chipiron.environments.morpion.bootstrap.performance import StageMeasurement
    from chipiron.environments.morpion.bootstrap.pipeline import stages
    from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
        MorpionPipelineGenerationManifest,
        save_pipeline_manifest,
    )

    paths = MorpionBootstrapPaths.from_work_dir(root)
    paths.ensure_directories()
    manifest_path = root / "export" / "generation_000001.json"
    save_pipeline_manifest(
        MorpionPipelineGenerationManifest(
            generation=1,
            created_at_utc="2026-01-01T00:00:00Z",
            tree_snapshot_path=str(manifest_path),
            dataset_status="not_started",
            training_status="not_started",
        ),
        paths.pipeline_manifest_path_for_generation(1),
    )
    leaderboard = root / "leaderboard.jsonl"
    leaderboard.unlink(missing_ok=True)
    from chipiron.environments.morpion.bootstrap.record_status import (
        persist_certified_leaderboard_candidates as original_persist,
    )

    stages.persist_certified_leaderboard_candidates = lambda **kwargs: original_persist(
        **kwargs, leaderboard_path=leaderboard
    )
    stages._now_timestamp_utc = lambda: "2026-01-01T00:00:00Z"
    args = MorpionBootstrapArgs(
        work_dir=root,
        pipeline_mode="artifact_pipeline",
        training_export_mode="sharded",
        dataset_family_target_policy=policy,
    )
    options = (
        {"use_streaming_sharded_reader": True, "reader_cache_dir": root / "cache"}
        if mode != "legacy"
        else {}
    )
    measurement = StageMeasurement()
    stages.run_pipeline_dataset_stage(args, generation=1, **options)
    observation = measurement.finish(mode=mode)
    performance = max(
        (root / "pipeline" / "performance").rglob("dataset-*.json"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    stage = json.loads(performance.read_text())
    outputs = {
        "rows": paths.rows_jsonl_path_for_generation(1),
        "manifest": paths.pipeline_manifest_path_for_generation(1),
        "status": paths.pipeline_dataset_status_path_for_generation(1),
        "leaderboard": leaderboard,
    }
    hashes = {}
    for name, path in outputs.items():
        with path.open("rb") as stream:
            hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        **observation,
        "stage": stage,
        "artifact_sha256": hashes,
        "rows_per_s": stage["rows_emitted"] / observation["elapsed_s"],
        "source_bytes": sum(p.stat().st_size for p in manifest_path.parent.iterdir()),
        "index_bytes": sum(
            p.stat().st_size for p in (root / "cache").glob("*.sqlite3")
        ),
    }


def main() -> None:
    """Generate 1k/5k/10k fixtures and compare all scientific artifact hashes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 5000, 10000])
    parser.add_argument("--payload-bytes", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-source", type=Path)
    parser.add_argument(
        "--all-policies",
        action="store_true",
        help="Compare complete artifact hashes for every target policy; timings are not individual fresh-process samples.",
    )
    parser.add_argument(
        "--escape-unicode",
        action="store_true",
        help="Exercise the reader's escaped-surrogate fallback.",
    )
    parser.add_argument(
        "--chain", action="store_true", help="Use one whole-generation PV family."
    )
    parser.add_argument(
        "--child", choices=["legacy", "cold", "reuse"], help=argparse.SUPPRESS
    )
    parser.add_argument("--fixture", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        if args.fixture is None or os.environ.get("CHIPIRON_SYNTHETIC_DATASET") != str(
            args.fixture
        ):
            parser.error("Child requires the parent's generated temporary fixture")
        if args.all_policies:
            from chipiron.environments.morpion.bootstrap.pv_family_targets import (
                PvFamilyTargetPolicy,
            )

            hashes = {}
            for policy in get_args(PvFamilyTargetPolicy):
                result = child(args.fixture, args.child, policy)
                hashes[policy] = result["artifact_sha256"]
            print(json.dumps({"artifact_sha256": hashes}))
        else:
            print(json.dumps(child(args.fixture, args.child)))
        return
    if min(args.sizes) < 1 or args.payload_bytes < 0:
        parser.error("Positive sizes and nonnegative payload bytes required")
    results = []
    for count in args.sizes:
        with tempfile.TemporaryDirectory(
            prefix="chipiron-synthetic-dataset-"
        ) as temporary:
            root = Path(temporary)
            generated_dataset(
                root / "export",
                count,
                padding=args.payload_bytes,
                chain=args.chain,
                escape_unicode=args.escape_unicode,
            )
            expected = None
            for mode in (
                ("legacy", "cold") if args.all_policies else ("legacy", "cold", "reuse")
            ):
                env = {
                    **os.environ,
                    "CHIPIRON_SYNTHETIC_DATASET": str(root),
                    "PYTHONDONTWRITEBYTECODE": "1",
                }
                if mode == "legacy" and args.legacy_source:
                    env["PYTHONPATH"] = str(args.legacy_source.resolve())
                start = time.perf_counter()
                process = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--child",
                        mode,
                        "--fixture",
                        str(root),
                        "--output",
                        str(args.output),
                        *(["--all-policies"] if args.all_policies else []),
                    ],
                    capture_output=True,
                    text=True,
                    env=env,
                    check=False,
                )
                if process.returncode:
                    raise RuntimeError(process.stderr)
                result = {
                    "total_nodes": count,
                    "mode": mode,
                    **json.loads(process.stdout),
                    "process_wall_s": time.perf_counter() - start,
                }
                if expected is None:
                    expected = result["artifact_sha256"]
                if result["artifact_sha256"] != expected:
                    raise AssertionError(
                        f"Scientific artifact mismatch at {count}/{mode}"
                    )
                results.append(result)
                print(json.dumps(result), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "payload_bytes": args.payload_bytes,
                "chain": args.chain,
                "escape_unicode": args.escape_unicode,
                "all_policies": args.all_policies,
                "legacy_source": str(args.legacy_source),
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
