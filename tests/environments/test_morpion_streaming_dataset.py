"""Exact differential tests for opt-in Dataset extraction from the shared index."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import TYPE_CHECKING, get_args

import pytest

from chipiron.environments.morpion.bootstrap.bootstrap_args import MorpionBootstrapArgs
from chipiron.environments.morpion.bootstrap.cycle_dataset import (
    streaming_rows_from_training_snapshot,
)
from chipiron.environments.morpion.bootstrap.pv_family_targets import (
    PvFamilyTargetPolicy,
)
from chipiron.environments.morpion.bootstrap.sharded_training_export import (
    load_morpion_sharded_training_tree_snapshot,
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

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
        MorpionBootstrapPaths,
    )


def generated_dataset(
    root: Path, count: int, *, generation: int = 1, padding: int = 0
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
                json.dump(row, stream, ensure_ascii=False)
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


def assert_parity(path: Path, args: MorpionBootstrapArgs, root: Path) -> None:
    """Compare every field, every counter and exact serialized UTF-8 JSONL bytes."""
    snapshot = load_morpion_sharded_training_tree_snapshot(path)
    legacy = streaming_rows_from_training_snapshot(
        args=args, snapshot=snapshot, generation=7
    )
    legacy_rows = tuple(legacy.rows)
    expected = root / "legacy.jsonl"
    actual = root / "indexed.jsonl"
    save_morpion_supervised_rows_streaming(
        rows=legacy_rows, metadata=legacy.metadata, path=expected
    )
    with ShardedTrainingGenerationView(path, cache_dir=root / "cache") as view:
        with disk_dataset_context(IndexedDatasetSource(view), args) as context:
            result = context.stream(7)
            assert result.metadata == legacy.metadata
            assert tuple(result.rows) == legacy_rows
            # A fresh pass and the actual saved bytes must be identical as well.
            result = context.stream(7)
            stats = save_morpion_supervised_rows_streaming(
                rows=result.rows, metadata=result.metadata, path=actual
            )
            assert stats.row_count == len(legacy_rows)
            assert actual.read_bytes() == expected.read_bytes()
        assert not view._connection.execute(
            "SELECT name FROM sqlite_temp_master WHERE type='table'"
        ).fetchall()


@pytest.fixture(scope="module")
def dataset_source(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Path]:
    """One immutable input per size and worker; every policy owns its index/output."""
    path = generated_dataset(tmp_path_factory.mktemp("dataset-source"), request.param)

    def fingerprints() -> dict[str, str]:
        result = {}
        for source in path.parent.iterdir():
            with source.open("rb") as stream:
                result[source.name] = hashlib.file_digest(stream, "sha256").hexdigest()
        return result

    before = fingerprints()
    for source in path.parent.iterdir():
        source.chmod(0o444)
    path.parent.chmod(0o555)
    try:
        yield path
        assert fingerprints() == before
    finally:
        path.parent.chmod(0o755)
        for source in path.parent.iterdir():
            source.chmod(0o644)


@pytest.mark.parametrize(
    "dataset_source",
    [
        pytest.param(30, marks=pytest.mark.fast),
        1000,
        pytest.param(5000, marks=pytest.mark.exhaustive),
        pytest.param(10000, marks=pytest.mark.exhaustive),
    ],
    indirect=True,
)
@pytest.mark.parametrize("policy", get_args(PvFamilyTargetPolicy))
def test_every_policy_generated_scale_exact_parity(
    tmp_path: Path, dataset_source: Path, policy: PvFamilyTargetPolicy
) -> None:
    """All nine scientific policies compare exactly at every requested population."""
    assert_parity(
        dataset_source,
        MorpionBootstrapArgs(work_dir=tmp_path, dataset_family_target_policy=policy),
        tmp_path,
    )


@pytest.mark.parametrize("policy", get_args(PvFamilyTargetPolicy))
@pytest.mark.parametrize(
    "options",
    [
        {"use_backed_up_value": False},
        {"require_exact_or_terminal": True, "min_depth": 3, "min_visit_count": 7},
        {"max_rows": 0},
        {"max_rows": 1},
        {"max_rows": 27},
        {"max_rows": 9999},
        {"min_depth": 99},
        {"dataset_family_prediction_blend": 0.0},
        {"dataset_family_prediction_blend": 1.0},
    ],
)
def test_filters_limits_and_blends(
    tmp_path: Path, policy: PvFamilyTargetPolicy, options: dict[str, object]
) -> None:
    """Filtering, max_rows boundaries, target source choices and blend endpoints match."""
    path = generated_dataset(tmp_path / "export", 100)
    args = replace(
        MorpionBootstrapArgs(work_dir=tmp_path, dataset_family_target_policy=policy),
        **options,
    )
    assert_parity(path, args, tmp_path)


def test_empty_generation(tmp_path: Path) -> None:
    """Empty summaries preserve nulls and counters."""
    path = generated_dataset(tmp_path / "export", 0)
    assert_parity(
        path,
        MorpionBootstrapArgs(
            work_dir=tmp_path,
        ),
        tmp_path,
    )


def test_generation_change_cache_reuse_and_source_guard(tmp_path: Path) -> None:
    """Reused indexes preserve multipass output; changed sources invalidate reuse."""
    path = generated_dataset(tmp_path / "export", 100)
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
    )
    assert_parity(path, args, tmp_path)
    with ShardedTrainingGenerationView(path, cache_dir=tmp_path / "cache") as view:
        assert view.index_reused
    path = generated_dataset(path.parent, 100, generation=2)
    assert_parity(path, args, tmp_path)
    with ShardedTrainingGenerationView(path, cache_dir=tmp_path / "cache") as view:
        assert view.index_reused
        with disk_dataset_context(IndexedDatasetSource(view), args) as context:
            result = context.stream(2)
            shard = path.parent / "updates-2.json"
            shard.write_text(shard.read_text() + " ")
            with pytest.raises(ValueError, match="changed during Dataset"):
                save_morpion_supervised_rows_streaming(
                    rows=result.rows,
                    metadata=result.metadata,
                    path=tmp_path / "changed.jsonl",
                )
            assert not (tmp_path / "changed.jsonl").exists()
            assert not (tmp_path / "changed.jsonl.tmp").exists()
    with ShardedTrainingGenerationView(path, cache_dir=tmp_path / "cache") as view:
        assert not view.index_reused


def test_interrupted_output_and_cycle_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failed writes preserve a completed output and remove temporary scratch tables."""
    path = generated_dataset(tmp_path / "export", 100)
    target = tmp_path / "rows.jsonl"
    target.write_text("previous completed output")
    with ShardedTrainingGenerationView(path, cache_dir=tmp_path / "cache") as view:
        with (
            pytest.raises(RuntimeError, match="interrupted"),
            disk_dataset_context(
                IndexedDatasetSource(view),
                MorpionBootstrapArgs(
                    work_dir=tmp_path,
                ),
            ) as context,
        ):
            stream = context.stream(1)
            original = context.adjust_row

            def interrupted(row: object) -> object:
                raise RuntimeError("interrupted")

            monkeypatch.setattr(context, "adjust_row", interrupted)
            with pytest.raises(RuntimeError, match="interrupted"):
                save_morpion_supervised_rows_streaming(
                    rows=stream.rows, metadata=stream.metadata, path=target
                )
            assert target.read_text() == "previous completed output"
            assert not target.with_suffix(".jsonl.tmp").exists()
            monkeypatch.setattr(context, "adjust_row", original)
            raise RuntimeError("interrupted")
        with disk_dataset_context(
            IndexedDatasetSource(view),
            MorpionBootstrapArgs(
                work_dir=tmp_path,
            ),
        ) as context:
            assert context.stream(1).metadata["num_rows"] > 0
    updates = path.parent / "updates-1.json"
    payload = json.loads(updates.read_text())
    payload["updates"][0]["child_ids"] = ["node-000000"]
    updates.write_text(json.dumps(payload))
    with (
        ShardedTrainingGenerationView(path, cache_dir=tmp_path / "cache") as view,
        pytest.raises(ValueError, match="cycle"),
        disk_dataset_context(
            IndexedDatasetSource(view), MorpionBootstrapArgs(work_dir=tmp_path)
        ),
    ):
        pass


def setup_dataset_worker(tmp_path: Path) -> MorpionBootstrapPaths:
    """Prepare one entirely synthetic pipeline generation."""
    from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
        MorpionBootstrapPaths,
    )
    from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
        MorpionPipelineGenerationManifest,
        save_pipeline_manifest,
    )

    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    tree = generated_dataset(tmp_path / "export", 100)
    save_pipeline_manifest(
        MorpionPipelineGenerationManifest(
            generation=1,
            created_at_utc="2026-01-01T00:00:00Z",
            tree_snapshot_path=str(tree),
            dataset_status="not_started",
            training_status="not_started",
        ),
        paths.pipeline_manifest_path_for_generation(1),
    )
    return paths


@pytest.mark.parametrize("policy", get_args(PvFamilyTargetPolicy))
def test_worker_artifacts_and_no_full_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: PvFamilyTargetPolicy
) -> None:
    """Compare manifests/status/leaderboard and forbid snapshots and whole-shard loads."""
    from chipiron.environments.morpion.bootstrap import (
        sharded_training_export as export,
    )
    from chipiron.environments.morpion.bootstrap.pipeline import stages

    paths = setup_dataset_worker(tmp_path)
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        training_export_mode="sharded",
        pipeline_mode="artifact_pipeline",
        dataset_family_target_policy=policy,
    )
    monkeypatch.setattr(stages, "_now_timestamp_utc", lambda: "2026-01-01T00:00:00Z")
    original_persist = stages.persist_certified_leaderboard_candidates
    monkeypatch.setattr(
        stages,
        "persist_certified_leaderboard_candidates",
        lambda **kwargs: original_persist(
            **kwargs, leaderboard_path=tmp_path / "leaderboard.jsonl"
        ),
    )
    initial = paths.pipeline_manifest_path_for_generation(1).read_bytes()
    stages.run_pipeline_dataset_stage(args, generation=1)
    files = [
        paths.rows_jsonl_path_for_generation(1),
        paths.pipeline_manifest_path_for_generation(1),
        paths.pipeline_dataset_status_path_for_generation(1),
        tmp_path / "leaderboard.jsonl",
    ]
    expected = [path.read_bytes() for path in files]
    paths.pipeline_manifest_path_for_generation(1).write_bytes(initial)
    (tmp_path / "leaderboard.jsonl").unlink(missing_ok=True)

    def forbidden(*_: object, **__: object) -> object:
        pytest.fail("full snapshot or full source shard used")

    monkeypatch.setattr(stages, "_load_training_snapshot_for_generation", forbidden)
    monkeypatch.setattr(export, "TrainingTreeSnapshot", forbidden)
    original_read = export._read_json

    def manifests_only(path: Path) -> object:
        if path.name in {"manifest.json", "generation_000001.json"}:
            return original_read(path)
        return forbidden(path)

    monkeypatch.setattr(export, "_read_json", manifests_only)
    stages.run_pipeline_dataset_stage(
        args,
        generation=1,
        use_streaming_sharded_reader=True,
        reader_cache_dir=tmp_path / "cache",
    )
    assert [path.read_bytes() for path in files] == expected


def test_malformed_source_fails_worker_and_releases_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Malformed shards never publish output or fall back to permissive whole loading."""
    from chipiron.environments.morpion.bootstrap.pipeline import stages
    from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
        load_pipeline_manifest,
    )

    paths = setup_dataset_worker(tmp_path)
    (tmp_path / "export" / "updates-1.json").write_text('{"updates":[')
    monkeypatch.setattr(
        stages,
        "_load_training_snapshot_for_generation",
        lambda **_: pytest.fail("fallback on malformed source"),
    )
    with pytest.raises(ValueError, match=r"Malformed|incomplete"):
        stages.run_pipeline_dataset_stage(
            MorpionBootstrapArgs(
                work_dir=tmp_path,
                training_export_mode="sharded",
                pipeline_mode="artifact_pipeline",
            ),
            generation=1,
            use_streaming_sharded_reader=True,
            reader_cache_dir=tmp_path / "cache",
        )
    assert not paths.rows_jsonl_path_for_generation(1).exists()
    assert not paths.pipeline_dataset_claim_path_for_generation(1).exists()
    assert (
        load_pipeline_manifest(
            paths.pipeline_manifest_path_for_generation(1)
        ).dataset_status
        == "failed"
    )


def test_unsupported_scalar_retains_legacy_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Targetless family members retain legacy behavior, including policy-specific errors."""
    from chipiron.environments.morpion.bootstrap.pipeline import stages

    setup_dataset_worker(tmp_path)
    updates = tmp_path / "export" / "updates-1.json"
    payload = json.loads(updates.read_text())
    payload["updates"][0].update(direct_value_scalar=None, backed_up_value_scalar=None)
    updates.write_text(json.dumps(payload))
    original = stages._load_training_snapshot_for_generation
    calls = []

    def legacy(**kwargs: object) -> object:
        calls.append(True)
        return original(**kwargs)

    monkeypatch.setattr(stages, "_load_training_snapshot_for_generation", legacy)
    monkeypatch.setattr(
        stages, "persist_certified_leaderboard_candidates", lambda **_: None
    )
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        training_export_mode="sharded",
        pipeline_mode="artifact_pipeline",
        dataset_family_target_policy="pv_mean_prediction",
    )
    with caplog.at_level("INFO"):
        stages.run_pipeline_dataset_stage(
            args,
            generation=1,
            use_streaming_sharded_reader=True,
            reader_cache_dir=tmp_path / "cache",
        )
    assert calls == [True]
    assert "missing_or_nonfinite_family_scalar" in caplog.text


def test_flat_snapshot_retains_legacy_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Opt-in leaves the existing flat snapshot loader and extraction available."""
    from anemone.training_export import save_training_tree_snapshot

    from chipiron.environments.morpion.bootstrap.pipeline import stages
    from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
        load_pipeline_manifest,
        save_pipeline_manifest,
    )

    paths = setup_dataset_worker(tmp_path)
    snapshot = load_morpion_sharded_training_tree_snapshot(
        tmp_path / "export" / "generation_000001.json"
    )
    flat = tmp_path / "flat.json"
    save_training_tree_snapshot(snapshot, flat)
    manifest = load_pipeline_manifest(paths.pipeline_manifest_path_for_generation(1))
    save_pipeline_manifest(
        replace(manifest, tree_snapshot_path=str(flat)),
        paths.pipeline_manifest_path_for_generation(1),
    )
    monkeypatch.setattr(
        stages, "persist_certified_leaderboard_candidates", lambda **_: None
    )
    with caplog.at_level("INFO"):
        stages.run_pipeline_dataset_stage(
            MorpionBootstrapArgs(
                work_dir=tmp_path,
                pipeline_mode="artifact_pipeline",
                training_export_mode="flat",
            ),
            generation=1,
            use_streaming_sharded_reader=True,
        )
    assert "unsupported_format_or_output" in caplog.text
    assert paths.rows_jsonl_path_for_generation(1).exists()


@pytest.mark.parametrize("policy", get_args(PvFamilyTargetPolicy))
def test_one_long_family_and_float_order(
    tmp_path: Path, policy: PvFamilyTargetPolicy
) -> None:
    """A whole-population family preserves compensated sums, ties and signed zeros."""
    path = generated_dataset(tmp_path / "export", 1000)
    updates_path = path.parent / "updates-1.json"
    payload = json.loads(updates_path.read_text())
    updates = payload["updates"]
    predictions = [1e16, 1.0, -1e16, -0.0, 0.0, 1e-20]
    for index, update in enumerate(updates):
        update["child_ids"] = (
            [updates[index + 1]["node_id"]] if index + 1 < len(updates) else []
        )
        update["direct_value_scalar"] = predictions[index % len(predictions)]
        update["backed_up_value_scalar"] = -0.0 if index % 2 else 0.0
        update["is_exact"] = update["is_terminal"] = False
    updates_path.write_text(json.dumps(payload))
    assert_parity(
        path,
        MorpionBootstrapArgs(work_dir=tmp_path, dataset_family_target_policy=policy),
        tmp_path,
    )


def test_leftover_interrupted_output_is_rebuilt(tmp_path: Path) -> None:
    """A killed writer's incomplete temporary file is never consumed as completed rows."""
    path = generated_dataset(tmp_path / "export", 30)
    target = tmp_path / "rows.jsonl"
    target.with_suffix(".jsonl.tmp").write_text('{"kind":"incomplete')
    with (
        ShardedTrainingGenerationView(path, cache_dir=tmp_path / "cache") as view,
        disk_dataset_context(
            IndexedDatasetSource(view), MorpionBootstrapArgs(work_dir=tmp_path)
        ) as context,
    ):
        stream = context.stream(1)
        stats = save_morpion_supervised_rows_streaming(
            rows=stream.rows, metadata=stream.metadata, path=target
        )
        assert stats.row_count == stream.metadata["num_rows"]
    assert not target.with_suffix(".jsonl.tmp").exists()


def test_launcher_dataset_flag_default_off(tmp_path: Path) -> None:
    """The opt-in is ephemeral and default disabled."""
    from chipiron.environments.morpion.bootstrap.launcher import launcher_args_from_cli

    default = launcher_args_from_cli(["--work-dir", str(tmp_path)])
    enabled = launcher_args_from_cli([
        "--work-dir",
        str(tmp_path),
        "--dataset-streaming-sharded-reader",
    ])
    assert not default.dataset_streaming_sharded_reader
    assert enabled.dataset_streaming_sharded_reader
