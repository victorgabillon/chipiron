"""Full-reference versus bounded v2 contracts; frozen v1 remains separate."""
# ruff: noqa: TRY003

from __future__ import annotations

import json
import sqlite3
import tracemalloc
from typing import TYPE_CHECKING, Any, get_args

import pytest

import chipiron.environments.morpion.bootstrap.sharded_training_reader as reader
import chipiron.environments.morpion.bootstrap.sharded_training_v2 as scientific
from chipiron.environments.morpion.bootstrap.bootstrap_args import MorpionBootstrapArgs
from chipiron.environments.morpion.bootstrap.cycle_dataset import (
    extract_rows_from_training_snapshot,
)
from chipiron.environments.morpion.bootstrap.pv_family_targets import (
    PvFamilyTargetPolicy,
)
from chipiron.environments.morpion.bootstrap.reevaluation_worker import (
    MorpionActiveModelNodeReevaluationEvaluator,
    node_values_to_patch_rows,
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
from tests.environments.test_morpion_sharded_training_v2 import (
    PRODUCER,
    advance,
    nodes_fixture,
)

if TYPE_CHECKING:
    from pathlib import Path


def source(root: Path, final_generation: int = 2) -> Path:
    """Write a real-codec late-parent counterexample using the pinned v2 writer."""
    nodes = nodes_fixture()
    scientific.save_scientific_training_generation(
        nodes=nodes, root_node_id="0", output_dir=root, generation=1, producer=PRODUCER
    )
    advance(nodes)
    path, _ = scientific.save_scientific_training_generation(
        nodes=nodes, root_node_id="0", output_dir=root, generation=2, producer=PRODUCER
    )
    if final_generation == 3:
        path, _ = scientific.save_scientific_training_generation(
            nodes=nodes,
            root_node_id="0",
            output_dir=root,
            generation=3,
            producer=PRODUCER,
        )
    return path


def view(path: Path, **kwargs: Any) -> ShardedTrainingGenerationView:
    """Opt in explicitly and keep the cache outside all source artifacts."""
    return ShardedTrainingGenerationView(
        path,
        cache_dir=path.parent.parent.parent / "cache",
        allow_scientific_v2=True,
        **kwargs,
    )


@pytest.mark.fast
def test_scientific_windows_export_order_and_contracts(tmp_path: Path) -> None:
    """Every scientific field/contract/provenance survives, including late parents."""
    path = source(tmp_path / "source")
    oracle = scientific.load_scientific_training_generation(path)
    with view(path) as actual:
        assert tuple(actual.iter_nodes_in_export_order()) == oracle.snapshot.nodes
        assert actual.snapshot_metadata == oracle.snapshot.metadata
        assert actual.created_at_unix_s == oracle.snapshot.created_at_unix_s
        assert actual.provenance == oracle.provenance
        ids = tuple(sorted(node.node_id for node in oracle.snapshot.nodes))
        for cursor in (None, "missing", *ids):
            for size in (0, 1, 2, 4, 9):
                expected = select_reevaluation_node_window(
                    ids, start_cursor=cursor, max_nodes=size
                )
                assert actual.select_sorted_node_window(cursor, size) == expected
                selected = actual.load_nodes(expected[0])
                assert selected == tuple(
                    next(n for n in oracle.snapshot.nodes if n.node_id == i)
                    for i in expected[0]
                )
                assert actual.load_value_contracts(expected[0]) == tuple(
                    oracle.value_contracts[i] for i in expected[0]
                )
        assert actual.load_nodes(("3",))[0].parent_ids == ("1", "2")
        assert actual.load_nodes(("2",))[0].effective_value_scalar == 9
        assert actual.load_nodes(("2",))[0].target_value_scalar == 3
        assert actual.load_nodes(("3",))[0].target_value_scalar is None
        contracts = actual.load_value_contracts(("3",))
        contracts[0]["target_source"] = "changed by consumer"
        assert actual.load_value_contracts(("3",))[0] == oracle.value_contracts["3"]
    with view(path) as reused:
        assert reused.index_reused


@pytest.mark.fast
def test_no_full_reader_snapshot_or_source_json_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Forbid full readers, snapshots and read_text on every source artifact."""
    path = source(tmp_path / "source")
    oracle = scientific.load_scientific_training_generation(path)

    def forbidden(*_args: object, **_kwargs: object) -> Any:
        pytest.fail("unbounded scientific reader used")

    monkeypatch.setattr(scientific, "load_scientific_training_generation", forbidden)
    monkeypatch.setattr(scientific, "TrainingTreeSnapshot", forbidden)
    from pathlib import Path as RealPath

    original = RealPath.read_text

    def bounded_read(p: Path, *args: Any, **kwargs: Any) -> str:
        if p.is_relative_to(path.parent.parent):
            forbidden()
        return original(p, *args, **kwargs)

    monkeypatch.setattr(RealPath, "read_text", bounded_read)
    with view(path) as actual:
        assert tuple(actual.iter_nodes_in_export_order()) == oracle.snapshot.nodes
        assert actual.load_value_contracts(("2", "3")) == tuple(
            oracle.value_contracts[i] for i in ("2", "3")
        )


@pytest.mark.fast
@pytest.mark.parametrize("policy", get_args(PvFamilyTargetPolicy))
@pytest.mark.parametrize("backups", [False, True])
def test_dataset_all_policy_exact_rows_metadata_jsonl(
    tmp_path: Path, policy: str, backups: bool
) -> None:
    """Retain every row/summary field and byte under filtering, blending and limits."""
    path = source(tmp_path / "source")
    oracle = scientific.load_scientific_training_generation(path)
    with view(path) as actual:
        for blend in (0.0, 0.25, 1.0):
            for criteria in (
                {},
                {"require_exact_or_terminal": True},
                {"max_rows": 0},
                {"max_rows": 1},
                {"max_rows": 2},
                {"min_depth": 1},
                {"min_visit_count": 4},
            ):
                args = MorpionBootstrapArgs(
                    work_dir=tmp_path,
                    dataset_family_target_policy=policy,
                    use_backed_up_value=backups,
                    dataset_family_prediction_blend=blend,
                    **criteria,
                )
                expected = extract_rows_from_training_snapshot(
                    args=args, snapshot=oracle.snapshot, generation=2
                )
                with disk_dataset_context(
                    IndexedDatasetSource(actual), args
                ) as context:
                    stream = context.stream(generation=2)
                    assert stream.metadata == expected.metadata
                    assert tuple(stream.rows) == expected.rows
                    for name, rows, metadata in (
                        ("full", expected.rows, expected.metadata),
                        ("indexed", context.rows(), stream.metadata),
                    ):
                        save_morpion_supervised_rows_streaming(
                            rows=rows,
                            metadata=metadata,
                            path=tmp_path / f"{name}.jsonl",
                        )
                    assert (tmp_path / "full.jsonl").read_bytes() == (
                        tmp_path / "indexed.jsonl"
                    ).read_bytes()


class StateScore:
    """Deterministic model double exercising the real state-reference adapter."""

    def evaluate(self, state: Any) -> float:
        """Score concrete decoded geometry without loading a neural bundle."""
        return float(state.moves) + 0.125


def test_reevaluation_scientific_rows_wrap_and_metadata(tmp_path: Path) -> None:
    """Compare the actual built-in adapter and snapshot fallback on cursor windows."""
    path = source(tmp_path / "source")
    oracle = scientific.load_scientific_training_generation(path)
    adapter = MorpionActiveModelNodeReevaluationEvaluator(tmp_path / "shared-model")
    adapter._evaluator = StateScore()
    ids = tuple(sorted(n.node_id for n in oracle.snapshot.nodes))
    with view(path) as actual:
        for cursor in (None, "missing", "1", "3"):
            for size in (1, 3, 4, 9):
                selected, next_cursor, wrap = actual.select_sorted_node_window(
                    cursor, size
                )
                assert (selected, next_cursor, wrap) == select_reevaluation_node_window(
                    ids, start_cursor=cursor, max_nodes=size
                )
                nodes = actual.load_nodes(selected)
                expected_nodes = tuple(
                    next(n for n in oracle.snapshot.nodes if n.node_id == i)
                    for i in selected
                )
                assert node_values_to_patch_rows(nodes) == node_values_to_patch_rows(
                    expected_nodes
                )
                assert adapter.evaluate_node_window(
                    nodes
                ) == adapter.evaluate_patch_rows(oracle.snapshot, selected)


def test_opt_in_source_changes_and_semantic_cache_corruption(tmp_path: Path) -> None:
    """Reject implicit activation and rebuild even valid-SQLite scientific corruption."""
    path = source(tmp_path / "source")
    with pytest.raises(ValueError, match="explicit"):
        ShardedTrainingGenerationView(path, cache_dir=tmp_path / "cache")
    with view(path) as actual:
        old_index = actual.index_path
        expected = actual.load_nodes(("2",))
    with sqlite3.connect(old_index) as connection:
        connection.execute("UPDATE updates SET sequence=sequence+1")
    with view(path) as actual:
        assert not actual.index_reused
        assert actual.load_nodes(("2",)) == expected
        # Producer identity is both explicitly keyed and content-fingerprinted.
        catalog = path.parent.parent / "manifest.json"
        for changed in (catalog, path):
            data = json.loads(changed.read_text())
            data["producer"]["identity"]["revision"] = "new-source-identity"
            changed.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="changed"):
            actual.validate_source_identity()
    with view(path) as changed:
        assert changed.index_path != old_index
        assert not changed.index_reused


def test_interrupted_build_not_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Aborted prefixes never publish a completed index and can be rebuilt."""
    path = source(tmp_path / "source")
    original = reader.iter_shard_rows

    def interrupted(*args: Any, **kwargs: Any) -> Any:
        for row in original(*args, **kwargs):
            yield row
            raise OSError("interrupted index build")

    with monkeypatch.context() as patch:
        patch.setattr(reader, "iter_shard_rows", interrupted)
        with pytest.raises(OSError):
            view(path)
    assert not list((tmp_path / "cache").glob("v*.sqlite3"))
    assert not list((tmp_path / "cache").glob(".building-*"))
    with view(path) as actual:
        assert not actual.index_reused


@pytest.mark.exhaustive
def test_creation_map_larger_than_manifest_cap_is_streamed(tmp_path: Path) -> None:
    """A multi-MiB O(N) creation map never enters a Python dictionary."""
    path = generated_scientific_source(tmp_path / "source", 3000, id_padding=1500)
    assert path.stat().st_size > reader.MANIFEST_MAX_BYTES
    tracemalloc.start()
    try:
        with view(path) as actual:
            assert actual.node_count() == 3000
            assert actual._v2_manifest["creation_index"] is None
            assert len(actual.select_sorted_node_window(None, 7)[0]) == 7
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 8 * 1024 * 1024


@pytest.mark.fast
@pytest.mark.parametrize(
    "field",
    ["target_value_scalar", "effective_value_source", "parent_ids", "value_contract"],
)
def test_missing_field_never_becomes_explicit_null(tmp_path: Path, field: str) -> None:
    """Rehashed invalid sources still fail rather than invoking v1 defaults."""
    import hashlib

    path = source(tmp_path / "source")
    shard = path.parent / "updates.jsonl"
    rows = [json.loads(line) for line in shard.read_text().splitlines()]
    del rows[1][field]
    shard.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    data = json.loads(path.read_text())
    data["update_shard"]["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        view(path)


@pytest.mark.parametrize("policy", get_args(PvFamilyTargetPolicy))
def test_missing_values_keep_existing_policy_semantics(
    tmp_path: Path, policy: str
) -> None:
    """Some policies permit targetless family members; others explicitly fail."""
    nodes = nodes_fixture()
    nodes[0].tree_evaluation.direct_value = None
    nodes[0].tree_evaluation.tree_value = None
    path, _ = scientific.save_scientific_training_generation(
        nodes=nodes,
        root_node_id="0",
        output_dir=tmp_path / "source",
        generation=1,
        producer=PRODUCER,
    )
    oracle = scientific.load_scientific_training_generation(path)
    args = MorpionBootstrapArgs(work_dir=tmp_path, dataset_family_target_policy=policy)
    try:
        expected = extract_rows_from_training_snapshot(
            args=args, snapshot=oracle.snapshot, generation=1
        )
    except KeyError as expected_error:
        with (
            view(path) as actual,
            pytest.raises(KeyError) as actual_error,
            disk_dataset_context(IndexedDatasetSource(actual), args),
        ):
            pass
        assert actual_error.value.args == expected_error.args
    else:
        with (
            view(path) as actual,
            disk_dataset_context(IndexedDatasetSource(actual), args) as context,
        ):
            stream = context.stream(1)
            assert stream.metadata == expected.metadata
            assert tuple(stream.rows) == expected.rows


@pytest.mark.parametrize("policy", get_args(PvFamilyTargetPolicy))
def test_v2_dataset_worker_complete_artifact_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: str
) -> None:
    """Compare complete rows, manifests, status and leaderboard with full oracle."""
    from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
        MorpionBootstrapPaths,
    )
    from chipiron.environments.morpion.bootstrap.pipeline import stages
    from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
        MorpionPipelineGenerationManifest,
        save_pipeline_manifest,
    )

    path = source(tmp_path / "source")
    oracle = scientific.load_scientific_training_generation(path)
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    save_pipeline_manifest(
        MorpionPipelineGenerationManifest(
            generation=2,
            created_at_utc="2026-01-01T00:00:00Z",
            tree_snapshot_path=str(path),
            dataset_status="not_started",
            training_status="not_started",
        ),
        paths.pipeline_manifest_path_for_generation(2),
    )
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        pipeline_mode="artifact_pipeline",
        training_export_mode="sharded",
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
    initial = paths.pipeline_manifest_path_for_generation(2).read_bytes()
    monkeypatch.setattr(
        stages, "_load_training_snapshot_for_generation", lambda **_: oracle.snapshot
    )
    stages.run_pipeline_dataset_stage(args, generation=2)
    files = (
        paths.rows_jsonl_path_for_generation(2),
        paths.pipeline_manifest_path_for_generation(2),
        paths.pipeline_dataset_status_path_for_generation(2),
        tmp_path / "leaderboard.jsonl",
    )
    expected = [p.read_bytes() if p.exists() else None for p in files]
    paths.pipeline_manifest_path_for_generation(2).write_bytes(initial)
    (tmp_path / "leaderboard.jsonl").unlink(missing_ok=True)

    def forbidden(**_kwargs: object) -> Any:
        pytest.fail("full reader used in indexed v2 worker")

    monkeypatch.setattr(stages, "_load_training_snapshot_for_generation", forbidden)
    monkeypatch.setattr(scientific, "load_scientific_training_generation", forbidden)
    for _ in range(2):
        stages.run_pipeline_dataset_stage(
            args,
            generation=2,
            use_streaming_sharded_reader=True,
            allow_scientific_v2=True,
            reader_cache_dir=tmp_path / "cache",
        )
        assert [p.read_bytes() if p.exists() else None for p in files] == expected


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("wrong_model", [False, True])
def test_v2_reevaluation_worker_patch_cursor_identity_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fallback: bool, wrong_model: bool
) -> None:
    """Preserve whole patch/cursor metadata, wrap and model-owned cursor reset."""
    from dataclasses import replace

    import chipiron.environments.morpion.bootstrap.reevaluation_worker as worker
    from chipiron.environments.morpion.bootstrap import (
        load_reevaluation_cursor,
        save_reevaluation_cursor,
    )
    from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
        load_pipeline_manifest,
        save_pipeline_manifest,
    )
    from tests.environments.test_morpion_streaming_reevaluation import (
        run_once,
        setup_worker,
    )

    paths = setup_worker(tmp_path)
    path = source(tmp_path / "scientific-source", final_generation=3)
    oracle = scientific.load_scientific_training_generation(path)
    manifest_path = paths.pipeline_manifest_path_for_generation(3)
    manifest = load_pipeline_manifest(manifest_path)
    save_pipeline_manifest(
        replace(manifest, tree_snapshot_path=str(path)), manifest_path
    )
    cursor = load_reevaluation_cursor(paths.pipeline_reevaluation_cursor_path)
    save_reevaluation_cursor(
        replace(
            cursor,
            next_node_cursor="3",
            model_bundle_path="different-model"
            if wrong_model
            else cursor.model_bundle_path,
        ),
        paths.pipeline_reevaluation_cursor_path,
    )
    initial_cursor = paths.pipeline_reevaluation_cursor_path.read_bytes()

    def adapter(**_kwargs: object) -> MorpionActiveModelNodeReevaluationEvaluator:
        result = MorpionActiveModelNodeReevaluationEvaluator(
            tmp_path / "common-model-identity"
        )
        result._evaluator = StateScore()
        return result

    monkeypatch.setattr(worker, "build_active_model_reevaluation_evaluator", adapter)
    monkeypatch.setattr(
        worker, "load_reevaluation_training_tree_snapshot", lambda _: oracle.snapshot
    )
    expected = run_once(paths, streaming=False, use_snapshot_value_fallback=fallback)
    expected_patch = paths.pipeline_reevaluation_patch_path.read_bytes()
    expected_cursor = paths.pipeline_reevaluation_cursor_path.read_bytes()
    paths.pipeline_reevaluation_patch_path.unlink()
    paths.pipeline_reevaluation_cursor_path.write_bytes(initial_cursor)
    monkeypatch.setattr(
        worker,
        "load_reevaluation_training_tree_snapshot",
        lambda _: pytest.fail("full reader used in indexed v2 Reevaluation"),
    )
    actual = run_once(
        paths, allow_scientific_v2=True, use_snapshot_value_fallback=fallback
    )
    assert actual == expected
    assert paths.pipeline_reevaluation_patch_path.read_bytes() == expected_patch
    assert paths.pipeline_reevaluation_cursor_path.read_bytes() == expected_cursor


@pytest.mark.parametrize(
    "mutation",
    [
        "version",
        "contract",
        "producer",
        "stale_parent",
        "cycle",
        "creation",
        "shard_version",
        "duplicate",
        "tail",
        "noncanonical",
    ],
)
def test_malformed_scientific_sources_rejected_before_cache_publication(
    tmp_path: Path, mutation: str
) -> None:
    """Do not project mixed schemas, malformed fields or stale graph links."""
    import hashlib

    path = source(tmp_path / "source")
    manifest = json.loads(path.read_text())
    if mutation == "version":
        manifest["format_version"] = 3
    elif mutation == "contract":
        manifest["scientific_contract"] = "future-contract"
    elif mutation == "producer":
        manifest["producer"]["identity"]["revision"] = "wrong-catalog-producer"
    elif mutation == "creation":
        del manifest["creation_index"]["3"]
    else:
        descriptor = (
            manifest["record_shards"][0]
            if mutation == "noncanonical"
            else manifest["update_shard"]
        )
        shard = path.parent.parent / descriptor["path"]
        rows = [json.loads(line) for line in shard.read_text().splitlines()]
        if mutation == "stale_parent":
            rows[4]["parent_ids"] = ["1"]
        elif mutation == "cycle":
            rows[1]["parent_ids"] = ["3"]
            rows[4]["child_ids"] = ["0"]
        elif mutation == "shard_version":
            rows[0]["format_version"] = 2.0
        elif mutation == "noncanonical":
            from tests.environments.test_morpion_sharded_training_v2 import A, B

            rows[4]["state_ref_payload"] = [0, [B, A]]
        payload = "\n".join(json.dumps(row) for row in rows) + "\n"
        if mutation == "duplicate":
            payload = payload.replace(
                '"target_value_scalar": 6.0',
                '"target_value_scalar": 6.0, "target_value_scalar": 6.0',
                1,
            )
        elif mutation == "tail":
            payload += "{"
        shard.write_text(payload)
        descriptor["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, TypeError)):
        view(path)
    assert not list((tmp_path / "cache").glob("v*.sqlite3"))


def test_generation_changes_and_v1_v2_cache_namespaces(tmp_path: Path) -> None:
    """Keep v1 scientific interpretation frozen and generation/contract keys distinct."""
    from chipiron.environments.morpion.bootstrap.sharded_training_export import (
        load_morpion_sharded_training_tree_snapshot,
    )
    from tests.environments.test_morpion_sharded_training_reader import write_generation

    legacy_path = write_generation(tmp_path / "v1", ("0", "1", "2", "3"))
    with ShardedTrainingGenerationView(
        legacy_path, cache_dir=tmp_path / "cache"
    ) as legacy:
        assert (
            tuple(legacy.iter_nodes_in_export_order())
            == load_morpion_sharded_training_tree_snapshot(legacy_path).nodes
        )
        legacy_key = json.loads(legacy.identity)
        legacy_index = legacy.index_path
        with pytest.raises(ValueError, match="unavailable"):
            legacy.load_value_contracts(("0",))
    path = source(tmp_path / "v2")
    first = path.parent.parent / "generation_000001/generation.json"
    with view(first) as actual:
        before = tuple(actual.iter_nodes_in_export_order())
        first_index = actual.index_path
    with view(path) as actual:
        assert actual.index_path not in (legacy_index, first_index)
        key = json.loads(actual.identity)
        assert key["format_kind"] != legacy_key["format_kind"]
        assert (
            key["scientific_contract"] == scientific.CONTRACT and key["generation"] == 2
        )
        assert key["producer"] == actual.provenance["producer"]
    with view(first) as actual:
        assert actual.index_reused
        assert tuple(actual.iter_nodes_in_export_order()) == before


def test_interrupted_dataset_output_and_source_change_preserve_completed_file(
    tmp_path: Path,
) -> None:
    """Interrupted passes leave old output intact, including failed source validation."""
    path = source(tmp_path / "source")
    output = tmp_path / "rows.jsonl"
    output.write_bytes(b"previous completed artifact\n")
    args = MorpionBootstrapArgs(work_dir=tmp_path)
    with (
        view(path) as actual,
        disk_dataset_context(IndexedDatasetSource(actual), args) as context,
    ):
        stream = context.stream(2)

        def interrupted() -> Any:
            for row in stream.rows:
                yield row
                raise OSError("interrupted output")

        with pytest.raises(OSError):
            save_morpion_supervised_rows_streaming(
                rows=interrupted(), metadata=stream.metadata, path=output
            )
        assert output.read_bytes() == b"previous completed artifact\n"
        # Change bytes after the summary pass; the writer must fail before rename.
        shard = path.parent / "updates.jsonl"
        shard.write_text(shard.read_text() + " ")
        with pytest.raises(ValueError, match=r"hash|changed"):
            save_morpion_supervised_rows_streaming(
                rows=context.rows(), metadata=stream.metadata, path=output
            )
        assert output.read_bytes() == b"previous completed artifact\n"
    assert not (tmp_path / "rows.jsonl.tmp").exists()


def test_reevaluation_rechecks_source_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changed source bytes cannot publish a v2 patch or advance its cursor."""
    from dataclasses import replace

    import chipiron.environments.morpion.bootstrap.reevaluation_worker as worker
    from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
        load_pipeline_manifest,
        save_pipeline_manifest,
    )
    from tests.environments.test_morpion_streaming_reevaluation import (
        run_once,
        setup_worker,
    )

    paths = setup_worker(tmp_path)
    path = source(tmp_path / "source", final_generation=3)
    manifest_path = paths.pipeline_manifest_path_for_generation(3)
    save_pipeline_manifest(
        replace(load_pipeline_manifest(manifest_path), tree_snapshot_path=str(path)),
        manifest_path,
    )
    initial = paths.pipeline_reevaluation_cursor_path.read_bytes()

    class ChangedSource:
        def evaluate(self, _state: Any) -> float:
            catalog = path.parent.parent / "manifest.json"
            catalog.write_text(catalog.read_text() + " ")
            return 1.25

    adapter = MorpionActiveModelNodeReevaluationEvaluator(tmp_path / "model")
    adapter._evaluator = ChangedSource()
    monkeypatch.setattr(
        worker, "build_active_model_reevaluation_evaluator", lambda **_: adapter
    )
    with pytest.raises(ValueError, match="changed"):
        run_once(paths, allow_scientific_v2=True)
    assert not paths.pipeline_reevaluation_patch_path.exists()
    assert paths.pipeline_reevaluation_cursor_path.read_bytes() == initial


@pytest.mark.parametrize("tree_present", [False, True])
def test_real_terminal_reference_and_reevaluation(
    tmp_path: Path, tree_present: bool
) -> None:
    """Keep terminal certainty and the existing null-tree/model distinction."""
    from anemone.checkpoints import AnchorCheckpointStatePayload
    from anemone.checkpoints.state_handles import (
        CheckpointBackedStateHandle,
        CheckpointStateResolver,
    )
    from anemone.node_evaluation.common import canonical_value
    from atomheart.games.morpion.dynamics import MorpionDynamics
    from valanga import Outcome
    from valanga.over_event import OverEvent

    from tests.environments.test_morpion_sharded_training_v2 import (
        CODEC,
        Evaluation,
        Node,
    )

    dynamics = MorpionDynamics()
    state = CODEC.load_state_ref((0, ()))
    for _ in range(200):
        actions = dynamics.all_legal_actions(state)
        if not actions:
            break
        state = dynamics.step(state, actions[0]).next_state
    assert not dynamics.all_legal_actions(state)
    terminal = canonical_value.make_terminal_value(
        score=float(state.moves), over_event=OverEvent(outcome=Outcome.DRAW)
    )
    resolver = CheckpointStateResolver(
        CODEC, {0: AnchorCheckpointStatePayload(CODEC.dump_state_ref(state))}
    )
    node = Node(
        0,
        state.moves,
        CheckpointBackedStateHandle(resolver, 0),
        Evaluation(terminal, terminal if tree_present else None, True),
    )
    path, _ = scientific.save_scientific_training_generation(
        nodes=(node,),
        root_node_id="0",
        output_dir=tmp_path / "source",
        generation=1,
        producer=PRODUCER,
    )
    oracle = scientific.load_scientific_training_generation(path)
    adapter = MorpionActiveModelNodeReevaluationEvaluator(tmp_path / "model")
    adapter._evaluator = StateScore()
    with view(path) as actual:
        nodes = actual.load_nodes(("0",))
        assert nodes == oracle.snapshot.nodes
        assert nodes[0].is_terminal and nodes[0].is_exact
        assert actual.load_value_contracts(("0",)) == (oracle.value_contracts["0"],)
        expected = adapter.evaluate_patch_rows(oracle.snapshot, ("0",))
        if tree_present:

            def forbidden(_state: object) -> float:
                pytest.fail("terminal tree value must bypass model inference")

            adapter._evaluator.evaluate = forbidden
        rows = adapter.evaluate_node_window(nodes)
        assert rows == expected
        # PR #82 preserves terminal direct values when no tree value exists.
        assert rows[0].metadata["source"] == "terminal_existing_value"


@pytest.mark.fast
@pytest.mark.parametrize("target", ["DIRECT_VALUE", "TREE_VALUE", "EFFECTIVE_VALUE"])
def test_selected_target_contract_payload_and_signed_zero(
    tmp_path: Path, target: str
) -> None:
    """Retain declared target source, null tree, signed zero and tagged value line."""
    from dataclasses import replace

    from anemone.node_evaluation.common import canonical_value
    from anemone.node_evaluation.common.value_snapshot import NodeTargetSource

    from tests.environments.test_morpion_sharded_training_v2 import Evaluation

    node = nodes_fixture()[0]
    node.child_ids = ()
    value = replace(canonical_value.make_forced_value(score=-0.0), line=[("branch", 7)])
    node.tree_evaluation = Evaluation(value, None)
    path, _ = scientific.save_scientific_training_generation(
        nodes=(node,),
        root_node_id="0",
        output_dir=tmp_path / "source",
        generation=1,
        producer=PRODUCER,
        target_source=NodeTargetSource[target],
    )
    oracle = scientific.load_scientific_training_generation(path)
    with view(path) as actual:
        row = actual.load_nodes(("0",))[0]
        assert row == oracle.snapshot.nodes[0]
        assert row.direct_value_scalar is not None
        assert row.direct_value_scalar.hex() == "-0x0.0p+0"
        assert (row.target_value_scalar is None) == (target == "TREE_VALUE")
        if row.target_value_scalar is not None:
            assert row.target_value_scalar.hex() == "-0x0.0p+0"
        assert actual.load_value_contracts(("0",)) == (oracle.value_contracts["0"],)


def test_metadata_change_before_first_fingerprint_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never pair stale parsed provenance with a newer source fingerprint."""
    path = source(tmp_path / "source")
    original = ShardedTrainingGenerationView._prepare_scientific_source

    def changed_during_preparation(self: ShardedTrainingGenerationView) -> None:
        original(self)
        data = json.loads(path.read_text())
        data["snapshot"]["metadata"]["changed"] = "before first fingerprint"
        path.write_text(json.dumps(data))

    monkeypatch.setattr(
        ShardedTrainingGenerationView,
        "_prepare_scientific_source",
        changed_during_preparation,
    )
    with pytest.raises(ValueError, match="changed while preparing"):
        view(path)
    assert not list((tmp_path / "cache").glob("*.sqlite3"))


def test_crash_left_temporary_index_is_never_reused(tmp_path: Path) -> None:
    """A crash artifact without atomic publication cannot become a usable cache."""
    path = source(tmp_path / "source")
    cache = tmp_path / "cache"
    cache.mkdir()
    interrupted = cache / ".building-interrupted.sqlite3"
    interrupted.write_bytes(b"SQLite format 3\0interrupted")
    with view(path) as actual:
        assert not actual.index_reused
        assert actual.node_count() == 4
        assert actual.index_path != interrupted
    with view(path) as repeated:
        assert repeated.index_reused
    assert interrupted.read_bytes() == b"SQLite format 3\0interrupted"


def test_scientific_export_order_differs_from_unicode_cursor_order(
    tmp_path: Path,
) -> None:
    """Preserve export order and Python cursor order independently, even with NUL."""
    nodes = nodes_fixture()
    advance(nodes)
    ids = ("😀", "a\0z", "é", "A")
    for node in nodes:
        mutable: Any = node
        mutable.id = ids[node.id]
        mutable.parent_ids = tuple(ids[i] for i in node.parent_ids)
        mutable.child_ids = tuple(ids[i] for i in node.child_ids)
    ordered = (nodes[2], nodes[0], nodes[3], nodes[1])
    path, _ = scientific.save_scientific_training_generation(
        nodes=ordered,
        root_node_id=ids[0],
        output_dir=tmp_path / "source",
        generation=1,
        producer=PRODUCER,
    )
    oracle = scientific.load_scientific_training_generation(path)
    with view(path) as actual:
        assert tuple(actual.iter_nodes_in_export_order()) == oracle.snapshot.nodes
        assert tuple(n.node_id for n in oracle.snapshot.nodes) == tuple(
            str(n.id) for n in ordered
        )
        sorted_ids = tuple(sorted(ids))
        for cursor in (None, "missing", *sorted_ids):
            expected = select_reevaluation_node_window(
                sorted_ids, start_cursor=cursor, max_nodes=3
            )
            assert actual.select_sorted_node_window(cursor, 3) == expected
            assert actual.load_nodes(expected[0]) == tuple(
                next(n for n in oracle.snapshot.nodes if n.node_id == i)
                for i in expected[0]
            )
        for policy in get_args(PvFamilyTargetPolicy):
            args = MorpionBootstrapArgs(
                work_dir=tmp_path, dataset_family_target_policy=policy
            )
            expected = extract_rows_from_training_snapshot(
                args=args, snapshot=oracle.snapshot, generation=1
            )
            with disk_dataset_context(IndexedDatasetSource(actual), args) as context:
                stream = context.stream(1)
                assert stream.metadata == expected.metadata
                assert tuple(stream.rows) == expected.rows
