"""Closed storage layout: exact fields, source authority and offline failure fences."""

import json
import sqlite3
from dataclasses import replace
from typing import TYPE_CHECKING, Any, get_args

import pytest
from anemone.node_evaluation.common.canonical_value import make_estimate_value

from chipiron.environments.morpion.bootstrap.bootstrap_args import MorpionBootstrapArgs
from chipiron.environments.morpion.bootstrap.closed_scientific_snapshot import (
    load_closed_scientific_snapshot,
    save_closed_scientific_snapshot,
)
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
from chipiron.environments.morpion.bootstrap.scientific_publication_initialization import (
    initialize_scientific_v2_publication,
)
from chipiron.environments.morpion.bootstrap.sharded_training_reader import (
    ShardedTrainingGenerationView,
    is_sharded_generation_manifest,
)
from chipiron.environments.morpion.bootstrap.sharded_training_v2 import (
    _sha,
    load_scientific_training_generation,
    save_scientific_training_generation,
)
from chipiron.environments.morpion.bootstrap.streaming_dataset import (
    IndexedDatasetSource,
    disk_dataset_context,
)
from chipiron.environments.morpion.learning import (
    save_morpion_supervised_rows_streaming,
)
from chipiron.environments.morpion.learning.target_confidence import (
    SAFE_FAMILY_POLICIES,
    DatasetConfidenceError,
)
from tests.environments.morpion_durable_v2_helpers import (
    scientific_producer,
    setup_scientific_baseline,
)
from tests.environments.test_morpion_scientific_training_reader import StateScore
from tests.environments.test_morpion_sharded_training_v2 import (
    PRODUCER,
    advance,
    nodes_fixture,
)

if TYPE_CHECKING:
    from pathlib import Path


def closed(root: Path, nodes: Any = None, **kwargs: Any) -> Path:
    """Run closed for the isolated synthetic study."""
    return save_closed_scientific_snapshot(
        nodes=nodes_fixture() if nodes is None else nodes,
        root_node_id="0",
        destination=root,
        generation=1,
        producer=PRODUCER,
        **kwargs,
    )[0]


def view(path: Path) -> ShardedTrainingGenerationView:
    """Run view for the isolated synthetic study."""
    return ShardedTrainingGenerationView(
        path, cache_dir=path.parent.parent / "cache", allow_closed_snapshot=True
    )


def committed(root: Path, nodes: Any) -> tuple[Path, dict[str, str]]:
    """Existing v2 protocol, newly generated scalar checkpoint fixture only."""
    args, paths, state = setup_scientific_baseline(root, initialize=False)
    producer = scientific_producer(args)
    path, _ = save_scientific_training_generation(
        nodes=nodes,
        root_node_id=str(nodes[0].id),
        output_dir=root / "fixture-source",
        generation=1,
        producer=producer,
    )
    state = replace(
        state,
        tree_size_at_last_save=len(nodes),
        latest_tree_snapshot_path=paths.relative_to_work_dir(path),
    )
    state = initialize_scientific_v2_publication(paths, state, producer=producer)
    return paths.work_dir / state.latest_tree_snapshot_path, producer


def assert_reference(path: Path) -> None:
    """Run assert reference for the isolated synthetic study."""
    oracle = load_closed_scientific_snapshot(path)
    ids = tuple(n.node_id for n in oracle.snapshot.nodes)
    with view(path) as actual:
        assert tuple(actual.iter_nodes_in_export_order()) == oracle.snapshot.nodes
        assert actual.snapshot_metadata == oracle.snapshot.metadata
        assert actual.created_at_unix_s == oracle.snapshot.created_at_unix_s
        assert actual.provenance == oracle.provenance
        assert actual.load_value_contracts(ids) == tuple(
            oracle.value_contracts[i] for i in ids
        )
        assert actual.load_creation_generations(ids) == tuple(
            oracle.creation_generations[i] for i in ids
        )
        adapter = MorpionActiveModelNodeReevaluationEvaluator(path.parent / "model")
        adapter._evaluator = StateScore()
        for cursor in (None, "missing", *ids):
            selected = actual.select_sorted_node_window(cursor, 3)
            assert selected == select_reevaluation_node_window(
                tuple(sorted(ids)), start_cursor=cursor, max_nodes=3
            )
            nodes = actual.load_nodes(selected[0])
            expected = tuple(
                n for i in selected[0] for n in oracle.snapshot.nodes if n.node_id == i
            )
            assert node_values_to_patch_rows(nodes) == node_values_to_patch_rows(
                expected
            )
            assert adapter.evaluate_node_window(nodes) == adapter.evaluate_patch_rows(
                oracle.snapshot, selected[0]
            )


def test_initial_closed_exact_deterministic_and_independent(tmp_path: Path) -> None:
    """Verify initial closed exact deterministic and independent."""
    nodes = nodes_fixture()
    path = closed(
        tmp_path / "first", nodes, created_at_unix_s=123.5, metadata={"extra": None}
    )
    other = closed(
        tmp_path / "repeat", nodes, created_at_unix_s=123.5, metadata={"extra": None}
    )
    assert {p.name: p.read_bytes() for p in path.parent.iterdir()} == {
        p.name: p.read_bytes() for p in other.parent.iterdir()
    }
    assert set(p.name for p in path.parent.iterdir()) == {
        "generation.json",
        "records.jsonl",
        "updates.jsonl",
    }
    old, _ = save_scientific_training_generation(
        nodes=nodes,
        root_node_id="0",
        output_dir=tmp_path / "old",
        generation=1,
        producer=PRODUCER,
        created_at_unix_s=123.5,
        metadata={"extra": None},
    )
    oracle = load_scientific_training_generation(old)
    actual = load_closed_scientific_snapshot(path)
    assert (
        actual.snapshot == oracle.snapshot
        and actual.value_contracts == oracle.value_contracts
    )
    assert (path.parent / "updates.jsonl").read_bytes().splitlines()[1:] == (
        old.parent / "updates.jsonl"
    ).read_bytes().splitlines()[1:]
    moved = tmp_path / "standalone"
    path.parent.rename(moved)
    assert_reference(moved / path.name)
    with pytest.raises(ValueError, match="explicit"):
        ShardedTrainingGenerationView(other, cache_dir=tmp_path / "cache")
    with pytest.raises(ValueError, match="Unsupported explicit"):
        is_sharded_generation_manifest(other)


def test_committed_copy_changed_values_parents_and_new_origins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify committed copy changed values parents and new origins."""
    nodes = nodes_fixture()
    nodes[1].child_ids = ()
    source, producer = committed(tmp_path / "run", nodes[:3])
    nodes[1].child_ids = (3,)
    # Source graph fixture must be closed before initializing a smaller graph.
    advance(nodes)
    nodes[0].tree_evaluation.direct_value = make_estimate_value(score=42)
    from atomheart.games.morpion.checkpoints import MorpionStateCheckpointCodec

    original = MorpionStateCheckpointCodec.load_state_ref
    calls = []

    def counted(self: Any, ref: Any) -> Any:
        calls.append(ref)
        return original(self, ref)

    # Patch only for counting: the committed validator source identity is unchanged.
    monkeypatch.setattr(MorpionStateCheckpointCodec, "load_state_ref", counted)
    path, stats = save_closed_scientific_snapshot(
        nodes=nodes,
        root_node_id="0",
        destination=tmp_path / "next",
        generation=2,
        producer=producer,
        committed_source=source,
    )
    assert stats["copied_records"] == 3 and stats["new_state_encodings"] == 1
    assert len(calls) == 1
    parent_lines = (
        (source.parent / "records.jsonl").read_bytes().splitlines(keepends=True)[1:]
    )
    assert all(
        line
        in (path.parent / "records.jsonl").read_bytes().splitlines(keepends=True)[1:]
        for line in parent_lines
    )
    assert load_closed_scientific_snapshot(path).creation_generations == {
        "0": 1,
        "1": 1,
        "2": 1,
        "3": 2,
    }
    assert_reference(path)


@pytest.mark.parametrize("encoding", [None, "jsonl-zstd-1"])
@pytest.mark.parametrize("policy", get_args(PvFamilyTargetPolicy))
@pytest.mark.parametrize("confidence", ["legacy_v1", "selected_value_v2"])
def test_all_policies_rows_metadata_and_jsonl(
    tmp_path: Path, policy: str, confidence: str, encoding: str | None
) -> None:
    """Verify all policies rows metadata and jsonl."""
    nodes = nodes_fixture()
    advance(nodes)
    path = closed(tmp_path / "source", nodes, encoding=encoding)
    raw = closed(tmp_path / "raw", nodes) if encoding else path
    oracle = load_closed_scientific_snapshot(raw)
    with view(path) as actual:
        for backup in (False, True):
            for criteria in (
                {},
                {"require_exact_or_terminal": True},
                {"max_rows": 0},
                {"max_rows": 2},
                {"min_depth": 1},
            ):
                args = MorpionBootstrapArgs(
                    work_dir=tmp_path,
                    dataset_family_target_policy=policy,
                    dataset_confidence_policy=confidence,
                    use_backed_up_value=backup,
                    **criteria,
                )

                def extract(args: Any = args) -> Any:
                    return extract_rows_from_training_snapshot(
                        args=args,
                        snapshot=oracle.snapshot,
                        generation=1,
                        value_contracts=oracle.value_contracts,
                        source_provenance=oracle.provenance,
                    )

                if (
                    confidence == "selected_value_v2"
                    and policy not in SAFE_FAMILY_POLICIES
                ):
                    with pytest.raises(
                        DatasetConfidenceError, match="gates pv_exact_then"
                    ):
                        extract()
                    with (
                        pytest.raises(
                            DatasetConfidenceError, match="gates pv_exact_then"
                        ),
                        disk_dataset_context(IndexedDatasetSource(actual), args),
                    ):
                        pytest.fail("ungated exact family policy")
                    continue
                expected = extract()
                with disk_dataset_context(
                    IndexedDatasetSource(actual), args
                ) as context:
                    stream = context.stream(1)
                    assert tuple(stream.rows) == expected.rows
                    assert stream.metadata == expected.metadata
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


@pytest.mark.parametrize(
    "change",
    [
        "hash",
        "certainty",
        "topology",
        "origin",
        "duplicate",
        "missing",
        "codec",
        "producer",
        "layout",
        "extra",
        "null",
    ],
)
def test_corrupt_output_rejected_by_both_readers(tmp_path: Path, change: str) -> None:
    """Verify corrupt output rejected by both readers."""
    path = closed(tmp_path / "source")
    header = json.loads(path.read_bytes())
    role = "records" if change in ("origin", "duplicate", "null") else "updates"
    shard = path.parent / f"{role}.jsonl"
    lines = [json.loads(line) for line in shard.read_bytes().splitlines()]
    if change == "hash":
        shard.write_bytes(shard.read_bytes() + b" ")
    elif change in ("codec", "producer"):
        header["producer"][
            "state_ref_contract" if change == "codec" else "api_sha256"
        ] = "unsupported"
    elif change == "layout":
        header["storage_layout_version"] = 2
    elif change == "extra":
        header["creation_index"] = {}
    else:
        if change == "certainty":
            lines[1]["value_contract"]["certainties"]["target"] = "FORCED"
        elif change == "topology":
            lines[1]["child_ids"] = ["missing"]
        elif change == "origin":
            lines[1]["creation_generation"] = 2
        elif change == "duplicate":
            lines[2] = lines[1]
        elif change == "missing":
            del lines[1]["target_value_scalar"]
        elif change == "null":
            lines[1]["state_ref_payload"] = None
        shard.write_text("".join(json.dumps(row) + "\n" for row in lines))
        header["record_shard" if role == "records" else "update_shard"]["sha256"] = (
            _sha(shard)
        )
    path.write_text(json.dumps(header))
    for read in (load_closed_scientific_snapshot, view):
        with pytest.raises((ValueError, TypeError, KeyError, sqlite3.IntegrityError)):
            result = read(path)
            if isinstance(result, ShardedTrainingGenerationView):
                result.close()


def test_offline_source_rejected_and_interruption_leaves_no_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify offline source rejected and interruption leaves no snapshot."""
    path = closed(tmp_path / "offline")
    with pytest.raises(ValueError, match="actual committed"):
        save_closed_scientific_snapshot(
            nodes=nodes_fixture(),
            root_node_id="0",
            destination=tmp_path / "bad",
            generation=2,
            producer=PRODUCER,
            committed_source=path,
        )
    import chipiron.environments.morpion.bootstrap.closed_scientific_snapshot as implementation

    def interrupted(*args: Any, **kwargs: Any) -> Any:
        message = "interrupted"
        raise RuntimeError(message)

    with monkeypatch.context() as patch:
        patch.setattr(implementation, "_validate_update", interrupted)
        with pytest.raises(RuntimeError, match="interrupted"):
            closed(tmp_path / "retry")
    assert not (tmp_path / "retry").exists()
    assert not list(tmp_path.glob(".closed-*"))
    assert_reference(closed(tmp_path / "retry"))


@pytest.mark.parametrize("encoding", [None, "jsonl-zstd-1"])
def test_cache_corruption_reuse_and_source_change(
    tmp_path: Path, encoding: str | None
) -> None:
    """Verify cache corruption reuse and source change."""
    path = closed(tmp_path / "source", encoding=encoding)
    with view(path) as current:
        cache = current.index_path
    with view(path) as reused:
        assert reused.index_reused
    with sqlite3.connect(cache) as connection:
        connection.execute("UPDATE records SET payload='{}'")
    with view(path) as rebuilt:
        assert not rebuilt.index_reused
        (
            path.parent / ("records.jsonl.zst" if encoding else "records.jsonl")
        ).write_bytes(b"corrupt")
        with pytest.raises(ValueError, match="hash mismatch"):
            rebuilt.validate_source_identity()


def test_unchanged_copy_never_accesses_old_states(tmp_path: Path) -> None:
    """Verify unchanged copy never accesses old states."""
    nodes = nodes_fixture()
    source, producer = committed(tmp_path / "run", nodes)
    before = [n.accesses for n in nodes]
    outputs = []
    for name in ("next", "repeat"):
        path, stats = save_closed_scientific_snapshot(
            nodes=nodes,
            root_node_id="0",
            destination=tmp_path / name,
            generation=2,
            producer=producer,
            committed_source=source,
        )
        assert stats["copied_records"] == 4 and stats["new_state_encodings"] == 0
        assert [n.accesses for n in nodes] == before
        outputs.append({p.name: p.read_bytes() for p in path.parent.iterdir()})
    assert outputs[0] == outputs[1]
    # Deliberately break only this fresh fixture's old committed dependency.
    # The complete new snapshots own independent bytes and need no old source.
    (source.parent / "records.jsonl").write_bytes(b"unavailable old source")
    assert_reference(tmp_path / "next/generation.json")


def test_corrupt_committed_source_and_wrong_identity_refused(tmp_path: Path) -> None:
    """Verify corrupt committed source and wrong identity refused."""
    nodes = nodes_fixture()
    source, producer = committed(tmp_path / "run", nodes)
    with pytest.raises(ValueError, match="producer/codec"):
        save_closed_scientific_snapshot(
            nodes=nodes,
            root_node_id="0",
            destination=tmp_path / "wrong",
            generation=2,
            producer={**producer, "evaluator": "different-model"},
            committed_source=source,
        )
    # Only this newly generated temporary fixture is corrupted.
    shard = source.parent / "records.jsonl"
    shard.write_bytes(
        shard.read_bytes().replace(b"creation_generation", b"creation_generatioX")
    )
    with pytest.raises(ValueError):
        save_closed_scientific_snapshot(
            nodes=nodes,
            root_node_id="0",
            destination=tmp_path / "corrupt",
            generation=2,
            producer=producer,
            committed_source=source,
        )
    assert not (tmp_path / "corrupt").exists()


@pytest.mark.parametrize("encoding", [None, "jsonl-zstd-1"])
def test_real_forced35_estimate42_all_policies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, encoding: str | None
) -> None:
    """Verify real forced35 estimate42 all policies."""
    from tests.environments.test_morpion_value_certainty import (
        endgame,
        make_runner,
        model_patch,
    )

    late, _ = endgame.__wrapped__()
    runner = make_runner(monkeypatch, state=late)
    runner.apply_reevaluation_patch(model_patch(runner, monkeypatch, 42, tmp_path))
    path = closed(
        tmp_path / "source", tuple(runner.iter_profile_nodes()), encoding=encoding
    )
    oracle = load_closed_scientific_snapshot(path)
    assert oracle.value_contracts["0"]["certainties"] == {
        "direct": "ESTIMATE",
        "tree": "FORCED",
        "effective": "FORCED",
        "target": "FORCED",
    }
    with view(path) as actual:
        for policy in get_args(PvFamilyTargetPolicy):
            for backup in (False, True):
                for exact in (False, True):
                    args = MorpionBootstrapArgs(
                        work_dir=tmp_path,
                        dataset_confidence_policy="selected_value_v2",
                        dataset_family_target_policy=policy,
                        use_backed_up_value=backup,
                        require_exact_or_terminal=exact,
                    )
                    if policy not in SAFE_FAMILY_POLICIES:
                        with (
                            pytest.raises(DatasetConfidenceError),
                            disk_dataset_context(IndexedDatasetSource(actual), args),
                        ):
                            pytest.fail("exact-family proof gate missing")
                        continue
                    expected = extract_rows_from_training_snapshot(
                        args=args,
                        snapshot=oracle.snapshot,
                        generation=1,
                        value_contracts=oracle.value_contracts,
                        source_provenance=oracle.provenance,
                    )
                    with disk_dataset_context(
                        IndexedDatasetSource(actual), args
                    ) as context:
                        assert tuple(context.rows()) == expected.rows
                    assert len(expected.rows) == (1 if exact and not backup else 2)
                    terminal = expected.rows[-1]
                    assert (
                        terminal.is_terminal
                        and terminal.is_exact
                        and terminal.target_value == 35
                    )
                    if not exact or backup:
                        parent = expected.rows[0]
                        assert parent.is_exact is backup
                        assert parent.metadata["target_confidence"][
                            "raw_target_certainty"
                        ] == ("FORCED" if backup else "ESTIMATE")
                        assert parent.metadata["target_confidence"][
                            "raw_target_value"
                        ] == (35 if backup else 42)


@pytest.mark.parametrize("role", ["records", "updates"])
def test_corruption_before_descriptor_cannot_inherit_canonical_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str
) -> None:
    """Verify corruption before descriptor cannot inherit canonical proof."""
    import chipiron.environments.morpion.bootstrap.closed_scientific_snapshot as implementation

    original = implementation._sha

    def corrupt(path: Path, *args: Any) -> Any:
        if path.name == f"{role}.jsonl":
            # Valid JSON and valid canonical ref, but no longer the validated state.
            rows = [json.loads(line) for line in path.read_bytes().splitlines()]
            if role == "records":
                rows[1]["state_ref_payload"] = rows[2]["state_ref_payload"]
            else:
                rows[1]["metadata"] = {"tampered": True}
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return original(path, *args)

    monkeypatch.setattr(implementation, "_sha", corrupt)
    with pytest.raises(ValueError, match="differ from"):
        closed(tmp_path / "corrupt")
    assert not (tmp_path / "corrupt").exists()


def test_late_parent_and_current_membership(tmp_path: Path) -> None:
    """Verify late parent and current membership."""
    nodes = nodes_fixture()
    source, producer = committed(tmp_path / "run", nodes)
    advance(nodes)
    path, stats = save_closed_scientific_snapshot(
        nodes=nodes,
        root_node_id="0",
        destination=tmp_path / "late-parent",
        generation=2,
        producer=producer,
        committed_source=source,
    )
    assert stats["new_state_encodings"] == 0
    assert load_closed_scientific_snapshot(path).snapshot.nodes[-1].parent_ids == (
        "1",
        "2",
    )
    assert_reference(path)
    nodes[1].child_ids = ()
    nodes[2].child_ids = ()
    path, stats = save_closed_scientific_snapshot(
        nodes=nodes[:3],
        root_node_id="0",
        destination=tmp_path / "current-only",
        generation=2,
        producer=producer,
        committed_source=source,
    )
    assert stats["copied_records"] == 3
    assert_reference(path)


def test_stale_validator_falls_back_to_full_canonical_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify stale validator falls back to full canonical checks."""
    from atomheart.games.morpion.checkpoints import MorpionStateCheckpointCodec

    from chipiron.environments.morpion.bootstrap import (
        scientific_validation_evidence as evidence,
    )

    nodes = nodes_fixture()
    source, producer = committed(tmp_path / "run", nodes)
    original = MorpionStateCheckpointCodec.load_state_ref
    calls = []

    def counted(self: Any, ref: Any) -> Any:
        calls.append(ref)
        return original(self, ref)

    monkeypatch.setattr(evidence, "validator_identity", lambda: {"changed": True})
    monkeypatch.setattr(MorpionStateCheckpointCodec, "load_state_ref", counted)
    path, stats = save_closed_scientific_snapshot(
        nodes=nodes,
        root_node_id="0",
        destination=tmp_path / "next",
        generation=2,
        producer=producer,
        committed_source=source,
    )
    assert len(calls) == 4 and stats["new_state_encodings"] == 0
    assert_reference(path)


def test_multiple_creation_generations(tmp_path: Path) -> None:
    """Verify multiple creation generations."""
    from tests.environments.morpion_durable_v2_helpers import (
        ScientificPublicationRunner,
        publish_scientific_patch,
        run_scientific_publication,
        scientific_nodes,
    )

    args, paths, _ = setup_scientific_baseline(tmp_path / "run")
    runner = ScientificPublicationRunner()
    runner.grow_delta = 1
    publish_scientific_patch(paths)
    state = run_scientific_publication(args, runner)
    assert state.generation == 2
    nodes = scientific_nodes(
        json.loads((paths.work_dir / state.latest_runtime_checkpoint_path).read_bytes())
    )
    path, stats = save_closed_scientific_snapshot(
        nodes=nodes,
        root_node_id="root",
        destination=tmp_path / "closed",
        generation=3,
        producer=scientific_producer(args),
        committed_source=paths.work_dir / state.latest_tree_snapshot_path,
    )
    expected = json.loads(
        (paths.work_dir / state.latest_tree_snapshot_path).read_bytes()
    )["creation_index"]
    assert load_closed_scientific_snapshot(path).creation_generations == expected
    assert set(expected.values()) == {1, 2}
    assert stats["new_state_encodings"] == 0
    assert_reference(path)


def test_abrupt_writer_exit_is_not_a_snapshot_commit(tmp_path: Path) -> None:
    """Verify abrupt writer exit is not a snapshot commit."""
    import os
    import subprocess
    import sys

    destination = tmp_path / "snapshot"
    code = """
import os, sys
from pathlib import Path
from chipiron.environments.morpion.bootstrap import closed_scientific_snapshot as impl
from tests.environments.test_morpion_sharded_training_v2 import nodes_fixture, PRODUCER
impl._validate_update = lambda row: os._exit(91)
impl.save_closed_scientific_snapshot(nodes=nodes_fixture(), root_node_id='0', destination=Path(sys.argv[1]), generation=1, producer=PRODUCER)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(destination)],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        check=False,
    )
    assert result.returncode == 91
    assert not destination.exists()
    orphans = set(tmp_path.glob(".closed-*"))
    assert orphans
    assert_reference(closed(destination))
    assert orphans.issubset(set(tmp_path.glob(".closed-*")))


def test_error_after_atomic_rename_preserves_complete_offline_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify error after atomic rename preserves complete offline output."""
    import chipiron.environments.morpion.bootstrap.closed_scientific_snapshot as implementation

    original = implementation.publish_directory_if_absent

    def interrupted(*args: Any) -> None:
        original(*args)
        message = "interrupted after rename"
        raise OSError(message)

    with monkeypatch.context() as patch:
        patch.setattr(implementation, "publish_directory_if_absent", interrupted)
        with pytest.raises(OSError, match="after rename"):
            closed(tmp_path / "snapshot")
    assert_reference(tmp_path / "snapshot/generation.json")
    with pytest.raises(ValueError, match="already exists"):
        closed(tmp_path / "snapshot")
    assert not (tmp_path / "run_state.json").exists()


def test_declared_model_identity_is_retained_exactly(tmp_path: Path) -> None:
    """Verify declared model identity is retained exactly."""
    producer = {**PRODUCER, "model_sha256": "d" * 64, "model_generation": "7"}
    path, _ = save_closed_scientific_snapshot(
        nodes=nodes_fixture(),
        root_node_id="0",
        destination=tmp_path / "source",
        generation=1,
        producer=producer,
        metadata={"model": {"name": "fixture", "checkpoint_sha256": "e" * 64}},
    )
    oracle = load_closed_scientific_snapshot(path)
    assert oracle.provenance["producer"]["identity"] == producer
    assert oracle.snapshot.metadata["model"]["checkpoint_sha256"] == "e" * 64
    assert_reference(path)


def test_single_pass_nodes_preserve_complete_scientific_bytes(tmp_path: Path) -> None:
    """The writer never requests a length, retains nodes, or traverses twice."""
    nodes = nodes_fixture()
    first, _ = save_closed_scientific_snapshot(
        nodes=nodes,
        root_node_id="0",
        destination=tmp_path / "sequence",
        generation=1,
        producer=PRODUCER,
    )
    second, stats = save_closed_scientific_snapshot(
        nodes=iter(nodes),
        root_node_id="0",
        destination=tmp_path / "iterator",
        generation=1,
        producer=PRODUCER,
    )
    assert stats["node_count"] == len(nodes)
    for name in ("records.jsonl", "updates.jsonl", "generation.json"):
        assert (first.parent / name).read_bytes() == (second.parent / name).read_bytes()
