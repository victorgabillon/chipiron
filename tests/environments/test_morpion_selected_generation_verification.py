"""Selected closures are authoritative; discovery is advisory and audit is explicit."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

import chipiron.environments.morpion.bootstrap.generation_publication_io as io
from chipiron.environments.morpion.bootstrap.pipeline.manifests import (
    pinned_generation_manifest,
)
from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
    load_pipeline_manifest,
    save_pipeline_manifest,
)
from chipiron.environments.morpion.bootstrap.pipeline_orchestrator import (
    load_available_pipeline_manifests,
    select_next_claimable_dataset_generation,
    select_next_claimable_training_generation,
)
from chipiron.environments.morpion.bootstrap.reevaluation_worker import (
    resolve_latest_reevaluation_tree_snapshot,
)
from tests.environments.morpion_durable_v2_helpers import (
    ScientificPublicationRunner,
    run_scientific_publication,
)
from tests.environments.test_morpion_discovery_projection import workspace

if TYPE_CHECKING:
    from pathlib import Path


def assert_pinned(root: Path, record: dict) -> None:
    """Check that an independent exclusive open cannot acquire the shared pin."""
    with (
        pytest.raises(BlockingIOError),
        io.capsule_lease(
            root,
            record["id"],
            exclusive=True,
            nonblocking=True,
            version=record.get("publication_version", 1),
        ),
    ):
        pass


@pytest.mark.parametrize("version", [1, 2])
def test_metadata_selection_then_one_complete_closure(
    tmp_path: Path, version: int
) -> None:
    """No validation while reading eligibility; one selected closure under a pin."""
    _, paths, state = workspace(tmp_path, version)
    records = io.publication_records(state.metadata)
    with patch.object(io, "validate_capsule", wraps=io.validate_capsule) as validate:
        manifests = load_available_pipeline_manifests(paths)
        assert list(manifests) == [1, 2]
        assert select_next_claimable_dataset_generation(paths, manifests) == 2
        assert select_next_claimable_training_generation(paths, manifests) is None
        assert not validate.called
        with pinned_generation_manifest(paths=paths, generation=2) as selected:
            assert selected == manifests[2]
            assert_pinned(tmp_path, records[-1])
        assert validate.call_count == 1
        assert validate.call_args.args == (tmp_path, records[-1])
    with io.capsule_lease(
        tmp_path, records[-1]["id"], exclusive=True, nonblocking=True, version=version
    ):
        pass
    with patch.object(io, "validate_capsule", wraps=io.validate_capsule) as validate:
        assert resolve_latest_reevaluation_tree_snapshot(paths) == (
            2,
            tmp_path / state.latest_tree_snapshot_path,
        )
        assert validate.call_count == 1


@pytest.mark.parametrize("version", [1, 2])
def test_unrelated_checkpoint_only_fails_explicit_audit(
    tmp_path: Path, version: int
) -> None:
    """An independent selected generation remains usable; audit still detects damage."""
    _, paths, state = workspace(tmp_path, version)
    records = io.publication_records(state.metadata)
    root = io.capsule_path(tmp_path, records[0]["id"], version)
    seal = io.read_object(root / io.capsule_seal(version))
    entry = next(e for e in seal["files"] if "checkpoint" in e["path"])
    damaged = root / entry["path"]
    damaged.write_bytes(damaged.read_bytes() + b" ")
    assert (
        select_next_claimable_dataset_generation(
            paths, load_available_pipeline_manifests(paths)
        )
        == 2
    )
    with pinned_generation_manifest(paths=paths, generation=2):
        pass
    assert resolve_latest_reevaluation_tree_snapshot(paths)[0] == 2
    # Full audit uses committed roots even when historical discovery is absent.
    paths.pipeline_manifest_path_for_generation(1).unlink()
    with pytest.raises(io.GenerationPublicationError, match="changed"):
        io.audit_generation_publications(tmp_path)


@pytest.mark.parametrize(
    "dependency",
    ["historical_record", "historical_update", "checkpoint", "model", "seal", "source"],
)
@pytest.mark.parametrize("failure", ["corrupt", "missing"])
def test_selected_dependency_fails_closed(
    tmp_path: Path, dependency: str, failure: str
) -> None:
    """Every sealed dependency is checked, including shared old immutable shards."""
    _, paths, state = workspace(tmp_path)
    records = io.publication_records(state.metadata)
    root = io.capsule_path(tmp_path, records[-1]["id"], 2)
    seal = io.read_object(root / io.capsule_seal(2))
    if dependency == "historical_record":
        selected = root / "tree_exports_sharded/generation_000001/records.jsonl"
        old = io.capsule_path(tmp_path, records[0]["id"], 2) / selected.relative_to(
            root
        )
        assert old.stat().st_ino == selected.stat().st_ino
        damaged = old  # Damage the historical name; selected closure must notice.
    elif dependency == "historical_update":
        damaged = root / "tree_exports_sharded/generation_000001/updates.jsonl"
    elif dependency in {"checkpoint", "model"}:
        damaged = root / next(
            e["path"] for e in seal["files"] if dependency in e["path"]
        )
    elif dependency == "seal":
        damaged = root / io.capsule_seal(2)
    else:
        damaged = tmp_path / state.latest_tree_snapshot_path
    if failure == "missing":
        # Removing only the old hardlink leaves selected bytes intact; remove the
        # selected reference here to exercise a truly missing dependency.
        if dependency == "historical_record":
            damaged = selected
        damaged.unlink()
    else:
        damaged.write_bytes(damaged.read_bytes() + b" ")
    manifests = load_available_pipeline_manifests(paths)
    assert select_next_claimable_dataset_generation(paths, manifests) == 2
    with (
        pytest.raises((ValueError, FileNotFoundError)),
        pinned_generation_manifest(paths=paths, generation=2),
    ):
        pytest.fail("Corrupted selected closure exposed to consumer")
    with pytest.raises((ValueError, FileNotFoundError)):
        resolve_latest_reevaluation_tree_snapshot(paths)  # Never fall back to G1.


@pytest.mark.parametrize(
    "change", ["projection", "path", "generation", "ancestry", "head", "missing_state"]
)
def test_selected_authority_is_reobserved(tmp_path: Path, change: str) -> None:
    """Advisory metadata never establishes membership or replaces exact authority."""
    _, paths, _ = workspace(tmp_path)
    selected = load_available_pipeline_manifests(paths)[2]
    if change == "projection":
        selected.metadata["pipeline_discovery"]["publication"]["descriptor_sha256"] = (
            "0" * 64
        )
    elif change == "path":
        selected = replace(selected, tree_snapshot_path="substitute.json")
    elif change == "generation":
        selected = replace(selected, generation=3, metadata={})
    elif change == "missing_state":
        paths.run_state_path.unlink()
    else:
        state = json.loads(paths.run_state_path.read_bytes())
        if change == "ancestry":
            state["metadata"][io.PUBLICATION_KEY]["generations"][-1]["parent_id"] = (
                "0" * 32
            )
        else:
            state["latest_runtime_checkpoint_path"] = "older-checkpoint.json"
        paths.run_state_path.write_text(json.dumps(state))
    with (
        pytest.raises((ValueError, FileNotFoundError)),
        io.manifest_read_pin(tmp_path, selected),
    ):
        pytest.fail("Stale/forged discovery was accepted")


def test_authority_read_follows_pin_and_concurrent_append_is_safe(
    tmp_path: Path,
) -> None:
    """A committed writer can append while readers retain the exact older capsule."""
    args, paths, state = workspace(tmp_path)
    record = io.publication_records(state.metadata)[-1]
    original = io.read_object
    observed = []

    def read(path: Path) -> dict:
        if path == paths.run_state_path:
            assert_pinned(tmp_path, record)
            observed.append(path)
        return original(path)

    with (
        patch.object(io, "read_object", read),
        pinned_generation_manifest(paths=paths, generation=2),
    ):
        assert observed == [paths.run_state_path]
    with pinned_generation_manifest(paths=paths, generation=2) as selected:
        from tests.environments.morpion_patch_receipt_helpers import (
            fixture_patch,
            publish_fixture_patch,
        )

        next_patch = replace(
            fixture_patch(),
            patch_id="next-patch",
            tree_generation=2,
            metadata={
                "scientific_publication": io.committed_scientific_source(
                    tmp_path / state.latest_tree_snapshot_path
                )
            },
        )
        publish_fixture_patch(paths, next_patch)
        newer = run_scientific_publication(args, ScientificPublicationRunner())
        assert newer.generation == 3
        io.validate_manifest_publication(tmp_path, selected)
        assert_pinned(tmp_path, record)
    assert (
        select_next_claimable_dataset_generation(
            paths, load_available_pipeline_manifests(paths)
        )
        == 3
    )


@pytest.mark.parametrize("streaming", [False, True])
def test_reevaluation_pin_survives_inference_and_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, streaming: bool
) -> None:
    """The existing lifetime pin covers evaluator calls, patch creation and cursor save."""
    import chipiron.environments.morpion.bootstrap.reevaluation_worker as worker
    from chipiron.environments.morpion.bootstrap.sharded_training_v2 import (
        load_scientific_training_generation,
    )
    from tests.environments.test_morpion_streaming_reevaluation import run_once

    _, paths, state = workspace(tmp_path)
    record = io.publication_records(state.metadata)[-1]
    oracle = load_scientific_training_generation(
        tmp_path / state.latest_tree_snapshot_path
    )
    monkeypatch.setattr(
        worker, "load_reevaluation_training_tree_snapshot", lambda _: oracle.snapshot
    )
    events = []

    class Evaluator:
        def evaluate_node_window(self, nodes: Any) -> Any:
            assert_pinned(tmp_path, record)
            events.append("inference")
            return worker.node_values_to_patch_rows(nodes)

    for name in ("_save_reevaluation_patch_exclusive", "save_reevaluation_cursor"):
        actual = getattr(worker, name)

        def pinned(
            *args: Any, actual: Any = actual, name: str = name, **kwargs: Any
        ) -> Any:
            assert_pinned(tmp_path, record)
            events.append(name)
            return actual(*args, **kwargs)

        monkeypatch.setattr(worker, name, pinned)
    result = run_once(
        paths, streaming=streaming, allow_scientific_v2=True, evaluator=Evaluator()
    )
    assert result.patch_written
    assert events == [
        "inference",
        "_save_reevaluation_patch_exclusive",
        "save_reevaluation_cursor",
    ]
    with io.capsule_lease(
        tmp_path, record["id"], exclusive=True, nonblocking=True, version=2
    ):
        pass


def test_reevaluation_source_change_during_inference_does_not_publish(
    tmp_path: Path,
) -> None:
    """A pin prevents collection, not malicious writes: the boundary recheck is real."""
    import chipiron.environments.morpion.bootstrap.reevaluation_worker as worker
    from tests.environments.test_morpion_streaming_reevaluation import run_once

    _, paths, state = workspace(tmp_path)
    checkpoint = tmp_path / state.latest_runtime_checkpoint_path

    class Evaluator:
        def evaluate_node_window(self, nodes: Any) -> Any:
            checkpoint.write_bytes(checkpoint.read_bytes() + b" ")
            return worker.node_values_to_patch_rows(nodes)

    before = paths.pipeline_reevaluation_cursor_path.read_bytes()
    with pytest.raises(ValueError, match="changed"):
        run_once(paths, allow_scientific_v2=True, evaluator=Evaluator())
    assert not paths.pipeline_reevaluation_patch_path.exists()
    assert paths.pipeline_reevaluation_cursor_path.read_bytes() == before


def test_dataset_pin_and_precommit_recheck(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed dependency cannot publish rows or success, even after reconstruction."""
    import chipiron.environments.morpion.bootstrap.pipeline.stages as stages

    args, paths, state = workspace(tmp_path)
    args = replace(args, dataset_confidence_policy="selected_value_v2")
    record = io.publication_records(state.metadata)[-1]
    actual = stages.save_morpion_supervised_rows_streaming
    checkpoint = tmp_path / state.latest_runtime_checkpoint_path

    def changed(**kwargs: Any) -> Any:
        assert_pinned(tmp_path, record)
        checkpoint.write_bytes(checkpoint.read_bytes() + b" ")
        return actual(**kwargs)

    monkeypatch.setattr(stages, "save_morpion_supervised_rows_streaming", changed)
    with pytest.raises(ValueError, match="changed"):
        stages.run_pipeline_dataset_stage(
            args,
            generation=2,
            use_streaming_sharded_reader=True,
            allow_scientific_v2=True,
            reader_cache_dir=tmp_path / "cache",
        )
    assert not paths.rows_jsonl_path_for_generation(2).exists()
    assert (
        load_pipeline_manifest(
            paths.pipeline_manifest_path_for_generation(2)
        ).dataset_status
        == "failed"
    )


def test_training_entry_retains_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Training gets the same selected authoritative scope as Dataset."""
    import chipiron.environments.morpion.bootstrap.pipeline.stages as stages

    args, paths, state = workspace(tmp_path)
    record = io.publication_records(state.metadata)[-1]
    path = paths.pipeline_manifest_path_for_generation(2)
    manifest = replace(
        load_pipeline_manifest(path), dataset_status="done", rows_path="synthetic.jsonl"
    )
    save_pipeline_manifest(manifest, path)

    def train(_args: Any, **kwargs: Any) -> Any:
        assert kwargs["manifest"] == manifest
        assert_pinned(tmp_path, record)
        return manifest

    monkeypatch.setattr(stages, "_run_pipeline_training_stage", train)
    assert stages.run_pipeline_training_stage(args, generation=2) == manifest


def test_generation_label_substitution_cannot_select_another_source(
    tmp_path: Path,
) -> None:
    """A valid G1 manifest copied under G2 must not relabel its scientific source."""
    _, paths, _ = workspace(tmp_path)
    first = load_pipeline_manifest(paths.pipeline_manifest_path_for_generation(1))
    save_pipeline_manifest(first, paths.pipeline_manifest_path_for_generation(2))
    assert (
        select_next_claimable_dataset_generation(
            paths, load_available_pipeline_manifests(paths)
        )
        == 2
    )
    with (
        pytest.raises(ValueError, match="generation mismatch"),
        pinned_generation_manifest(paths=paths, generation=2),
    ):
        pytest.fail("A source from another generation was exposed")
    with pytest.raises(ValueError, match="generation mismatch"):
        resolve_latest_reevaluation_tree_snapshot(paths)
