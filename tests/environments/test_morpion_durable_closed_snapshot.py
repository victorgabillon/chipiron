"""The closed-layout adapter shares one durable capsule/receipt transaction."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, get_args

import pytest

from chipiron.environments.morpion.bootstrap.checkpoint_patch_receipts import (
    receipt_ledger,
)
from chipiron.environments.morpion.bootstrap.closed_scientific_snapshot import (
    load_closed_scientific_snapshot,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    PUBLICATION_KEY,
    GenerationPublicationError,
    capsule_lease,
    capsule_path,
    publication_records,
    validate_capsule,
)
from chipiron.environments.morpion.bootstrap.pv_family_targets import (
    PvFamilyTargetPolicy,
)
from chipiron.environments.morpion.bootstrap.sharded_training_reader import (
    ShardedTrainingGenerationView,
)
from tests.environments.morpion_durable_closed_helpers import (
    ClosedPublicationRunner,
    run_closed_publication,
    setup_closed_baseline,
)
from tests.environments.morpion_durable_v2_helpers import publish_scientific_patch


def test_closed_commit_retry(tmp_path: Path) -> None:
    """Verify closed commit retry."""
    args, paths, initial = setup_closed_baseline(tmp_path)
    publish_scientific_patch(paths)
    runner = ClosedPublicationRunner()
    runner.grow_delta = 1
    state = run_closed_publication(args, runner)
    assert state.generation == 2 and runner.apply_calls == 1
    assert state.metadata[PUBLICATION_KEY]["schema_version"] == 3
    assert set(receipt_ledger(state)) == {"patch-1"}
    ref = load_closed_scientific_snapshot(tmp_path / state.latest_tree_snapshot_path)
    assert ref.creation_generations == {"root": 1, "leaf": 1, "new": 2}
    assert ref.snapshot.nodes[1].parent_ids == ("root", "new")
    with ShardedTrainingGenerationView(
        tmp_path / state.latest_tree_snapshot_path,
        allow_closed_snapshot=True,
        cache_dir=tmp_path / "cache",
    ) as view:
        assert tuple(view.iter_nodes_in_export_order()) == ref.snapshot.nodes
        assert view.provenance == ref.provenance
    for record in publication_records(state.metadata):
        seal = validate_capsule(tmp_path, record)
        scientific = [
            e["path"]
            for e in seal["files"]
            if e["path"].startswith("scientific_snapshot/")
        ]
        assert sorted(scientific) == [
            "scientific_snapshot/generation.json",
            "scientific_snapshot/records.jsonl",
            "scientific_snapshot/updates.jsonl",
        ]
        assert "catalog_sha256" not in record["scientific_identity"]
    restarted = ClosedPublicationRunner()
    recovered = run_closed_publication(args, restarted, max_cycles=0)
    assert recovered.generation == 2 and restarted.apply_calls == 0
    assert not paths.pipeline_reevaluation_patch_path.exists()
    from tests.environments.morpion_durable_closed_helpers import publish_closed_patch

    publish_closed_patch(paths, "patch-2")
    third_runner = ClosedPublicationRunner()
    third_runner.grow_delta = 1
    third = run_closed_publication(args, third_runner)
    assert third.generation == 3
    assert load_closed_scientific_snapshot(
        tmp_path / initial.latest_tree_snapshot_path
    ).creation_generations == {"root": 1, "leaf": 1}


@pytest.mark.parametrize("policy", get_args(PvFamilyTargetPolicy))
@pytest.mark.parametrize("confidence", ["legacy_v1", "selected_value_v2"])
def test_closed_dataset_all_policies(
    tmp_path: Path, policy: str, confidence: str
) -> None:
    """Verify closed dataset all policies."""
    from chipiron.environments.morpion.bootstrap.cycle_dataset import (
        extract_rows_from_training_snapshot,
    )
    from chipiron.environments.morpion.bootstrap.streaming_dataset import (
        IndexedDatasetSource,
        disk_dataset_context,
    )
    from chipiron.environments.morpion.learning import (
        save_morpion_supervised_rows_streaming,
    )

    args, _paths, state = setup_closed_baseline(tmp_path)
    args = replace(
        args, dataset_family_target_policy=policy, dataset_confidence_policy=confidence
    )
    source = tmp_path / state.latest_tree_snapshot_path
    ref = load_closed_scientific_snapshot(source)
    with ShardedTrainingGenerationView(
        source, allow_closed_snapshot=True, cache_dir=tmp_path / "cache"
    ) as view:
        kwargs = dict(
            args=args,
            snapshot=ref.snapshot,
            generation=1,
            value_contracts=ref.value_contracts,
            source_provenance=ref.provenance,
        )
        if confidence == "selected_value_v2" and policy.startswith("pv_exact_then_"):
            with pytest.raises(ValueError, match="gates pv_exact_then"):
                extract_rows_from_training_snapshot(**kwargs)
            with (
                pytest.raises(ValueError, match="gates pv_exact_then"),
                disk_dataset_context(IndexedDatasetSource(view), args),
            ):
                pass
            return
        expected = extract_rows_from_training_snapshot(**kwargs)
        with disk_dataset_context(IndexedDatasetSource(view), args) as ctx:
            stream = ctx.stream(1)
            assert (
                tuple(stream.rows) == expected.rows
                and stream.metadata == expected.metadata
            )
            save_morpion_supervised_rows_streaming(
                rows=ctx.rows(),
                metadata=stream.metadata,
                path=tmp_path / "indexed.jsonl",
            )
        save_morpion_supervised_rows_streaming(
            rows=expected.rows,
            metadata=expected.metadata,
            path=tmp_path / "reference.jsonl",
        )
        assert (tmp_path / "indexed.jsonl").read_bytes() == (
            tmp_path / "reference.jsonl"
        ).read_bytes()


@pytest.mark.parametrize(
    "target", ["records", "updates", "checkpoint", "model", "seal"]
)
def test_committed_corruption_fails_even_with_cache(
    tmp_path: Path, target: str
) -> None:
    """Verify committed corruption fails even with cache."""
    args, paths, state = setup_closed_baseline(tmp_path)
    source = tmp_path / state.latest_tree_snapshot_path
    with ShardedTrainingGenerationView(
        source, allow_closed_snapshot=True, cache_dir=tmp_path / "cache"
    ):
        pass
    record = publication_records(state.metadata)[0]
    seal = validate_capsule(tmp_path, record)
    victims = {
        "records": source.parent / "records.jsonl",
        "updates": source.parent / "updates.jsonl",
        "checkpoint": tmp_path / state.latest_runtime_checkpoint_path,
        "seal": tmp_path / record["descriptor_path"],
        "model": capsule_path(tmp_path, record["id"], 3)
        / seal["model_pins"][0]["path"]
        / "param.pt",
    }
    with victims[target].open("ab") as f:
        f.write(b"corrupt")
    with pytest.raises((ValueError, OSError)):
        run_closed_publication(args, ClosedPublicationRunner(), max_cycles=0)
    with pytest.raises((ValueError, OSError)):
        ShardedTrainingGenerationView(
            source, allow_closed_snapshot=True, cache_dir=tmp_path / "cache"
        )
    assert paths.run_state_path.exists()


def test_previous_reader_pinned_across_new_head(tmp_path: Path) -> None:
    """Verify previous reader pinned across new head."""
    args, paths, state = setup_closed_baseline(tmp_path)
    publish_scientific_patch(paths)
    source = tmp_path / state.latest_tree_snapshot_path
    record = publication_records(state.metadata)[0]
    with ShardedTrainingGenerationView(
        source, allow_closed_snapshot=True, cache_dir=tmp_path / "cache"
    ) as view:
        expected = tuple(view.iter_nodes_in_export_order())
        runner = ClosedPublicationRunner()
        runner.grow_delta = 1
        current = run_closed_publication(args, runner)
        assert current.generation == 2
        with (
            pytest.raises(BlockingIOError),
            capsule_lease(
                tmp_path, record["id"], version=3, exclusive=True, nonblocking=True
            ),
        ):
            pass
        assert tuple(view.iter_nodes_in_export_order()) == expected
        view.validate_source_identity()
    with capsule_lease(
        tmp_path, record["id"], version=3, exclusive=True, nonblocking=True
    ):
        pass


def test_concurrent_growth_owner_and_source_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify concurrent growth owner and source pin."""
    import subprocess
    import sys

    import chipiron.environments.morpion.bootstrap.generation_publication as pub

    args, paths, initial = setup_closed_baseline(tmp_path)
    publish_scientific_patch(paths)
    record = publication_records(initial.metadata)[0]
    original = pub.DurableGenerationPublication.publish

    def publish(self: Any, state: Any, **kwargs: Any) -> Any:
        """Publish."""
        with (
            pytest.raises(BlockingIOError),
            capsule_lease(
                tmp_path, record["id"], version=3, exclusive=True, nonblocking=True
            ),
        ):
            pass
        code = "from pathlib import Path; from chipiron.environments.morpion.bootstrap import MorpionBootstrapPaths; from chipiron.environments.morpion.bootstrap.generation_publication_io import growth_receipt_ownership; import sys;\nwith growth_receipt_ownership(MorpionBootstrapPaths.from_work_dir(Path(sys.argv[1]))): pass"
        result = subprocess.run(
            [sys.executable, "-c", code, str(tmp_path)], capture_output=True, timeout=20
        )
        assert result.returncode != 0 and b"BlockingIOError" in result.stderr
        return original(self, state, **kwargs)

    monkeypatch.setattr(pub.DurableGenerationPublication, "publish", publish)
    runner = ClosedPublicationRunner()
    runner.grow_delta = 1
    run_closed_publication(args, runner)
    with capsule_lease(
        tmp_path, record["id"], version=3, exclusive=True, nonblocking=True
    ):
        pass


def test_private_and_published_uncommitted_are_invisible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify private and published uncommitted are invisible."""
    import chipiron.environments.morpion.bootstrap.generation_publication as pub

    args, paths, _state = setup_closed_baseline(tmp_path)
    publish_scientific_patch(paths)
    original = pub.publish_directory_if_absent

    def publish(staging: Path, final: Path, root: Path) -> Any:
        """Publish."""
        if final.parent.name != "objects":
            return original(staging, final, root)
        for p in [staging / "scientific_snapshot/generation.json"]:
            with pytest.raises(GenerationPublicationError):
                load_closed_scientific_snapshot(p)
        original(staging, final, root)
        with pytest.raises(GenerationPublicationError):
            load_closed_scientific_snapshot(
                final / "scientific_snapshot/generation.json"
            )
        assert (
            publication_records(
                json.loads(paths.run_state_path.read_text())["metadata"]
            )[-1]["generation"]
            == 1
        )
        return None

    monkeypatch.setattr(pub, "publish_directory_if_absent", publish)
    runner = ClosedPublicationRunner()
    runner.grow_delta = 1
    run_closed_publication(args, runner)


@pytest.mark.parametrize(
    "change",
    [
        "record",
        "new_record",
        "updates",
        "producer",
        "codec",
        "checkpoint",
        "model",
        "source",
        "transfer",
    ],
)
def test_changed_attempt_or_source_never_commits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """Verify changed attempt or source never commits."""
    import chipiron.environments.morpion.bootstrap.generation_publication as pub

    args, paths, initial = setup_closed_baseline(tmp_path)
    publish_scientific_patch(paths)
    previous = paths.run_state_path.read_bytes()
    original = pub.DurableGenerationPublication.publish

    def publish(self: Any, state: Any, **kwargs: Any) -> Any:
        """Publish."""
        source = tmp_path / state.latest_tree_snapshot_path
        if change in {"record", "new_record", "updates"}:
            path = source.parent / (
                "updates.jsonl" if change == "updates" else "records.jsonl"
            )
            rows = path.read_text().splitlines()
            i = 1 if change == "new_record" else len(rows) - 1
            row = json.loads(rows[i])
            row["unexpected"] = True
            rows[i] = json.dumps(row)
            path.write_text("\n".join(rows) + "\n")
        elif change in {"producer", "codec"}:
            data = json.loads(source.read_text())
            if change == "producer":
                data["producer"]["identity"]["revision"] = "different"
            else:
                data["producer"]["state_ref_contract"] = "unknown"
            source.write_text(json.dumps(data))
        elif change == "checkpoint":
            state = replace(
                state,
                latest_runtime_checkpoint_path=initial.latest_runtime_checkpoint_path,
            )
        elif change == "model":
            self.model_pins[0]["sha256"] = "0" * 64
        elif change == "source":
            with (tmp_path / initial.latest_tree_snapshot_path).open("ab") as f:
                f.write(b" ")
        elif change == "transfer":
            self.closed_validation["manifest_sha256"] = "0" * 64
        return original(self, state, **kwargs)

    monkeypatch.setattr(pub.DurableGenerationPublication, "publish", publish)
    runner = ClosedPublicationRunner()
    runner.grow_delta = 1
    with pytest.raises((ValueError, OSError)):
        run_closed_publication(args, runner)
    assert paths.run_state_path.read_bytes() == previous


def test_unchanged_canonical_checks_are_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify unchanged canonical checks are reused."""
    from atomheart.games.morpion.checkpoints import MorpionStateCheckpointCodec

    args, paths, _ = setup_closed_baseline(tmp_path)
    publish_scientific_patch(paths)
    calls = []
    original = MorpionStateCheckpointCodec.load_state_ref

    def load(self: Any, *a: Any, **kw: Any) -> Any:
        """Load."""
        calls.append(1)
        return original(self, *a, **kw)

    monkeypatch.setattr(MorpionStateCheckpointCodec, "load_state_ref", load)
    runner = ClosedPublicationRunner()
    runner.grow_delta = 1
    run_closed_publication(args, runner)
    assert len(calls) == 1  # Only the new node, no copied canonical references.
    calls.clear()
    from tests.environments.morpion_durable_closed_helpers import publish_closed_patch

    publish_closed_patch(paths, "patch-next")
    runner = ClosedPublicationRunner()
    runner.grow_delta = 1
    run_closed_publication(args, runner)
    assert calls == []


@pytest.mark.parametrize("target", ["records", "checkpoint", "model", "config"])
def test_damage_after_seal_before_commit_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    """Verify damage after seal before commit is rejected."""
    import chipiron.environments.morpion.bootstrap.generation_publication as pub

    args, paths, _initial = setup_closed_baseline(tmp_path)
    publish_scientific_patch(paths)
    before = paths.run_state_path.read_bytes()
    original = pub.publish_directory_if_absent

    def publish(staging: Path, final: Path, root: Path) -> Any:
        """Publish."""
        original(staging, final, root)
        if final.parent.name == "objects":
            path = {
                "records": final / "scientific_snapshot/records.jsonl",
                "checkpoint": next((final / "checkpoint").iterdir()),
                "model": next((final / "models").rglob("param.pt")),
                "config": final / "bootstrap_config.json",
            }[target]
            with path.open("ab") as f:
                f.write(b"corrupt")

    monkeypatch.setattr(pub, "publish_directory_if_absent", publish)
    with pytest.raises((ValueError, OSError)):
        run_closed_publication(args, ClosedPublicationRunner())
    assert paths.run_state_path.read_bytes() == before


def test_default_routing_and_old_publication_refuse_closed(tmp_path: Path) -> None:
    """Verify default routing and old publication refuse closed."""
    from chipiron.environments.morpion.bootstrap.sharded_training_reader import (
        is_sharded_generation_manifest,
    )
    from tests.environments.morpion_durable_v2_helpers import run_scientific_publication

    args, _paths, state = setup_closed_baseline(tmp_path)
    source = tmp_path / state.latest_tree_snapshot_path
    with pytest.raises(ValueError, match="explicit allow_closed_snapshot"):
        ShardedTrainingGenerationView(source, cache_dir=tmp_path / "cache")
    with pytest.raises(ValueError, match="matching publication version"):
        run_scientific_publication(args, ClosedPublicationRunner(), max_cycles=0)
    with pytest.raises(ValueError, match="Unsupported explicit training-export format"):
        is_sharded_generation_manifest(source)


def test_discovery_repair_and_stale_projection(tmp_path: Path) -> None:
    """Verify discovery repair and stale projection."""
    from chipiron.environments.morpion.bootstrap.generation_publication_io import (
        validate_manifest_publication,
    )
    from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
        load_pipeline_manifest,
    )

    args, paths, state = setup_closed_baseline(tmp_path)
    path = paths.pipeline_manifest_path_for_generation(1)
    expected = path.read_bytes()
    path.unlink()
    assert (
        run_closed_publication(args, ClosedPublicationRunner(), max_cycles=0) == state
    )
    assert path.read_bytes() == expected
    data = json.loads(expected)
    data["tree_snapshot_path"] = "genesis-closed/generation.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        validate_manifest_publication(tmp_path, load_pipeline_manifest(path))
    with pytest.raises(ValueError):
        run_closed_publication(args, ClosedPublicationRunner(), max_cycles=0)


@pytest.mark.parametrize(
    "boundary",
    ["before_capsule", "after_capsule", "before_state", "after_state", "discovery"],
)
def test_closed_genesis_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    """Verify closed genesis retry."""
    import os

    from chipiron.environments.morpion.bootstrap import generation_publication as pub
    from chipiron.environments.morpion.bootstrap import (
        scientific_publication_initialization as init,
    )
    from chipiron.environments.morpion.bootstrap.checkpoint_patch_receipts import (
        GrowthPatchReceiptSession,
    )
    from chipiron.environments.morpion.bootstrap.generation_publication_io import (
        artifact_digest,
    )
    from tests.environments.morpion_durable_v2_helpers import scientific_producer

    args, paths, prepared = setup_closed_baseline(tmp_path, initialize=False)
    original_publish = pub.publish_directory_if_absent
    original_replace = os.replace
    original_manifest = GrowthPatchReceiptSession.publish_manifest
    digest = artifact_digest(tmp_path / "genesis-closed")

    def fail() -> None:
        """Fail."""
        message = "injected genesis interruption"
        raise OSError(message)

    def publish(staging: Path, final: Path, root: Path) -> Any:
        """Publish."""
        if boundary == "before_capsule":
            fail()
        original_publish(staging, final, root)
        if boundary == "after_capsule":
            fail()

    def replace_state(src: Any, dst: Any) -> Any:
        """Replace state."""
        state = Path(dst) == paths.run_state_path
        if state and boundary == "before_state":
            fail()
        original_replace(src, dst)
        if state and boundary == "after_state":
            fail()

    def manifest(session: Any, state: Any) -> Any:
        """Manifest."""
        if boundary == "discovery":
            fail()
        return original_manifest(session, state)

    with monkeypatch.context() as m:
        m.setattr(pub, "publish_directory_if_absent", publish)
        m.setattr(os, "replace", replace_state)
        m.setattr(GrowthPatchReceiptSession, "publish_manifest", manifest)
        with pytest.raises(OSError, match="injected genesis"):
            init.initialize_scientific_v2_publication(
                paths,
                prepared,
                producer=scientific_producer(args),
                closed_scientific_snapshot=True,
            )
    if paths.run_state_path.exists():
        state = run_closed_publication(args, ClosedPublicationRunner(), max_cycles=0)
    else:
        state = init.initialize_scientific_v2_publication(
            paths,
            prepared,
            producer=scientific_producer(args),
            closed_scientific_snapshot=True,
        )
    assert state.generation == 1 and len(publication_records(state.metadata)) == 1
    assert artifact_digest(tmp_path / "genesis-closed") == digest
