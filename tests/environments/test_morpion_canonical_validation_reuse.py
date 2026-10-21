"""Committed scientific evidence is distinct from disposable reader caches."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import pytest
from atomheart.games.morpion.checkpoints import MorpionStateCheckpointCodec

from chipiron.environments.morpion.bootstrap import load_bootstrap_run_state
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    GenerationPublicationError,
    file_digest,
    publication_records,
    validate_capsule,
)
from chipiron.environments.morpion.bootstrap.run_state import save_bootstrap_run_state
from chipiron.environments.morpion.bootstrap.sharded_training_reader import (
    ShardedTrainingGenerationView,
)
from chipiron.environments.morpion.bootstrap.sharded_training_v2 import (
    load_scientific_training_generation,
)
from tests.environments.morpion_durable_v2_helpers import (
    ScientificPublicationRunner,
    publish_scientific_patch,
    run_scientific_publication,
    setup_scientific_baseline,
)


def count_decodes(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Count actual canonical reconstructions without altering their behavior."""
    original = MorpionStateCheckpointCodec.load_state_ref
    calls = []

    def decode(self: Any, ref: object) -> Any:
        calls.append(ref)
        return original(self, ref)

    monkeypatch.setattr(MorpionStateCheckpointCodec, "load_state_ref", decode)
    return calls


@pytest.mark.parametrize("new_nodes", [0, 1])
def test_reuses_only_committed_immutable_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, new_nodes: int
) -> None:
    """Genesis validates all records; subsequent publication validates only new ones."""
    calls = count_decodes(monkeypatch)
    args, paths, _initial = setup_scientific_baseline(tmp_path)
    assert len(calls) == 2
    publish_scientific_patch(paths)
    calls.clear()
    runner = ScientificPublicationRunner()
    runner.grow_delta = new_nodes
    state = run_scientific_publication(args, runner)
    assert len(calls) == new_nodes
    records = publication_records(state.metadata)
    seal = validate_capsule(tmp_path, records[-1])
    proof = seal["canonical_validation"]
    assert proof["scientific_identity"] == seal["scientific_identity"]
    assert proof["reused_from"]["capsule_id"] == records[-2]["id"]
    assert proof["reused_from"]["descriptor_sha256"] == records[-2]["descriptor_sha256"]
    assert proof["record_shards"][-1]["count"] == new_nodes
    # Public readers never inherit the publisher's reuse scope or its temporary cache.
    source = tmp_path / state.latest_tree_snapshot_path
    calls.clear()
    with ShardedTrainingGenerationView(
        source, allow_scientific_v2=True, cache_dir=tmp_path / "cache"
    ) as view:
        assert len(calls) == 2 + new_nodes
        indexed = tuple(view.iter_nodes_in_export_order())
    assert indexed == load_scientific_training_generation(source).snapshot.nodes


@pytest.mark.parametrize(
    "change",
    ["missing", "implementation", "python", "codec", "optimization", "backend"],
)
def test_missing_or_different_validator_performs_full_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """An older committed capsule remains readable but cannot supply new evidence."""
    args, paths, state = setup_scientific_baseline(tmp_path)
    record = publication_records(state.metadata)[-1]
    path = tmp_path / record["descriptor_path"]
    payload = json.loads(path.read_bytes())
    if change == "missing":
        del payload["canonical_validation"]
    elif change == "optimization":
        payload["canonical_validation"]["validator"]["python"][3] += 1
    else:
        key = {
            "implementation": "implementation_sha256",
            "python": "python",
            "codec": "state_ref_contract",
            "backend": "ijson_backend",
        }[change]
        payload["canonical_validation"]["validator"][key] = "previous-validator"
    # Simulate a separately committed predecessor format/validator in this new fixture.
    path.write_text(json.dumps(payload))
    record["descriptor_sha256"] = file_digest(path)
    save_bootstrap_run_state(state, paths.run_state_path, durable=True)
    # This is a different committed test head; reconstruct its discovery on recovery.
    paths.pipeline_manifest_path_for_generation(state.generation).unlink()
    calls = count_decodes(monkeypatch)
    publish_scientific_patch(paths)
    result = run_scientific_publication(args, ScientificPublicationRunner())
    assert len(calls) == 2
    seal = validate_capsule(tmp_path, publication_records(result.metadata)[-1])
    assert seal["canonical_validation"]["reused_from"] is None


@pytest.mark.parametrize("change", ["version", "source", "producer", "shard", "extra"])
def test_malformed_committed_evidence_fails_closed(tmp_path: Path, change: str) -> None:
    """Even coherently sealed but incompatible evidence cannot authorize reuse."""
    args, paths, state = setup_scientific_baseline(tmp_path)
    record = publication_records(state.metadata)[-1]
    path = tmp_path / record["descriptor_path"]
    payload = json.loads(path.read_bytes())
    proof = payload["canonical_validation"]
    if change == "version":
        proof["version"] = True
    elif change == "source":
        proof["scientific_identity"]["source_generation"] = 2
    elif change == "producer":
        proof["scientific_identity"]["producer"]["revision"] = "unrelated"
    elif change == "shard":
        proof["record_shards"][0]["sha256"] = "0" * 64
    else:
        proof["unrecognized"] = True
    path.write_text(json.dumps(payload))
    record["descriptor_sha256"] = file_digest(path)
    save_bootstrap_run_state(state, paths.run_state_path, durable=True)
    # This is a different committed test head; reconstruct its discovery on recovery.
    paths.pipeline_manifest_path_for_generation(state.generation).unlink()
    publish_scientific_patch(paths)
    with pytest.raises(GenerationPublicationError, match=r"canonical|Canonical"):
        run_scientific_publication(args, ScientificPublicationRunner())
    assert load_bootstrap_run_state(paths.run_state_path).generation == 1
    assert paths.pipeline_reevaluation_patch_path.exists()


def test_corrupt_historical_shard_cannot_use_evidence(tmp_path: Path) -> None:
    """Committed evidence never replaces a fresh content hash of historical bytes."""
    args, paths, state = setup_scientific_baseline(tmp_path)
    publish_scientific_patch(paths)
    source = tmp_path / state.latest_tree_snapshot_path
    # Populate a valid consumer cache before corrupting a newly generated fixture.
    with ShardedTrainingGenerationView(
        source, allow_scientific_v2=True, cache_dir=tmp_path / "cache"
    ):
        pass
    shard = source.parent / "records.jsonl"
    shard.write_bytes(shard.read_bytes().replace(b'"root"', b'"fake"', 1))
    with pytest.raises(GenerationPublicationError, match="dependency changed"):
        run_scientific_publication(args, ScientificPublicationRunner())
    with pytest.raises(GenerationPublicationError, match="dependency changed"):
        ShardedTrainingGenerationView(
            source, allow_scientific_v2=True, cache_dir=tmp_path / "cache"
        )


@pytest.mark.parametrize("field", ["identity", "payload"])
def test_invalid_cache_never_becomes_canonical_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    """Corrupt cache identity/payload forces full validation outside the private scope."""
    _args, _paths, state = setup_scientific_baseline(tmp_path)
    source = tmp_path / state.latest_tree_snapshot_path
    cache = tmp_path / "cache"
    with ShardedTrainingGenerationView(
        source, allow_scientific_v2=True, cache_dir=cache
    ) as view:
        index = view.index_path
    with sqlite3.connect(index) as conn:
        if field == "identity":
            conn.execute("UPDATE completed SET identity='unrelated'")
        else:
            conn.execute("UPDATE records SET payload='{}'")
    calls = count_decodes(monkeypatch)
    with ShardedTrainingGenerationView(
        source, allow_scientific_v2=True, cache_dir=cache
    ) as view:
        assert not view.index_reused
        assert len(calls) == 2


@pytest.mark.parametrize(
    "change", ["new_reference", "certainty", "topology", "creation"]
)
def test_incremental_validation_still_rejects_current_corruption(
    tmp_path: Path, change: str
) -> None:
    """New canonical refs and all mutable scientific fields keep strict validation."""
    args, paths, _state = setup_scientific_baseline(tmp_path)
    publish_scientific_patch(paths)

    class CorruptRunner(ScientificPublicationRunner):
        def export_scientific_training_snapshot(
            self, *args: Any, **kwargs: Any
        ) -> Path:
            path = super().export_scientific_training_snapshot(*args, **kwargs)
            manifest = json.loads(path.read_bytes())
            if change == "creation":
                manifest["creation_index"]["new"] = 1
            else:
                shard = path.parent / (
                    "records.jsonl" if change == "new_reference" else "updates.jsonl"
                )
                rows = [json.loads(line) for line in shard.read_text().splitlines()]
                row = rows[-1]
                if change == "new_reference":
                    row["state_ref_payload"] = None
                elif change == "certainty":
                    row["value_contract"]["values"]["direct"]["certainty"] = "invented"
                else:
                    row["parent_ids"] = []
                shard.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
                desc = (
                    manifest["record_shards"][-1]
                    if change == "new_reference"
                    else manifest["update_shard"]
                )
                desc["sha256"] = file_digest(shard)
            path.write_text(json.dumps(manifest))
            return path

    runner = CorruptRunner()
    runner.grow_delta = 1
    with pytest.raises((ValueError, TypeError)):
        run_scientific_publication(args, runner)
    assert load_bootstrap_run_state(paths.run_state_path).generation == 1
    # A fresh retry starts at the committed parent and succeeds with the original receipt.
    retry = ScientificPublicationRunner()
    retry.grow_delta = 1
    result = run_scientific_publication(args, retry)
    assert result.generation == 2 and retry.apply_calls == 1


def test_v2_reader_remains_pinned_during_incremental_publication(
    tmp_path: Path,
) -> None:
    """A real competing process publishes with reuse while the old reader stays valid."""
    from tests.environments.test_morpion_generation_publication_concurrency import (
        _line,
    )

    args, paths, initial = setup_scientific_baseline(tmp_path)
    publish_scientific_patch(paths)
    script = """
import sys
from pathlib import Path
import chipiron.environments.morpion.bootstrap.generation_publication as publication
from tests.environments.morpion_durable_v2_helpers import (
    ScientificPublicationRunner, scientific_args, run_scientific_publication,
)
root=Path(sys.argv[1])
original=publication.publish_directory_if_absent
def publish(staging,final,work_dir):
    if final.parent.name == 'objects':
        print('owned',flush=True)
        assert sys.stdin.readline().strip() == 'continue'
    original(staging,final,work_dir)
publication.publish_directory_if_absent=publish
runner=ScientificPublicationRunner()
state=run_scientific_publication(scientific_args(root),runner)
print(f'committed:{state.generation}:{runner.apply_calls}',flush=True)
"""
    with (
        ShardedTrainingGenerationView(
            tmp_path / initial.latest_tree_snapshot_path,
            allow_scientific_v2=True,
            cache_dir=tmp_path / "reader-cache",
        ) as view,
        subprocess.Popen(
            [sys.executable, "-c", script, str(tmp_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ) as process,
    ):
        try:
            expected = tuple(view.iter_nodes_in_export_order())
            assert _line(process) == "owned"
            assert load_bootstrap_run_state(paths.run_state_path).generation == 1
            with pytest.raises(BlockingIOError):
                run_scientific_publication(args, ScientificPublicationRunner())
            assert tuple(view.iter_nodes_in_export_order()) == expected
            view.validate_source_identity()
            assert process.stdin is not None
            process.stdin.write("continue\n")
            process.stdin.flush()
            assert _line(process) == "committed:2:1"
            assert process.wait(timeout=20) == 0
            assert tuple(view.iter_nodes_in_export_order()) == expected
            view.validate_source_identity()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=20)
    current = load_bootstrap_run_state(paths.run_state_path)
    seal = validate_capsule(tmp_path, publication_records(current.metadata)[-1])
    assert seal["canonical_validation"]["reused_from"] is not None


def test_validator_identity_includes_selected_parser_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same package bytes do not authorize a different selected parser backend."""
    import ijson

    from chipiron.environments.morpion.bootstrap.scientific_validation_evidence import (
        validator_identity,
    )

    current = validator_identity()
    monkeypatch.setattr(ijson, "backend", "different-parser-selection")
    changed = validator_identity()
    assert current["implementation_sha256"] == changed["implementation_sha256"]
    assert current != changed
    assert current["python"][3] == sys.flags.optimize
