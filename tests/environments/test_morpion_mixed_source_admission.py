"""C5's C3 admission boundary; synthetic sealed evidence is never a Growth writer."""

import copy
import json
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from chipiron.environments.morpion.bootstrap.bootstrap_args import MorpionBootstrapArgs
from chipiron.environments.morpion.bootstrap.cycle_dataset import (
    extract_rows_from_training_snapshot,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    PUBLICATION_KEY,
    file_digest,
    object_digest,
    private_generation_validation,
    validate_capsule,
)
from chipiron.environments.morpion.bootstrap.prediction_lineage import PredictionLineage
from chipiron.environments.morpion.learning import (
    load_morpion_supervised_rows,
    save_morpion_supervised_rows_streaming,
)
from chipiron.environments.morpion.learning import mixed_source as admission
from tests.environments.test_morpion_mixed_dataset_training import (
    mixed_source,
    scientific_fixture,
    sha,
)

if TYPE_CHECKING:
    from pathlib import Path


def committed_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    historical_claim: bool = False,
    homogeneous: bool = False,
) -> tuple[Path, dict[str, Any], list[str]]:
    """Authenticate exact fixture evidence at the specified C3 API, not in C5.

    Only the C3 reader is replaced. Run-state selection, manifest/checkpoint/ref
    bindings and interpretation remain real C5 code. This is interface evidence,
    explicitly not evidence of actual Growth ownership or write instrumentation.
    """
    original = scientific_fixture(tmp_path)
    mixed = mixed_source(original)
    root = tmp_path / "workspace"
    identifier = "c" * 32
    capsule = root / "generation_publications_v3/objects" / identifier
    from chipiron.environments.morpion.bootstrap.closed_scientific_snapshot import (
        save_closed_scientific_snapshot,
    )
    from chipiron.environments.morpion.bootstrap.scientific_generation_publication import (
        bind_mixed_closed_source,
        validate_closed_export,
    )
    from tests.environments.test_morpion_sharded_training_v2 import (
        PRODUCER,
        advance,
        nodes_fixture,
    )

    stage = root / "generation_publications_v3/staging" / identifier
    stage.mkdir(parents=True)
    (stage / "checkpoint").write_text("checkpoint")
    (stage / "bootstrap_config.json").write_text("{}\n")
    producer = {
        **PRODUCER,
        "config_sha256": sha("{}"),
        "evaluator": sha("A" if homogeneous else "B"),
    }
    nodes = nodes_fixture()
    advance(nodes)
    validated: dict[str, Any] = {}
    with private_generation_validation(stage):
        path, _ = save_closed_scientific_snapshot(
            nodes=nodes,
            root_node_id="0",
            destination=stage / "scientific_snapshot",
            generation=1,
            producer=producer,
            _validation_result=validated,
        )
    header_producer = json.loads(path.read_text())["producer"]
    origins = tuple(
        replace(
            origin,
            producer_sha256=object_digest({
                **header_producer,
                "identity": {
                    **header_producer["identity"],
                    "evaluator": origin.evaluator_sha256,
                },
            }),
            config_sha256=producer["config_sha256"],
        )
        for origin in mixed.origins
    )
    mixed = replace(
        mixed,
        origins=origins[:1] if homogeneous else origins,
        current={
            node: origins[0 if homogeneous else mixed.origins.index(origin)]
            for node, origin in mixed.current.items()
        },
    )
    lineage = PredictionLineage(
        sha("checkpoint"),
        mixed.origins,
        None,
        (),
        tuple(
            (node, mixed.origins.index(origin))
            for node, origin in mixed.current.items()
        ),
    )
    (stage / "prediction_lineage.json").write_bytes(lineage.to_bytes())
    # Separate fixture write evidence; the origin map is not its own qualification.
    evidence = {node: origin.evidence_sha256 for node, origin in mixed.current.items()}
    (stage / "prediction_evidence.json").write_text(json.dumps(evidence))
    reference = {
        **mixed.provenance["prediction_lineage"],
        "sha256": file_digest(stage / "prediction_lineage.json"),
        "evidence_sha256": file_digest(stage / "prediction_evidence.json"),
    }
    if not historical_claim:
        bind_mixed_closed_source(
            path, staging=stage, prediction_lineage=reference, validated=validated
        )
    header = json.loads(path.read_text())
    identity, canonical = validate_closed_export(
        path,
        staging=stage,
        generation=1,
        producer=producer,
        checkpoint=stage / "checkpoint",
        config_hash=producer["config_sha256"],
        work_dir=root,
        parent=None,
        validated=validated,
    )
    descriptor = {
        "protocol": PUBLICATION_KEY,
        "id": identifier,
        "generation": 1,
        "publication_version": 3,
        "parent_id": None,
        "scientific_identity": identity,
        "prediction_lineage": reference,
        "sharded_snapshot_path": "scientific_snapshot/generation.json",
        "bootstrap_config_hash": producer["config_sha256"],
        "model_pins": [],
        "canonical_validation": canonical,
        "files": [
            {
                "path": str(item.relative_to(stage)),
                "size": item.stat().st_size,
                "sha256": file_digest(item),
            }
            for item in sorted(stage.rglob("*"))
            if item.is_file()
        ],
    }
    seal_name = ".generation_publication_v3.json"
    (stage / seal_name).write_text(json.dumps(descriptor))
    capsule.parent.mkdir(parents=True)
    stage.rename(capsule)
    path = capsule / "scientific_snapshot/generation.json"
    record = {
        "id": identifier,
        "generation": 1,
        "publication_version": 3,
        "parent_id": None,
        "tree_snapshot_path": str(path.relative_to(root)),
        "checkpoint_path": str((capsule / "checkpoint").relative_to(root)),
        "descriptor_path": str((capsule / seal_name).relative_to(root)),
        "descriptor_sha256": file_digest(capsule / seal_name),
        "scientific_identity": identity,
        "prediction_lineage": reference,
    }
    (root / "run_state.json").write_text(
        json.dumps({
            "generation": 1,
            "latest_runtime_checkpoint_path": record["checkpoint_path"],
            "latest_tree_snapshot_path": record["tree_snapshot_path"],
            "metadata": {
                PUBLICATION_KEY: {
                    "schema_version": 3,
                    "head_id": identifier,
                    "generations": [record],
                }
            },
        })
    )
    assert identity["producer_sha256"] == object_digest(header["producer"])
    validate_capsule(root, record)
    calls: list[str] = []

    def authenticate(work_dir: Path, selected: dict[str, Any]) -> SimpleNamespace:
        assert work_dir == root and selected == record
        if (
            file_digest(capsule / "prediction_evidence.json")
            != reference["evidence_sha256"]
        ):
            raise ValueError("fixture write evidence changed")  # noqa: TRY003
        if file_digest(capsule / "prediction_lineage.json") != reference["sha256"]:
            raise ValueError("fixture lineage changed")  # noqa: TRY003
        assert all(
            evidence[node] == origin.evidence_sha256
            for node, origin in mixed.current.items()
        )
        calls.append(selected["id"])
        return SimpleNamespace(
            lineage=lineage,
            creation_generations=mixed.creation_generations,
            current_origins=mixed.current,
        )

    module = SimpleNamespace(read_committed_prediction_origins=authenticate)
    real_import = admission.importlib.import_module

    def import_c3(name: str, package: str | None = None) -> Any:
        if (
            name
            == "chipiron.environments.morpion.bootstrap.prediction_origin_publication"
        ):
            return module
        return real_import(name, package)

    monkeypatch.setattr(admission.importlib, "import_module", import_c3)
    return path, header, calls


def test_committed_reader_authenticates_writes_before_mixed_interpretation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A/B identity comes from C3; an active evaluator is never used as a default."""
    path, header, calls = committed_fixture(tmp_path, monkeypatch)
    result = admission.read_mixed_dataset_source(path, header)
    assert calls == ["c" * 32]
    assert {origin.evaluator_sha256 for origin in result.current.values()} == {
        sha("A"),
        sha("B"),
    }
    assert sum(result.provenance["direct_origin_counts"]) == len(result.current)
    assert result.provenance["prediction_lineage"] == header["prediction_lineage"]
    # Same source, but a requested/active evaluator claim cannot relabel its map.
    bad = copy.deepcopy(header)
    bad["producer"]["identity"]["evaluator"] = sha("new active")
    with pytest.raises(ValueError, match="contradicts committed"):
        admission.read_mixed_dataset_source(path, bad)


@pytest.mark.parametrize(
    "file", ["prediction_lineage.json", "prediction_evidence.json"]
)
def test_missing_or_contradictory_write_evidence_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, file: str
) -> None:
    """A previously matching origin declaration does not survive changed evidence."""
    path, header, _calls = committed_fixture(tmp_path, monkeypatch)
    (path.parent.parent / file).write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        admission.read_mixed_dataset_source(path, header)


def test_c3_absence_and_uncommitted_origins_remain_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No normal execution can substitute unit fixtures for an absent C3 reader."""
    path, header, _calls = committed_fixture(tmp_path, monkeypatch)

    def missing(_name: str) -> Any:
        raise ModuleNotFoundError("C3 not installed")  # noqa: TRY003

    monkeypatch.setattr(admission.importlib, "import_module", missing)
    with pytest.raises(ValueError, match="requires C3 actual Growth"):
        admission.read_mixed_dataset_source(path, header)
    with pytest.raises(ValueError, match="committed closed capsule"):
        admission.read_mixed_dataset_source(tmp_path / "generation.json", header)


@pytest.mark.parametrize(
    "corruption", ["missing", "tree_as_neural", "certified_as_neural", "bad_index"]
)
def test_serialized_rows_refuse_contradictory_origin_semantics(
    tmp_path: Path, corruption: str
) -> None:
    """Recovery revalidates per-slot provenance without recomputing target math."""
    source = scientific_fixture(tmp_path)
    mixed = mixed_source(source)
    args = MorpionBootstrapArgs.new_scientific(
        tmp_path,
        target_source="effective" if corruption == "certified_as_neural" else "tree",
    )
    rows = extract_rows_from_training_snapshot(
        args=args,
        snapshot=source.snapshot,
        generation=1,
        value_contracts=source.value_contracts,
        source_provenance=mixed.provenance,
        mixed_source=mixed,
    )
    path = tmp_path / "mixed.jsonl"
    save_morpion_supervised_rows_streaming(
        rows=rows.rows, metadata=rows.metadata, path=path
    )
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    # JSONL records retain their existing row envelope; use the real serializer.
    selected = next(
        record
        for record in lines
        if record.get("kind") == "morpion_supervised_row"
        and (
            record["row"]["metadata"]["target_confidence"]["raw_target_origin"]["kind"]
            == (
                "certified_value"
                if corruption == "certified_as_neural"
                else "derived_search_value"
            )
        )
    )
    confidence = selected["row"]["metadata"]["target_confidence"]
    if corruption == "missing":
        del confidence["direct_prediction_origin"]
    elif corruption == "bad_index":
        confidence["direct_prediction_origin"] = 99
    else:
        confidence["raw_target_origin"] = {
            "kind": "direct_prediction",
            "origin_index": 0,
        }
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    with pytest.raises(ValueError, match="origin"):
        load_morpion_supervised_rows(path)


@pytest.mark.parametrize("target", ["effective", "tree", "direct"])
def test_authenticated_mixed_eager_indexed_creation_and_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: Any,
) -> None:
    """Real source pins/index/extraction/recovery use the C3 boundary exactly."""
    from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
        MorpionBootstrapPaths,
    )
    from chipiron.environments.morpion.bootstrap.closed_scientific_snapshot import (
        load_closed_scientific_snapshot,
    )
    from chipiron.environments.morpion.bootstrap.dataset_publication import (
        _validate_mixed_rows,
    )
    from chipiron.environments.morpion.bootstrap.sharded_training_reader import (
        ShardedTrainingGenerationView,
    )
    from chipiron.environments.morpion.bootstrap.streaming_dataset import (
        IndexedDatasetSource,
        disk_dataset_context,
    )

    path, _header, calls = committed_fixture(tmp_path, monkeypatch)
    args = MorpionBootstrapArgs.new_scientific(tmp_path, target_source=target)
    source = load_closed_scientific_snapshot(path)
    expected = extract_rows_from_training_snapshot(
        args=args,
        snapshot=source.snapshot,
        generation=1,
        value_contracts=source.value_contracts,
        source_provenance=source.provenance,
        mixed_source=source.mixed_source,
    )
    rows_path = tmp_path / "rows.jsonl"
    with (
        ShardedTrainingGenerationView(
            path, cache_dir=tmp_path / "index", allow_closed_snapshot=True
        ) as view,
        disk_dataset_context(IndexedDatasetSource(view), args) as context,
    ):
        actual = context.stream(1)
        assert tuple(actual.rows) == expected.rows
        assert actual.metadata == expected.metadata
        save_morpion_supervised_rows_streaming(
            rows=context.rows(), metadata=actual.metadata, path=rows_path
        )
    assert calls  # Neither reader silently admitted a supplied origin table.
    assert load_morpion_supervised_rows(rows_path) == expected
    root = tmp_path / "workspace"
    record = json.loads((root / "run_state.json").read_text())["metadata"][
        PUBLICATION_KEY
    ]["generations"][0]
    admitted = {
        "generation": 1,
        "source": {
            key: record[key]
            for key in ("id", "descriptor_sha256", "scientific_identity")
        },
    }
    _validate_mixed_rows(MorpionBootstrapPaths.from_work_dir(root), rows_path, admitted)
    # Change only one node's origin to another otherwise valid interned model.
    # Serialization remains structurally valid; source admission must reject it.
    damaged = list(expected.rows)
    index = next(
        i
        for i, row in enumerate(damaged)
        if row.metadata["target_confidence"]["direct_prediction_origin"] is not None
    )
    row = damaged[index]
    metadata = copy.deepcopy(row.metadata)
    confidence = metadata["target_confidence"]
    confidence["direct_prediction_origin"] = 1 - confidence["direct_prediction_origin"]
    if confidence["raw_target_origin"]["kind"] == "direct_prediction":
        confidence["raw_target_origin"]["origin_index"] = confidence[
            "direct_prediction_origin"
        ]
    damaged[index] = replace(row, metadata=metadata)
    save_morpion_supervised_rows_streaming(
        rows=damaged, metadata=expected.metadata, path=rows_path
    )
    with pytest.raises(ValueError, match="authenticated source values/origins"):
        _validate_mixed_rows(
            MorpionBootstrapPaths.from_work_dir(root), rows_path, admitted
        )


@pytest.mark.parametrize("homogeneous", [False, True])
def test_tracked_historical_header_cannot_hide_mixed_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    homogeneous: bool,
) -> None:
    """Historical untracked support stays intact; tracked claims must be truthful."""
    from chipiron.environments.morpion.bootstrap.closed_scientific_snapshot import (
        load_closed_scientific_snapshot,
    )
    from chipiron.environments.morpion.bootstrap.sharded_training_reader import (
        ShardedTrainingGenerationView,
    )

    path, _header, _calls = committed_fixture(
        tmp_path, monkeypatch, historical_claim=True, homogeneous=homogeneous
    )
    if homogeneous:
        source = load_closed_scientific_snapshot(path)
        assert source.mixed_source is None
        with ShardedTrainingGenerationView(
            path, cache_dir=tmp_path / "index", allow_closed_snapshot=True
        ) as view:
            assert view.mixed_source is None
    else:
        with pytest.raises(ValueError, match="Homogeneous claim"):
            load_closed_scientific_snapshot(path)
        with pytest.raises(ValueError, match="Homogeneous claim"):
            ShardedTrainingGenerationView(
                path, cache_dir=tmp_path / "index", allow_closed_snapshot=True
            )
