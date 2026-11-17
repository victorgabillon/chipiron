"""Production C3 authentication for real mixed eager/indexed Dataset sources."""

import copy
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest

from chipiron.environments.morpion.bootstrap.bootstrap_args import MorpionBootstrapArgs
from chipiron.environments.morpion.bootstrap.closed_scientific_snapshot import (
    load_closed_scientific_snapshot,
)
from chipiron.environments.morpion.bootstrap.cycle_dataset import (
    extract_rows_from_training_snapshot,
)
from chipiron.environments.morpion.bootstrap.dataset_publication import (
    _validate_mixed_rows,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    publication_records,
)
from chipiron.environments.morpion.bootstrap.model_activation import (
    activate_model,
    prepare_model_activation_request,
)
from chipiron.environments.morpion.bootstrap.sharded_training_reader import (
    ShardedTrainingGenerationView,
)
from chipiron.environments.morpion.bootstrap.streaming_dataset import (
    IndexedDatasetSource,
    disk_dataset_context,
)
from chipiron.environments.morpion.learning import (
    load_morpion_supervised_rows,
    save_morpion_supervised_rows_streaming,
)
from tests.environments.morpion_fenced_dataset_helpers import run_dataset
from tests.environments.test_morpion_prediction_origin_writes import grow
from tests.environments.test_morpion_progressive_activation import setup

if TYPE_CHECKING:
    from pathlib import Path

    from chipiron.environments.morpion.bootstrap.pv_family_targets import (
        PvFamilyTargetPolicy,
    )
    from chipiron.environments.morpion.learning.target_policy import TargetSource


@pytest.fixture(scope="module")
def mixed(tmp_path_factory: Any) -> Any:
    """Publish real A/B writes and accept Dataset using the configured target policy."""
    root = tmp_path_factory.mktemp("real-mixed-source")
    with pytest.MonkeyPatch.context() as mp:
        args, paths, _ = setup(root, mp)
        request = prepare_model_activation_request(paths, training_generation=1)
        activate_model(paths, request)
        state = grow(args, paths)
        result = run_dataset(args, generation=state.generation)
        assert result.dataset_status == "done"
    return paths, publication_records(state.metadata)[-1]


@pytest.mark.parametrize("target", ["effective", "tree", "direct"])
@pytest.mark.parametrize("family", ["none", "pv_blend_mean_prediction"])
def test_real_mixed_eager_indexed_targets_and_recovery(
    mixed: Any, tmp_path: Path, target: TargetSource, family: PvFamilyTargetPolicy
) -> None:
    """Both readers authenticate every source node, then apply unchanged target math."""
    paths, record = mixed
    path = paths.work_dir / record["tree_snapshot_path"]
    args = replace(
        MorpionBootstrapArgs.new_scientific(tmp_path, target_source=target),
        dataset_family_target_policy=family,
    )
    source = load_closed_scientific_snapshot(path)
    assert source.mixed_source is not None
    assert len({o.evaluator_sha256 for o in source.mixed_source.current.values()}) == 2
    expected = extract_rows_from_training_snapshot(
        args=args,
        snapshot=source.snapshot,
        generation=record["generation"],
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
        actual = context.stream(record["generation"])
        assert tuple(actual.rows) == expected.rows
        assert actual.metadata == expected.metadata
        save_morpion_supervised_rows_streaming(
            rows=context.rows(), metadata=actual.metadata, path=rows_path
        )
    assert load_morpion_supervised_rows(rows_path) == expected
    admitted = {
        "generation": record["generation"],
        "source": {
            key: record[key]
            for key in ("id", "descriptor_sha256", "scientific_identity")
        },
    }
    _validate_mixed_rows(paths, rows_path, admitted)
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
        _validate_mixed_rows(paths, rows_path, admitted)


def test_model_only_stale_inference_and_old_packet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reuse the real C4 race scenario with the new fresh activation factory."""
    from tests.environments import test_morpion_rolling_reevaluation as c4

    monkeypatch.setattr(c4, "setup", setup)
    c4.test_obsolete_inference_discards_only_batch_and_stale_packet_refuses(
        tmp_path, monkeypatch
    )


def test_model_only_source_retention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retain incomplete source until accepted completion, including report pins."""
    from tests.environments import test_morpion_rolling_reevaluation as c4

    monkeypatch.setattr(c4, "setup", setup)
    import subprocess
    import sys

    with monkeypatch.context() as bounded:
        args, paths, _ = setup(tmp_path / "bounded", bounded)
        c4.propose(args, 1)
        grow(args, paths)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from pathlib import Path; from chipiron.environments.morpion.bootstrap.bootstrap_paths import MorpionBootstrapPaths; from tests.environments.test_morpion_rolling_recovery_report import analyze; import sys; r=analyze(MorpionBootstrapPaths.from_work_dir(Path(sys.argv[1]))); assert r['validated'],r",
                str(paths.work_dir),
            ],
            check=False,
            timeout=30,
        )
        assert result.returncode == 0
    c4.test_retirement_cannot_drop_incomplete_pass(tmp_path / "complete", monkeypatch)


@pytest.mark.parametrize("boundary", ["before", "after"])
def test_model_only_receipt_crash_and_exact_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    """Run C4's actual process-death/duplicate receipt contract on mixed sources."""
    from tests.environments import test_morpion_rolling_reevaluation as c4

    monkeypatch.setattr(c4, "setup", setup)
    c4.test_crash_recovers_receipt_cursor_once(tmp_path, monkeypatch, boundary)


def test_false_homogeneous_header_refuses_real_mixed_writes(
    mixed: Any, tmp_path: Path
) -> None:
    """Rehashing a homogeneous header cannot erase authenticated A/B write origins."""
    import json
    import shutil

    from chipiron.environments.morpion.bootstrap.generation_publication_io import (
        PUBLICATION_KEY,
        file_digest,
    )
    from chipiron.environments.morpion.bootstrap.sharded_training_v2_contract import (
        CONTRACT,
    )

    paths, source_record = mixed
    root = tmp_path / "copy"
    shutil.copytree(paths.work_dir, root)
    state_path = root / "run_state.json"
    state = json.loads(state_path.read_text())
    record = next(
        r
        for r in state["metadata"][PUBLICATION_KEY]["generations"]
        if r["id"] == source_record["id"]
    )
    header_path = root / record["tree_snapshot_path"]
    header = json.loads(header_path.read_text())
    header["scientific_contract"] = CONTRACT
    for key in ("source_policy", "prediction_lineage"):
        header.pop(key)
    header_path.write_text(json.dumps(header))
    identity = record["scientific_identity"]
    identity["scientific_contract"] = CONTRACT
    identity["generation_descriptor_sha256"] = file_digest(header_path)
    for key in ("source_policy", "prediction_lineage"):
        identity.pop(key)
    descriptor_path = root / record["descriptor_path"]
    descriptor = json.loads(descriptor_path.read_text())
    descriptor["scientific_identity"] = identity
    descriptor["canonical_validation"]["scientific_identity"] = identity
    entry = next(
        e
        for e in descriptor["files"]
        if e["path"] == "scientific_snapshot/generation.json"
    )
    entry.update(size=header_path.stat().st_size, sha256=file_digest(header_path))
    descriptor_path.write_text(json.dumps(descriptor))
    record["descriptor_sha256"] = file_digest(descriptor_path)
    state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="Homogeneous claim"):
        load_closed_scientific_snapshot(header_path)
    with pytest.raises(ValueError, match="Homogeneous claim"):
        ShardedTrainingGenerationView(
            header_path, cache_dir=tmp_path / "bad-index", allow_closed_snapshot=True
        )
