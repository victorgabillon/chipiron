"""Selected-value scientific confidence, with frozen Dataset v1 compatibility."""

from __future__ import annotations

import copy
import json
import shutil
from dataclasses import replace
from typing import TYPE_CHECKING, Any, get_args

import pytest
from anemone.node_evaluation.common import canonical_value

from chipiron.environments.morpion.bootstrap import MorpionBootstrapArgs
from chipiron.environments.morpion.bootstrap.cycle_dataset import (
    extract_rows_from_training_snapshot,
)
from chipiron.environments.morpion.bootstrap.pv_family_targets import (
    PvFamilyTargetPolicy,
)
from chipiron.environments.morpion.bootstrap.sharded_training_reader import (
    ShardedTrainingGenerationView,
)
from chipiron.environments.morpion.bootstrap.sharded_training_v2 import (
    load_scientific_training_generation,
)
from chipiron.environments.morpion.bootstrap.streaming_dataset import (
    IndexedDatasetSource,
    disk_dataset_context,
)
from chipiron.environments.morpion.learning import (
    MorpionSupervisedRows,
    load_morpion_supervised_rows,
    morpion_supervised_rows_from_dict,
    morpion_supervised_rows_to_dict,
    save_morpion_supervised_rows,
    save_morpion_supervised_rows_streaming,
)
from chipiron.environments.morpion.learning.target_confidence import (
    SAFE_FAMILY_POLICIES,
    DatasetConfidenceError,
)
from docs.research.dataset_confidence_v2.audit import (
    export_case,
    family_counterexample,
    generated_cases,
)
from tests.environments.test_morpion_value_certainty import (
    endgame,
    make_runner,
    model_patch,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def sources(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Generate the actual PR #82 proof and model prediction once in scratch."""
    cases, witness = generated_cases(tmp_path_factory.mktemp("confidence-sources"))
    return {
        **{f"{case.name}-{case.generation}": case for case in cases},
        "witness": witness,
    }


def extract(args: MorpionBootstrapArgs, oracle: Any) -> MorpionSupervisedRows:
    """Supply explicit independent confidence; a flat snapshot alone is insufficient."""
    return extract_rows_from_training_snapshot(
        args=args,
        snapshot=oracle.snapshot,
        generation=oracle.provenance["source_generation"],
        value_contracts=oracle.value_contracts,
        source_provenance=oracle.provenance,
    )


@pytest.mark.fast
@pytest.mark.parametrize("policy", get_args(PvFamilyTargetPolicy))
@pytest.mark.parametrize("backup", [False, True])
@pytest.mark.parametrize("exact_only", [False, True])
def test_actual_endgame_all_nine_policies(
    tmp_path: Path,
    sources: dict[str, Any],
    policy: Any,
    backup: bool,
    exact_only: bool,
) -> None:
    """Keep FORCED(35), never certify ESTIMATE(42), and gate all exact transfers."""
    case = sources["actual-endgame-2"]
    oracle = load_scientific_training_generation(case.path)
    assert oracle.value_contracts["0"]["certainties"] == {
        "direct": "ESTIMATE",
        "tree": "FORCED",
        "effective": "FORCED",
        "target": "FORCED",
    }
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        dataset_confidence_policy="selected_value_v2",
        dataset_family_target_policy=policy,
        use_backed_up_value=backup,
        require_exact_or_terminal=exact_only,
    )
    with ShardedTrainingGenerationView(
        case.path, cache_dir=tmp_path / "cache", allow_scientific_v2=True
    ) as view:
        if policy not in SAFE_FAMILY_POLICIES:
            with pytest.raises(DatasetConfidenceError, match="gates pv_exact_then"):
                extract(args, oracle)
            with (
                pytest.raises(DatasetConfidenceError, match="gates pv_exact_then"),
                disk_dataset_context(IndexedDatasetSource(view), args),
            ):
                pytest.fail("gated family policy accepted")
            return
        expected = extract(args, oracle)
        with disk_dataset_context(IndexedDatasetSource(view), args) as context:
            stream = context.stream(2)
            actual = MorpionSupervisedRows(tuple(stream.rows), stream.metadata)
        assert actual == expected
    assert len(expected.rows) == (1 if exact_only and not backup else 2)
    terminal = expected.rows[-1]
    assert terminal.is_terminal and terminal.is_exact and terminal.target_value == 35
    assert (
        terminal.metadata["target_confidence"]["training_target_certainty"]
        == "TERMINAL"
    )
    if not exact_only or backup:
        parent = expected.rows[0]
        confidence = parent.metadata["target_confidence"]
        assert confidence["node_is_exact"] is True
        assert confidence["raw_target_value"] == (35 if backup else 42)
        assert confidence["raw_target_certainty"] == (
            "FORCED" if backup else "ESTIMATE"
        )
        assert parent.is_exact is backup
        assert parent.target_value == (
            35
            if backup
            else {
                "none": 42,
                "pv_mean_prediction": 38.5,
                "pv_min_prediction": 35,
                "pv_blend_mean_prediction": 41.125,
                "pv_blend_min_prediction": 40.25,
            }[policy]
        )
        assert confidence["training_target_certainty"] == (
            "FORCED" if backup else "ESTIMATE"
        )
        assert parent.metadata["family_exact_target"] is None
    for suffix in ("json", "jsonl"):
        path = tmp_path / f"rows.{suffix}"
        if suffix == "json":
            save_morpion_supervised_rows(expected, path)
        else:
            save_morpion_supervised_rows_streaming(
                rows=expected.rows, metadata=expected.metadata, path=path
            )
            assert json.loads(path.read_text().splitlines()[0])["format_version"] == 2
        assert load_morpion_supervised_rows(path) == expected


@pytest.mark.fast
def test_family_membership_does_not_prove_equal_state_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain a concrete legal terminal 71 from an ancestor in terminal-35's PV family."""
    nodes, witness = family_counterexample(monkeypatch)
    assert witness["alternative_terminal_value"]["score"] == 71
    assert witness["alternative_terminal_value"]["certainty"] == "TERMINAL"
    case = export_case(tmp_path, "counterexample", 1, nodes)
    oracle = load_scientific_training_generation(case.path)
    for policy in SAFE_FAMILY_POLICIES:
        rows = extract(
            MorpionBootstrapArgs(
                work_dir=tmp_path,
                dataset_confidence_policy="selected_value_v2",
                dataset_family_target_policy=policy,
                use_backed_up_value=False,
            ),
            oracle,
        )
        first, last = rows.rows[0], rows.rows[-1]
        assert first.metadata["family_representative_node_id"] == last.node_id
        assert not first.is_exact
        assert (
            first.metadata["target_confidence"]["training_target_certainty"]
            == "ESTIMATE"
        )
        assert last.target_value == 35 and last.is_exact


@pytest.mark.parametrize("policy", sorted(SAFE_FAMILY_POLICIES))
def test_estimated_backup_does_not_inherit_direct_proof(
    tmp_path: Path,
    sources: dict[str, Any],
    policy: Any,
) -> None:
    """Preserve API-supplied FORCED(9) separately from the selected estimated 3."""
    oracle = load_scientific_training_generation(sources["declared-api-proof-1"].path)
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        dataset_confidence_policy="selected_value_v2",
        dataset_family_target_policy=policy,
    )
    rows = extract(args, oracle)
    assert (
        rows.rows[0].metadata["target_confidence"]["raw_target_certainty"] == "ESTIMATE"
    )
    assert not rows.rows[0].is_exact
    assert not extract(replace(args, require_exact_or_terminal=True), oracle).rows
    proved = extract(replace(args, use_backed_up_value=False), oracle).rows[0]
    assert proved.target_value == 9 and proved.is_exact
    assert proved.metadata["target_confidence"]["training_target_certainty"] == "FORCED"


@pytest.mark.fast
def test_missing_confidence_never_uses_node_flag(
    tmp_path: Path, sources: dict[str, Any]
) -> None:
    """Leave an unavailable selected certificate unknown rather than inventing proof."""
    oracle = load_scientific_training_generation(sources["actual-endgame-2"].path)
    contracts = copy.deepcopy(oracle.value_contracts)
    contracts["0"]["values"]["direct"] = None
    contracts["0"]["certainties"]["direct"] = None
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        dataset_confidence_policy="selected_value_v2",
        use_backed_up_value=False,
    )
    rows = extract_rows_from_training_snapshot(
        args=args,
        snapshot=oracle.snapshot,
        generation=2,
        value_contracts=contracts,
        source_provenance=oracle.provenance,
    )
    assert not rows.rows[0].is_exact
    assert rows.rows[0].metadata["target_confidence"]["raw_target_certainty"] is None
    with pytest.raises(DatasetConfidenceError, match="explicit per-value"):
        extract_rows_from_training_snapshot(
            args=args, snapshot=oracle.snapshot, generation=2
        )


@pytest.mark.fast
def test_same_state_contradiction_is_rejected(
    tmp_path: Path, sources: dict[str, Any]
) -> None:
    """Reject opposing same-state certificates, without comparing distinct family states."""
    oracle = load_scientific_training_generation(sources["actual-endgame-2"].path)
    contracts = copy.deepcopy(oracle.value_contracts)
    contracts["0"]["values"]["direct"]["certainty"] = "FORCED"
    contracts["0"]["certainties"]["direct"] = "FORCED"
    with pytest.raises(DatasetConfidenceError, match="same state/objective"):
        extract_rows_from_training_snapshot(
            args=MorpionBootstrapArgs(
                work_dir=tmp_path,
                dataset_confidence_policy="selected_value_v2",
                use_backed_up_value=False,
            ),
            snapshot=oracle.snapshot,
            generation=2,
            value_contracts=contracts,
            source_provenance=oracle.provenance,
        )


def test_v1_stays_frozen_and_v2_rejects_corrupt_confidence(
    tmp_path: Path, sources: dict[str, Any]
) -> None:
    """Retain historical flags and reject unsupported new contracts explicitly."""
    oracle = load_scientific_training_generation(sources["actual-endgame-2"].path)
    legacy = extract(
        MorpionBootstrapArgs(
            work_dir=tmp_path,
            use_backed_up_value=False,
            dataset_family_target_policy="pv_exact_then_mean_prediction",
        ),
        oracle,
    )
    assert [row.target_value for row in legacy.rows] == [42, 42]
    assert all(row.is_exact for row in legacy.rows)
    assert legacy.metadata["dataset_version"] == 1
    assert all("target_confidence" not in row.metadata for row in legacy.rows)
    rows = extract(
        MorpionBootstrapArgs(
            work_dir=tmp_path,
            dataset_confidence_policy="selected_value_v2",
            use_backed_up_value=False,
        ),
        oracle,
    )
    payload = morpion_supervised_rows_to_dict(rows)
    payload["scientific_rows"][0]["is_exact"] = True
    with pytest.raises(DatasetConfidenceError, match="confidence"):
        morpion_supervised_rows_from_dict(payload)


@pytest.mark.fast
def test_equal_number_does_not_certify_estimate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep a neural 35 estimated even though this state's exact tree value is 35."""
    late, _ = endgame.__wrapped__()
    runner = make_runner(monkeypatch, state=late)
    runner.apply_reevaluation_patch(model_patch(runner, monkeypatch, 35, tmp_path))
    case = export_case(tmp_path, "equal-score", 1, tuple(runner.iter_profile_nodes()))
    oracle = load_scientific_training_generation(case.path)
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        use_backed_up_value=False,
        dataset_confidence_policy="selected_value_v2",
    )
    row = extract(args, oracle).rows[0]
    assert row.target_value == 35 and not row.is_exact
    assert row.metadata["target_confidence"]["raw_target_certainty"] == "ESTIMATE"


@pytest.mark.fast
@pytest.mark.parametrize("policy", sorted(SAFE_FAMILY_POLICIES))
def test_distinct_states_can_have_different_certificates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: Any
) -> None:
    """Accept API-declared distinct-state proofs 9/4, without claiming tiny boards solved."""
    runner = make_runner(monkeypatch)
    parent, child = tuple(runner.iter_profile_nodes())
    for node, score in ((parent, 9), (child, 4)):
        node.tree_evaluation.direct_value = canonical_value.make_forced_value(
            score=score
        )
        node.tree_evaluation.tree_value = canonical_value.make_estimate_value(score=2)
    case = export_case(
        tmp_path, "distinct-certificates", 1, tuple(runner.iter_profile_nodes())
    )
    oracle = load_scientific_training_generation(case.path)
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        use_backed_up_value=False,
        dataset_confidence_policy="selected_value_v2",
        dataset_family_target_policy=policy,
    )
    rows = extract(args, oracle)
    assert [row.target_value for row in rows.rows] == [9, 4]
    assert all(row.is_exact for row in rows.rows)
    assert (
        rows.rows[0].metadata["family_representative_node_id"] == rows.rows[1].node_id
    )


@pytest.mark.parametrize(
    "mutation", ["promote", "source", "missing", "contract", "version", "envelope"]
)
def test_serialized_confidence_cannot_invent_proof(
    tmp_path: Path, sources: dict[str, Any], mutation: str
) -> None:
    """Fail closed on scalar/source/certainty/envelope corruption, including promotion."""
    oracle = load_scientific_training_generation(sources["actual-endgame-2"].path)
    rows = extract(
        MorpionBootstrapArgs(
            work_dir=tmp_path,
            dataset_confidence_policy="selected_value_v2",
            use_backed_up_value=False,
        ),
        oracle,
    )
    data = morpion_supervised_rows_to_dict(rows)
    row = data["scientific_rows"][0]
    confidence = row["metadata"]["target_confidence"]
    if mutation == "promote":
        confidence["training_target_certainty"] = "FORCED"
        row["is_exact"] = True
    elif mutation == "source":
        confidence["raw_target_source"] = "tree_value"
    elif mutation == "missing":
        del confidence["raw_target_certainty"]
    elif mutation == "contract":
        data["metadata"]["dataset_confidence_contract"] = "future"
    elif mutation == "version":
        data["metadata"]["dataset_version"] = 3
    else:
        data["format_version"] = 1
    with pytest.raises(DatasetConfidenceError):
        morpion_supervised_rows_from_dict(data)


def test_ordinary_estimates_and_zero_blend_remain_usable(
    tmp_path: Path, sources: dict[str, Any]
) -> None:
    """Retain heuristic provenance even when blend zero leaves the number unchanged."""
    oracle = load_scientific_training_generation(sources["ordinary-1"].path)
    for policy in SAFE_FAMILY_POLICIES:
        args = MorpionBootstrapArgs(
            work_dir=tmp_path,
            dataset_confidence_policy="selected_value_v2",
            dataset_family_target_policy=policy,
            dataset_family_prediction_blend=0,
        )
        rows = extract(args, oracle)
        assert rows.rows and all(not row.is_exact for row in rows.rows)
        assert all(
            row.metadata["target_confidence"]["training_target_certainty"] == "ESTIMATE"
            for row in rows.rows
        )
        assert not extract(replace(args, require_exact_or_terminal=True), oracle).rows


def test_pipeline_identity_and_bounded_confidence_windows(
    tmp_path: Path, sources: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Publish a tiny v2 Dataset using only indexed one-node confidence windows."""
    from chipiron.environments.morpion.bootstrap import (
        sharded_training_v2 as scientific,
    )
    from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
        MorpionBootstrapPaths,
    )
    from chipiron.environments.morpion.bootstrap.pipeline import stages
    from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
        MorpionPipelineGenerationManifest,
        save_pipeline_manifest,
    )

    case = sources["actual-endgame-2"]
    oracle = load_scientific_training_generation(case.path)
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        pipeline_mode="artifact_pipeline",
        training_export_mode="sharded",
        dataset_confidence_policy="selected_value_v2",
        use_backed_up_value=False,
    )
    expected = extract(args, oracle)
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    save_pipeline_manifest(
        MorpionPipelineGenerationManifest(
            generation=2,
            created_at_utc="2026-10-21T00:00:00Z",
            tree_snapshot_path=str(case.path),
            dataset_status="not_started",
            training_status="not_started",
        ),
        paths.pipeline_manifest_path_for_generation(2),
    )
    original = ShardedTrainingGenerationView.load_value_contracts
    windows = []

    def recorded(self: Any, ids: tuple[str, ...]) -> Any:
        windows.append(len(ids))
        return original(self, ids)

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("full snapshot used by the indexed Dataset v2 path")

    monkeypatch.setattr(ShardedTrainingGenerationView, "load_value_contracts", recorded)
    monkeypatch.setattr(scientific, "load_scientific_training_generation", forbidden)
    monkeypatch.setattr(scientific, "TrainingTreeSnapshot", forbidden)
    monkeypatch.setattr(stages, "_load_training_snapshot_for_generation", forbidden)
    for _ in range(2):
        manifest = stages.run_pipeline_dataset_stage(
            args,
            generation=2,
            use_streaming_sharded_reader=True,
            allow_scientific_v2=True,
            reader_cache_dir=tmp_path / "cache",
        )
        assert (
            load_morpion_supervised_rows(paths.rows_jsonl_path_for_generation(2))
            == expected
        )
        for key in (
            "dataset_version",
            "dataset_confidence_contract",
            "dataset_extraction_identity",
            "scientific_source_provenance",
        ):
            assert manifest.metadata[key] == expected.metadata[key]
    assert windows and max(windows) == 1
    # The source index is shared across confidence policies; Dataset identities differ.
    with ShardedTrainingGenerationView(
        case.path, cache_dir=tmp_path / "cache", allow_scientific_v2=True
    ) as view:
        assert view.index_reused
    assert not list(tmp_path.rglob("*.tmp"))


def test_pipeline_v2_does_not_overwrite_v1_artifact(
    tmp_path: Path, sources: dict[str, Any]
) -> None:
    """Require a distinct output destination instead of reinterpreting saved v1 rows."""
    from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
        MorpionBootstrapPaths,
    )
    from chipiron.environments.morpion.bootstrap.pipeline import stages
    from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
        MorpionPipelineGenerationManifest,
        save_pipeline_manifest,
    )

    case = sources["actual-endgame-2"]
    oracle = load_scientific_training_generation(case.path)
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    paths.ensure_directories()
    manifest_path = paths.pipeline_manifest_path_for_generation(2)
    save_pipeline_manifest(
        MorpionPipelineGenerationManifest(
            generation=2,
            created_at_utc="2026-10-21T00:00:00Z",
            tree_snapshot_path=str(case.path),
            dataset_status="done",
            training_status="not_started",
        ),
        manifest_path,
    )
    rows = extract(MorpionBootstrapArgs(work_dir=tmp_path), oracle)
    rows_path = paths.rows_jsonl_path_for_generation(2)
    save_morpion_supervised_rows_streaming(
        rows=rows.rows, metadata=rows.metadata, path=rows_path
    )
    before = (rows_path.read_bytes(), manifest_path.read_bytes())
    with pytest.raises(DatasetConfidenceError, match="version disagrees"):
        stages.run_pipeline_dataset_stage(
            MorpionBootstrapArgs(
                work_dir=tmp_path,
                pipeline_mode="artifact_pipeline",
                training_export_mode="sharded",
                dataset_confidence_policy="selected_value_v2",
            ),
            generation=2,
            use_streaming_sharded_reader=True,
            allow_scientific_v2=True,
        )
    assert (rows_path.read_bytes(), manifest_path.read_bytes()) == before
    assert not paths.pipeline_dataset_claim_path_for_generation(2).exists()


def test_source_change_cannot_publish_confident_output(
    tmp_path: Path, sources: dict[str, Any]
) -> None:
    """Retain prior bytes and remove staged v2 rows if the source changes mid-pass."""
    case = sources["actual-endgame-2"]
    source_root = tmp_path / "source"
    shutil.copytree(case.path.parent.parent, source_root)
    path = source_root / "generation_000002" / "generation.json"
    oracle = load_scientific_training_generation(path)
    args = MorpionBootstrapArgs(
        work_dir=tmp_path,
        dataset_confidence_policy="selected_value_v2",
        use_backed_up_value=False,
    )
    expected = extract(args, oracle)
    output = tmp_path / "rows.jsonl"
    save_morpion_supervised_rows_streaming(
        rows=expected.rows, metadata=expected.metadata, path=output
    )
    original = output.read_bytes()
    with (
        ShardedTrainingGenerationView(
            path, cache_dir=tmp_path / "cache", allow_scientific_v2=True
        ) as view,
        disk_dataset_context(IndexedDatasetSource(view), args) as context,
    ):
        stream = context.stream(2)

        def changed_rows() -> Any:
            for row in stream.rows:
                yield row
                descriptor = json.loads(path.read_text())
                descriptor["snapshot"]["metadata"]["changed"] = True
                path.write_text(json.dumps(descriptor))

        with pytest.raises(ValueError, match="changed"):
            save_morpion_supervised_rows_streaming(
                rows=changed_rows(), metadata=stream.metadata, path=output
            )
    assert output.read_bytes() == original
    assert not output.with_suffix(".jsonl.tmp").exists()


def test_generation_identity_must_match(
    tmp_path: Path, sources: dict[str, Any]
) -> None:
    """Fail rather than associate correct confidence with a different generation."""
    case = sources["actual-endgame-2"]
    oracle = load_scientific_training_generation(case.path)
    args = MorpionBootstrapArgs(
        work_dir=tmp_path, dataset_confidence_policy="selected_value_v2"
    )
    with pytest.raises(DatasetConfidenceError, match="generation disagrees"):
        extract_rows_from_training_snapshot(
            args=args,
            snapshot=oracle.snapshot,
            generation=1,
            value_contracts=oracle.value_contracts,
            source_provenance=oracle.provenance,
        )
    with (
        ShardedTrainingGenerationView(
            case.path, cache_dir=tmp_path / "cache", allow_scientific_v2=True
        ) as view,
        disk_dataset_context(IndexedDatasetSource(view), args) as context,
        pytest.raises(DatasetConfidenceError, match="generation disagrees"),
    ):
        context.stream(1)
