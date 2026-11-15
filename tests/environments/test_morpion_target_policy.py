"""Explicit Anemone target slots, scientific confidence and legacy boundaries."""

import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from chipiron.environments.morpion.bootstrap.bootstrap_args import MorpionBootstrapArgs
from chipiron.environments.morpion.bootstrap.config import (
    bootstrap_config_from_args,
    bootstrap_config_from_dict,
    bootstrap_config_sha256,
    bootstrap_config_to_dict,
    validate_bootstrap_config_change,
)
from chipiron.environments.morpion.bootstrap.cycle_dataset import (
    extract_rows_from_training_snapshot,
    iter_rows_from_training_snapshot,
    streaming_rows_from_training_snapshot,
)
from chipiron.environments.morpion.bootstrap.derived.runtime import args_from_config
from chipiron.environments.morpion.bootstrap.sharded_training_reader import (
    ShardedTrainingGenerationView,
)
from chipiron.environments.morpion.bootstrap.sharded_training_v2 import (
    load_scientific_training_generation,
    save_scientific_training_generation,
)
from chipiron.environments.morpion.bootstrap.streaming_dataset import (
    IndexedDatasetSource,
    disk_dataset_context,
)
from chipiron.environments.morpion.learning import (
    load_morpion_supervised_rows,
    save_morpion_supervised_rows_streaming,
)
from chipiron.environments.morpion.learning.target_confidence import (
    DatasetConfidenceError,
)
from tests.environments.test_morpion_sharded_training_v2 import (
    PRODUCER,
    advance,
    flat,
    nodes_fixture,
)

if TYPE_CHECKING:
    from chipiron.environments.morpion.bootstrap.pv_family_targets import (
        PvFamilyTargetPolicy,
    )
    from chipiron.environments.morpion.learning.target_policy import TargetSource


@pytest.mark.parametrize("source", ["effective", "tree", "direct"])
@pytest.mark.parametrize("family", ["none", "pv_blend_mean_prediction"])
def test_target_policy_dag_eager_flat_and_indexed_equivalence(
    tmp_path: Path, source: TargetSource, family: PvFamilyTargetPolicy
) -> None:
    """A joined legal DAG keeps one target/Value association through all readers."""
    nodes = nodes_fixture()
    advance(nodes)
    path, _stats = save_scientific_training_generation(
        nodes=nodes,
        root_node_id="0",
        output_dir=tmp_path / "v2",
        generation=1,
        producer=PRODUCER,
    )
    scientific = load_scientific_training_generation(path)
    args = replace(
        MorpionBootstrapArgs.new_scientific(tmp_path, target_source=source),
        dataset_family_target_policy=family,
    )
    expected = extract_rows_from_training_snapshot(
        snapshot=scientific.snapshot,
        args=args,
        generation=1,
        value_contracts=scientific.value_contracts,
        source_provenance=scientific.provenance,
    )
    assert expected.rows
    assert expected.metadata["dataset_target_source"] == source
    # Flat serialization carries scalars; scientific certainty stays an explicit sidecar.
    flat_rows = extract_rows_from_training_snapshot(
        snapshot=flat(nodes),
        args=args,
        generation=1,
        value_contracts=scientific.value_contracts,
        source_provenance=scientific.provenance,
    )
    assert flat_rows == expected
    assert (
        tuple(
            iter_rows_from_training_snapshot(
                snapshot=flat(nodes),
                args=args,
                generation=1,
                value_contracts=scientific.value_contracts,
                source_provenance=scientific.provenance,
            )
        )
        == expected.rows
    )
    stream = streaming_rows_from_training_snapshot(
        snapshot=flat(nodes),
        args=args,
        generation=1,
        value_contracts=scientific.value_contracts,
        source_provenance=scientific.provenance,
    )
    assert tuple(stream.rows) == expected.rows
    assert stream.metadata == expected.metadata
    by_id = {node.node_id: node for node in scientific.snapshot.nodes}
    for row in expected.rows:
        raw = getattr(by_id[row.node_id], f"{source}_value_scalar")
        assert row.metadata["raw_target"] == raw
        confidence = row.metadata["target_confidence"]
        assert confidence["raw_target_source"] == f"{source}_value"
        assert (
            confidence["raw_target_certainty"]
            == scientific.value_contracts[row.node_id]["certainties"][source]
        )
        if family == "none":
            assert row.target_value == raw
        if source == "effective":
            assert (
                row.metadata["effective_value_source"]
                == by_id[row.node_id].effective_value_source
            )
    partial = next(row for row in expected.rows if row.node_id == "2")
    assert partial.metadata["raw_target"] == (3 if source == "tree" else 9)
    assert by_id["3"].parent_ids == ("1", "2")
    with (
        ShardedTrainingGenerationView(
            path, cache_dir=tmp_path / "cache", allow_scientific_v2=True
        ) as view,
        disk_dataset_context(IndexedDatasetSource(view), args) as context,
    ):
        indexed = context.stream(1)
        assert tuple(indexed.rows) == expected.rows
        assert indexed.metadata == expected.metadata
        save_morpion_supervised_rows_streaming(
            rows=context.rows(),
            metadata=indexed.metadata,
            path=tmp_path / "indexed.jsonl",
        )
    save_morpion_supervised_rows_streaming(
        rows=expected.rows, metadata=expected.metadata, path=tmp_path / "eager.jsonl"
    )
    assert (tmp_path / "indexed.jsonl").read_bytes() == (
        tmp_path / "eager.jsonl"
    ).read_bytes()
    assert load_morpion_supervised_rows(tmp_path / "eager.jsonl") == expected


def test_historical_configuration_identity_and_explicit_policy_refusals(
    tmp_path: Path,
) -> None:
    """Missing target policy means exactly the historical bool, never effective."""
    legacy = bootstrap_config_from_args(MorpionBootstrapArgs(tmp_path))
    payload = bootstrap_config_to_dict(legacy)
    assert isinstance(payload["dataset"], dict)
    assert "target_source" not in payload["dataset"]
    assert (
        bootstrap_config_to_dict(
            bootstrap_config_from_dict(json.loads(json.dumps(payload)))
        )
        == payload
    )
    current = bootstrap_config_from_args(MorpionBootstrapArgs.new_scientific(tmp_path))
    assert current.dataset.target_source == "effective"
    assert current.dataset.use_backed_up_value is None
    assert bootstrap_config_sha256(current) != bootstrap_config_sha256(legacy)
    assert bootstrap_config_from_dict(bootstrap_config_to_dict(current)) == current
    with pytest.raises(ValueError, match="target_source"):
        validate_bootstrap_config_change(legacy, current)
    with pytest.raises(ValueError, match="target_source"):
        MorpionBootstrapArgs(tmp_path, dataset_target_source="effective")
    for source in ("tree", "direct"):
        alternative = replace(
            current, dataset=replace(current.dataset, target_source=source)
        )
        with pytest.raises(ValueError, match="target_source"):
            validate_bootstrap_config_change(current, alternative)


@pytest.mark.parametrize(
    ("source", "backed_up"),
    [
        (None, True),
        (None, False),
        ("effective", None),
        ("tree", None),
        ("direct", None),
    ],
)
def test_derived_configuration_restores_target_policy(
    tmp_path: Path, source: TargetSource | None, backed_up: bool | None
) -> None:
    """Persisted recipes retain exact target selection and confidence on derivation."""
    original = (
        MorpionBootstrapArgs(tmp_path, use_backed_up_value=backed_up)
        if source is None
        else MorpionBootstrapArgs.new_scientific(tmp_path, target_source=source)
    )
    payload = bootstrap_config_to_dict(bootstrap_config_from_args(original))
    persisted = bootstrap_config_from_dict(json.loads(json.dumps(payload)))
    restored = args_from_config(tmp_path / "derived", persisted)
    assert restored.work_dir == tmp_path / "derived"
    assert restored.dataset_target_source == source
    assert restored.use_backed_up_value is backed_up
    assert restored.dataset_confidence_policy == original.dataset_confidence_policy
    assert bootstrap_config_to_dict(bootstrap_config_from_args(restored)) == payload
    assert bootstrap_config_sha256(bootstrap_config_from_args(restored)) == (
        bootstrap_config_sha256(persisted)
    )


def test_explicit_targets_require_scientific_certainty(tmp_path: Path) -> None:
    """Plain flat/legacy data cannot invent the missing per-value certificate."""
    snapshot = flat(nodes_fixture())
    for source in ("effective", "tree", "direct"):
        args = MorpionBootstrapArgs.new_scientific(tmp_path, target_source=source)
        with pytest.raises(DatasetConfidenceError, match="per-value"):
            extract_rows_from_training_snapshot(
                args=args, snapshot=snapshot, generation=1
            )
        with pytest.raises(DatasetConfidenceError, match="per-value"):
            extract_rows_from_training_snapshot(
                args=replace(args, dataset_confidence_policy="legacy_v1"),
                snapshot=snapshot,
                generation=1,
            )


@pytest.mark.parametrize("source", ["effective", "tree", "direct"])
@pytest.mark.parametrize("complete", [False, True])
def test_real_search_selected_certainty_and_terminal_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: TargetSource,
    complete: bool,
) -> None:
    """Real direct winners and full-opening FORCED proofs keep their own certainty."""
    from tests.environments.test_morpion_effective_provenance import _runner

    runner = _runner(monkeypatch, complete=complete)
    path, _stats = save_scientific_training_generation(
        nodes=tuple(runner.iter_profile_nodes()),
        root_node_id="0",
        output_dir=tmp_path / "source",
        generation=1,
        producer=PRODUCER,
    )
    scientific = load_scientific_training_generation(path)
    args = MorpionBootstrapArgs.new_scientific(tmp_path, target_source=source)
    rows = extract_rows_from_training_snapshot(
        args=args,
        snapshot=scientific.snapshot,
        generation=1,
        value_contracts=scientific.value_contracts,
        source_provenance=scientific.provenance,
    )
    root = rows.rows[0]
    assert root.target_value == getattr(
        scientific.snapshot.nodes[0], f"{source}_value_scalar"
    )
    assert root.is_exact is (complete and source != "direct")
    assert root.metadata["target_confidence"]["training_target_certainty"] == (
        "FORCED" if root.is_exact else "ESTIMATE"
    )
    assert root.is_terminal is False
    if source == "tree":
        assert len(rows.rows) == 1  # No direct fallback on the leaf.
    elif complete:
        terminal = rows.rows[1]
        assert terminal.is_terminal and terminal.is_exact
        assert (
            terminal.metadata["target_confidence"]["training_target_certainty"]
            == "TERMINAL"
        )
    if source == "effective":
        corrupted = replace(
            scientific.snapshot,
            nodes=(
                replace(
                    scientific.snapshot.nodes[0],
                    effective_value_source="unavailable_legacy",
                ),
            ),
        )
        with pytest.raises(ValueError, match="unavailable"):
            extract_rows_from_training_snapshot(
                args=args,
                snapshot=corrupted,
                generation=1,
                value_contracts=scientific.value_contracts,
                source_provenance=scientific.provenance,
            )


def test_new_scientific_cli_default_and_resume_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only fresh preparation defaults to effective; resumed recipes remain explicit."""
    from chipiron.environments.morpion.bootstrap.launcher import (
        _scientific_args,
        launcher_args_from_cli,
    )

    monkeypatch.setenv("MORPION_LEADERBOARD_PATH", str(tmp_path / "leaderboard.jsonl"))
    base = [
        "--scientific",
        "--work-dir",
        str(tmp_path),
        "--scientific-operation",
        "prepare-continuous",
    ]
    fresh = launcher_args_from_cli(base).bootstrap_args
    assert (
        fresh.dataset_target_source == "effective" and fresh.use_backed_up_value is None
    )
    direct = launcher_args_from_cli([
        *base,
        "--dataset-target-source",
        "direct",
    ]).bootstrap_args
    assert direct.dataset_target_source == "direct"
    legacy = launcher_args_from_cli([*base, "--no-use-backed-up-value"]).bootstrap_args
    assert legacy.dataset_target_source is None and legacy.use_backed_up_value is False
    with pytest.raises(SystemExit):
        launcher_args_from_cli([
            *base,
            "--dataset-target-source",
            "effective",
            "--use-backed-up-value",
        ])
    config = bootstrap_config_from_args(
        replace(
            fresh, runtime_checkpoint_format="sharded", training_export_mode="sharded"
        )
    )
    with pytest.raises(ValueError, match="target policy changed"):
        _scientific_args(direct, config)
    resumed = _scientific_args(MorpionBootstrapArgs(tmp_path), config)
    assert resumed.dataset_target_source == "effective"


@pytest.mark.parametrize("streaming", [False, True])
def test_training_bundle_and_optimizer_provenance(
    tmp_path: Path, streaming: bool
) -> None:
    """One tiny training step persists semantics; incompatible continuation fails."""
    from chipiron.environments.morpion.bootstrap.generation_publication_io import (
        artifact_digest,
        file_digest,
    )
    from chipiron.environments.morpion.players.evaluators.neural_networks.bundle import (
        load_morpion_model_bundle,
    )
    from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
        MorpionStreamingTrainingArgs,
        MorpionTrainingArgs,
        train_morpion_regressor,
        train_morpion_regressor_streaming,
    )
    from chipiron.environments.morpion.players.evaluators.neural_networks.training.continuation import (
        MorpionTrainingContinuation,
    )

    nodes = nodes_fixture()
    advance(nodes)
    path, _stats = save_scientific_training_generation(
        nodes=nodes,
        root_node_id="0",
        output_dir=tmp_path / "source",
        generation=1,
        producer=PRODUCER,
    )
    scientific = load_scientific_training_generation(path)

    def dataset(source: TargetSource) -> Path:
        rows = extract_rows_from_training_snapshot(
            args=MorpionBootstrapArgs.new_scientific(tmp_path, target_source=source),
            snapshot=scientific.snapshot,
            generation=1,
            value_contracts=scientific.value_contracts,
            source_provenance=scientific.provenance,
        )
        result = tmp_path / f"{source}.jsonl"
        save_morpion_supervised_rows_streaming(
            rows=rows.rows, metadata=rows.metadata, path=result
        )
        return result

    args = MorpionTrainingArgs(
        dataset_file=dataset("effective"),
        output_dir=tmp_path / "model",
        model_kind="linear",
        num_epochs=1,
        batch_size=4,
        validation_fraction=0.0,
        device="cpu",
        shuffle=False,
        continuation=MorpionTrainingContinuation(candidate_name="linear", seed=7),
    )

    def train(settings: MorpionTrainingArgs) -> None:
        if streaming:
            train_morpion_regressor_streaming(
                MorpionStreamingTrainingArgs(settings, row_chunk_size=2)
            )
        else:
            train_morpion_regressor(settings)

    train(args)
    _model, _model_args, manifest = load_morpion_model_bundle(Path(args.output_dir))
    assert manifest.target_kind == "effective_value"
    policy = manifest.metadata["dataset_target_policy"]
    assert policy["source"] == "effective" and policy["family_target_policy"] == "none"
    assert manifest.metadata["continuation"]["dataset_target_policy"] == policy
    assert (tmp_path / "model" / "optimizer.pt").is_file()
    assert manifest.metadata["dataset_extraction_identity"]
    parent = MorpionTrainingContinuation(
        candidate_name="linear",
        seed=8,
        parent_bundle=str(args.output_dir),
        parent_bundle_sha256=artifact_digest(Path(args.output_dir)),
        parent_weights_sha256=file_digest(tmp_path / "model" / "param.pt"),
    )
    train(replace(args, output_dir=tmp_path / "continued", continuation=parent))
    with pytest.raises(ValueError, match="output semantics"):
        train(
            replace(
                args,
                dataset_file=dataset("tree"),
                output_dir=tmp_path / "refused",
                continuation=parent,
            )
        )


def test_scientific_publication_carries_policy_through_checkpoint_and_training(
    tmp_path: Path,
) -> None:
    """Existing publication identities bind the new recipe without coordination changes."""
    from chipiron.environments.morpion.bootstrap.config import load_bootstrap_config
    from chipiron.environments.morpion.bootstrap.generation_publication_io import (
        publication_records,
    )
    from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
        load_pipeline_manifest,
    )
    from chipiron.environments.morpion.bootstrap.run_state import (
        load_bootstrap_run_state,
    )
    from chipiron.environments.morpion.bootstrap.training_publication import (
        accepted_selected_model,
    )
    from chipiron.environments.morpion.players.evaluators.neural_networks.bundle import (
        load_morpion_model_bundle,
    )
    from tests.environments.morpion_activation_helpers import setup_activation

    _args, paths, _initial = setup_activation(
        tmp_path / "work", target_source="effective"
    )
    state = load_bootstrap_run_state(paths.run_state_path)
    config = load_bootstrap_config(paths.bootstrap_config_path)
    assert config.dataset.target_source == "effective"
    assert state.metadata["bootstrap_config_hash"] == bootstrap_config_sha256(config)
    head = publication_records(state.metadata)[-1]
    assert head["scientific_identity"]["producer"]["identity"][
        "config_sha256"
    ] == bootstrap_config_sha256(config)
    assert state.latest_runtime_checkpoint_path is not None
    assert (paths.work_dir / state.latest_runtime_checkpoint_path).exists()
    dataset = load_pipeline_manifest(paths.pipeline_manifest_path_for_generation(1))
    assert dataset.metadata["dataset_target_source"] == "effective"
    _reference, selected = accepted_selected_model(paths, state, 1)
    _model, _model_args, manifest = load_morpion_model_bundle(
        paths.work_dir / selected["path"]
    )
    assert manifest.target_kind == "effective_value"
    assert manifest.metadata["dataset_target_policy"]["source"] == "effective"
    assert (
        manifest.metadata["dataset_extraction_identity"]
        == dataset.metadata["dataset_extraction_identity"]
    )


def test_effective_source_certainty_and_serialized_policy_refusals(
    tmp_path: Path,
) -> None:
    """A scalar cannot borrow another slot's certainty or a different header policy."""
    import copy

    from chipiron.environments.morpion.learning.tree_to_dataset import (
        morpion_supervised_rows_from_dict,
        morpion_supervised_rows_to_dict,
    )

    nodes = nodes_fixture()
    advance(nodes)
    path, _stats = save_scientific_training_generation(
        nodes=nodes,
        root_node_id="0",
        output_dir=tmp_path / "source",
        generation=1,
        producer=PRODUCER,
    )
    scientific = load_scientific_training_generation(path)
    args = MorpionBootstrapArgs.new_scientific(tmp_path)
    contracts = copy.deepcopy(dict(scientific.value_contracts))
    contracts["2"]["certainties"]["effective"] = None
    with pytest.raises(DatasetConfidenceError):
        extract_rows_from_training_snapshot(
            args=args,
            snapshot=scientific.snapshot,
            generation=1,
            value_contracts=contracts,
            source_provenance=scientific.provenance,
        )
    wrong_source = replace(
        scientific.snapshot,
        nodes=tuple(
            replace(node, effective_value_source="tree_child")
            if node.node_id == "2"
            else node
            for node in scientific.snapshot.nodes
        ),
    )
    with pytest.raises(DatasetConfidenceError, match="source and Value"):
        extract_rows_from_training_snapshot(
            args=args,
            snapshot=wrong_source,
            generation=1,
            value_contracts=scientific.value_contracts,
            source_provenance=scientific.provenance,
        )
    rows = extract_rows_from_training_snapshot(
        args=args,
        snapshot=scientific.snapshot,
        generation=1,
        value_contracts=scientific.value_contracts,
        source_provenance=scientific.provenance,
    )
    wrong_header = replace(
        rows, metadata={**rows.metadata, "dataset_target_source": "tree"}
    )
    with pytest.raises(DatasetConfidenceError, match="policies disagree"):
        morpion_supervised_rows_to_dict(wrong_header)
    payload = morpion_supervised_rows_to_dict(rows)
    assert isinstance(payload["metadata"], dict)
    payload["metadata"]["dataset_target_source"] = "tree"
    with pytest.raises(DatasetConfidenceError, match="policies disagree"):
        morpion_supervised_rows_from_dict(payload)
    save_morpion_supervised_rows_streaming(
        rows=rows.rows, metadata=rows.metadata, path=tmp_path / "rows.jsonl"
    )
    lines = (tmp_path / "rows.jsonl").read_text().splitlines()
    header = json.loads(lines[0])
    header["metadata"]["dataset_target_source"] = "tree"
    lines[0] = json.dumps(header)
    (tmp_path / "rows.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(DatasetConfidenceError, match="policies disagree"):
        load_morpion_supervised_rows(tmp_path / "rows.jsonl")


def test_effective_certificates_survive_explicit_family_smoothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A full-opening proof and terminal Value remain raw despite smoothing."""
    from tests.environments.test_morpion_effective_provenance import _runner

    runner = _runner(monkeypatch, complete=True)
    path, _stats = save_scientific_training_generation(
        nodes=tuple(runner.iter_profile_nodes()),
        root_node_id="0",
        output_dir=tmp_path / "source",
        generation=1,
        producer=PRODUCER,
    )
    scientific = load_scientific_training_generation(path)
    rows = extract_rows_from_training_snapshot(
        args=replace(
            MorpionBootstrapArgs.new_scientific(tmp_path),
            dataset_family_target_policy="pv_mean_prediction",
        ),
        snapshot=scientific.snapshot,
        generation=1,
        value_contracts=scientific.value_contracts,
        source_provenance=scientific.provenance,
    )
    assert len(rows.rows) == 2
    for row in rows.rows:
        assert row.is_exact
        assert row.target_value == row.metadata["raw_target"] == 35
        assert (
            row.metadata["target_confidence"]["training_target_source"] == "raw_target"
        )
