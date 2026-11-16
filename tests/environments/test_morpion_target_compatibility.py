"""Comparison may cross declared target policies; continuation must not."""

import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chipiron.environments.morpion.bootstrap import (
    continuous_publication as publication,
)
from chipiron.environments.morpion.bootstrap import promotion
from chipiron.environments.morpion.bootstrap import training_publication as training
from chipiron.environments.morpion.bootstrap.bootstrap_args import MorpionBootstrapArgs
from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    MorpionBootstrapPaths,
)
from chipiron.environments.morpion.bootstrap.cycle_dataset import (
    extract_rows_from_training_snapshot,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    artifact_digest,
    file_digest,
)
from chipiron.environments.morpion.bootstrap.learning_policy import (
    MorpionContinualLearningPolicy,
)
from chipiron.environments.morpion.bootstrap.sharded_training_v2 import (
    load_scientific_training_generation,
    save_scientific_training_generation,
)
from chipiron.environments.morpion.learning import (
    save_morpion_supervised_rows_streaming,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.bundle import (
    load_morpion_model_bundle,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
    MorpionTrainingArgs,
    train_morpion_regressor,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training.continuation import (
    MorpionTrainingContinuation,
)
from tests.environments.test_morpion_sharded_training_v2 import (
    PRODUCER,
    advance,
    nodes_fixture,
)


@pytest.fixture(scope="module")
def trained(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, MorpionTrainingArgs, dict[str, Any]]:
    """Train real backup and effective models; retain separate immutable manifests."""
    root = tmp_path_factory.mktemp("target-comparison")
    nodes = nodes_fixture()
    advance(nodes)
    path, _ = save_scientific_training_generation(
        nodes=nodes,
        root_node_id="0",
        output_dir=root / "source",
        generation=1,
        producer=PRODUCER,
    )
    source = load_scientific_training_generation(path)
    settings = None
    for name in ("legacy", "effective"):
        args = (
            MorpionBootstrapArgs(root, dataset_confidence_policy="selected_value_v2")
            if name == "legacy"
            else MorpionBootstrapArgs.new_scientific(root)
        )
        data = extract_rows_from_training_snapshot(
            args=args,
            snapshot=source.snapshot,
            generation=1,
            value_contracts=source.value_contracts,
            source_provenance=source.provenance,
        )
        rows = root / f"{name}.jsonl"
        save_morpion_supervised_rows_streaming(
            rows=data.rows, metadata=data.metadata, path=rows
        )
        settings = MorpionTrainingArgs(
            rows,
            root / name,
            model_kind="linear",
            num_epochs=1,
            batch_size=2,
            device="cpu",
            stable_validation_split=True,
            validation_fraction=0.5,
            validation_seed=17,
            continuation=MorpionTrainingContinuation("candidate", 7),
        )
        train_morpion_regressor(settings)
    assert settings is not None
    _, _, manifest = load_morpion_model_bundle(root / "effective")
    assert manifest.metadata["state_split"]["validation_count"] > 0
    return root, settings, manifest.metadata["state_split"]


def _reference(paths: MorpionBootstrapPaths, bundle: Path) -> Any:
    """Pin the real bundle bytes through J4.2's existing proof store."""
    return publication.put_proof(
        paths,
        {
            "path": str(bundle.relative_to(paths.work_dir)),
            "sha256": artifact_digest(bundle),
        },
    )


def test_cross_policy_comparison_uses_current_population_without_relabeling(
    trained: tuple[Path, MorpionTrainingArgs, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both comparison entry points evaluate a historical incumbent fairly."""
    root, settings, population = trained
    paths = MorpionBootstrapPaths.from_work_dir(root)
    args = replace(
        MorpionBootstrapArgs.new_scientific(root),
        validation_fraction=0.5,
        validation_seed=17,
        training_device="cpu",
        continual_learning=MorpionContinualLearningPolicy(),
    )
    from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
        evaluation,
    )

    original_evaluate = evaluation.evaluate_streaming_metrics
    evaluated = []

    def observe(**kwargs: Any) -> Any:
        evaluated.append((
            Path(kwargs["args"].dataset_file),
            kwargs["args"].validation_indices,
            kwargs["split"],
        ))
        return original_evaluate(**kwargs)

    monkeypatch.setattr(evaluation, "evaluate_streaming_metrics", observe)
    before = {name: artifact_digest(root / name) for name in ("legacy", "effective")}
    losses = {
        name: promotion._evaluate(
            paths,
            args,
            Path(settings.dataset_file),
            _reference(paths, root / name),
            population,
        )
        for name in before
    }
    _, _, candidate = load_morpion_model_bundle(root / "effective")
    result = SimpleNamespace(
        model_bundle_paths={"candidate": "effective"},
        evaluator_metrics={
            "candidate": SimpleNamespace(
                validation_loss=candidate.metadata["validation_loss"]
            )
        },
        selected_evaluator_name="candidate",
    )
    attempt = training.TrainingAttempt(
        {
            "incumbent": {"path": "legacy", "sha256": before["legacy"]},
            "round_number": 1,
        },
        root / "attempt",
    )
    compared = training._compare_incumbent(
        paths, paths, args, attempt, result, Path(settings.dataset_file)
    )
    assert len(evaluated) == 3
    assert all(observation == evaluated[0] for observation in evaluated)
    assert evaluated[0][0] == Path(settings.dataset_file)
    assert evaluated[0][2] == "validation"
    assert (
        load_morpion_model_bundle(root / "legacy")[2].metadata["state_split"]
        != population
    )
    assert compared["validation_population"] == population
    assert compared["incumbent"]["validation_loss"] == losses["legacy"]
    assert compared["candidate_validation_losses"]["candidate"] == losses["effective"]
    assert {name: artifact_digest(root / name) for name in before} == before
    assert load_morpion_model_bundle(root / "legacy")[2].target_kind == "backup_value"


@pytest.mark.parametrize(
    "damage",
    [
        "unknown_kind",
        "undeclared_kind",
        "missing_policy",
        "null_policy",
        "unknown_policy",
        "mismatched_kind",
        "missing_provenance",
        "wrong_lineage_policy",
        "unknown_family",
        "invalid_blend",
        "invalid_scientific_contract",
    ],
)
def test_comparison_and_parent_reject_bad_target_declarations(
    trained: tuple[Path, MorpionTrainingArgs, dict[str, Any]],
    tmp_path: Path,
    damage: str,
) -> None:
    """Rehashed malformed bundles still fail their semantic and provenance checks."""
    root, settings, population = trained
    damaged = tmp_path / "damaged"
    shutil.copytree(root / "effective", damaged)
    path = damaged / "morpion_manifest.json"
    payload = json.loads(path.read_text())
    metadata = payload["metadata"]
    if damage == "unknown_kind":
        payload["target_kind"] = "unknown"
    elif damage == "undeclared_kind":
        payload.pop("target_kind")
    elif damage == "missing_policy":
        metadata.pop("dataset_target_policy")
    elif damage == "null_policy":
        metadata["dataset_target_policy"] = None
    elif damage == "unknown_policy":
        metadata["dataset_target_policy"]["source"] = "unknown"
    elif damage == "mismatched_kind":
        payload["target_kind"] = "backup_value"
    elif damage == "missing_provenance":
        metadata.pop("scientific_source_provenance")
    elif damage == "unknown_family":
        metadata["dataset_target_policy"]["family_target_policy"] = "unknown"
    elif damage == "invalid_blend":
        metadata["dataset_target_policy"]["family_prediction_blend"] = -1.0
    elif damage == "invalid_scientific_contract":
        metadata["scientific_source_provenance"]["scientific_contract"] = "unknown"
    else:
        metadata["continuation"]["dataset_target_policy"]["source"] = "tree"
    path.write_text(json.dumps(payload))
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    args = replace(
        MorpionBootstrapArgs.new_scientific(tmp_path),
        validation_fraction=0.5,
        validation_seed=17,
        training_device="cpu",
    )
    with pytest.raises(ValueError):
        promotion._evaluate(
            paths,
            args,
            Path(settings.dataset_file),
            _reference(paths, damaged),
            population,
        )
    continuation = MorpionTrainingContinuation(
        "candidate",
        8,
        str(damaged),
        artifact_digest(damaged),
        file_digest(damaged / "param.pt"),
    )
    with pytest.raises(ValueError):
        train_morpion_regressor(
            replace(
                settings, output_dir=tmp_path / "refused", continuation=continuation
            )
        )
    assert not (tmp_path / "refused" / "param.pt").exists()


def test_comparison_preserves_digest_population_and_dataset_identity_checks(
    trained: tuple[Path, MorpionTrainingArgs, dict[str, Any]], tmp_path: Path
) -> None:
    """Valid semantics cannot bypass byte, held-out population or Dataset bindings."""
    root, settings, population = trained
    paths = MorpionBootstrapPaths.from_work_dir(root)
    args = replace(
        MorpionBootstrapArgs.new_scientific(root),
        validation_fraction=0.5,
        validation_seed=17,
        training_device="cpu",
        continual_learning=MorpionContinualLearningPolicy(),
    )
    reference = publication.put_proof(paths, {"path": "effective", "sha256": "0" * 64})
    with pytest.raises(ValueError, match="bytes changed"):
        promotion._evaluate(
            paths, args, Path(settings.dataset_file), reference, population
        )
    with pytest.raises(ValueError, match="population differs"):
        promotion._evaluate(
            paths,
            args,
            Path(settings.dataset_file),
            _reference(paths, root / "effective"),
            {**population, "validation_count": 999},
        )
    copied = tmp_path / "effective"
    shutil.copytree(root / "effective", copied)
    manifest_path = copied / "morpion_manifest.json"
    payload = json.loads(manifest_path.read_text())
    payload["metadata"]["dataset_extraction_identity"] = "0" * 64
    manifest_path.write_text(json.dumps(payload))
    result = SimpleNamespace(
        model_bundle_paths={"candidate": "effective"},
        evaluator_metrics={
            "candidate": SimpleNamespace(
                validation_loss=payload["metadata"]["validation_loss"]
            )
        },
        selected_evaluator_name="candidate",
    )
    attempt = training.TrainingAttempt(
        {
            "incumbent": {"path": "legacy", "sha256": artifact_digest(root / "legacy")},
            "round_number": 1,
        },
        root / "attempt",
    )
    with pytest.raises(ValueError, match="current Dataset"):
        training._compare_incumbent(
            paths,
            MorpionBootstrapPaths.from_work_dir(tmp_path),
            args,
            attempt,
            result,
            Path(settings.dataset_file),
        )


def test_continuation_keeps_policy_and_candidate_lineage_strict(
    trained: tuple[Path, MorpionTrainingArgs, dict[str, Any]], tmp_path: Path
) -> None:
    """Comparison permission never grants a cross-policy or cross-candidate parent."""
    root, settings, _ = trained
    for name, candidate in (("legacy", "candidate"), ("effective", "other")):
        bundle = root / name
        continuation = MorpionTrainingContinuation(
            candidate,
            8,
            str(bundle),
            artifact_digest(bundle),
            file_digest(bundle / "param.pt"),
        )
        with pytest.raises(ValueError, match=r"semantics differ|candidate lineage"):
            train_morpion_regressor(
                replace(settings, output_dir=tmp_path / name, continuation=continuation)
            )


def test_implicit_historical_bundle_keeps_replay_but_not_scientific_declaration(
    trained: tuple[Path, MorpionTrainingArgs, dict[str, Any]],
    tmp_path: Path,
) -> None:
    """The April fallback remains visible as implicit and never upgrades a policy."""
    import torch

    from chipiron.environments.morpion.players.evaluators.neural_networks.bundle import (
        _manifest_to_dict,
    )

    root, settings, population = trained
    historical = tmp_path / "historical"
    shutil.copytree(root / "legacy", historical)
    path = historical / "morpion_manifest.json"
    payload = json.loads(path.read_text())
    payload.pop("target_kind")
    path.write_text(json.dumps(payload))
    before = artifact_digest(historical)
    original, _, original_manifest = load_morpion_model_bundle(root / "legacy")
    replayed, model_args, manifest = load_morpion_model_bundle(historical)
    assert manifest.target_kind == "backup_value"
    assert not manifest.target_kind_declared
    assert "target_kind" not in _manifest_to_dict(manifest)
    inputs = torch.zeros(2, model_args.input_dim)
    torch.testing.assert_close(replayed(inputs), original(inputs), rtol=0, atol=0)
    paths = MorpionBootstrapPaths.from_work_dir(tmp_path)
    args = replace(
        MorpionBootstrapArgs.new_scientific(tmp_path),
        validation_fraction=0.5,
        validation_seed=17,
        training_device="cpu",
    )
    with pytest.raises(ValueError, match="declared model target"):
        promotion._evaluate(
            paths,
            args,
            Path(settings.dataset_file),
            _reference(paths, historical),
            population,
        )
    continuation = MorpionTrainingContinuation(
        "candidate", 8, str(historical), before, file_digest(historical / "param.pt")
    )
    with pytest.raises(ValueError, match="declared model target"):
        train_morpion_regressor(
            replace(
                settings, output_dir=tmp_path / "refused", continuation=continuation
            )
        )
    legacy_args = replace(args, dataset_target_source=None, use_backed_up_value=True)
    historical_loss = promotion._evaluate(
        paths,
        legacy_args,
        root / "legacy.jsonl",
        _reference(paths, historical),
        original_manifest.metadata["state_split"],
    )
    assert historical_loss == original_manifest.metadata["validation_loss"]
    # A frozen legacy recipe retains its historical continuation meaning.
    train_morpion_regressor(
        replace(
            settings,
            dataset_file=root / "legacy.jsonl",
            output_dir=tmp_path / "legacy-continued",
            continuation=continuation,
        )
    )
    assert artifact_digest(historical) == before
