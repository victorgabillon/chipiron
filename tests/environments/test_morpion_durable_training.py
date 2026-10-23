"""Authoritative Training results retain exact input and leave Growth unchanged."""

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from chipiron.environments.morpion.bootstrap import training_publication as publication
from chipiron.environments.morpion.bootstrap.derived.provenance import atomic_json
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    file_digest,
    read_object,
)
from chipiron.environments.morpion.bootstrap.pipeline.stages import (
    run_pipeline_training_stage,
)
from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
    load_pipeline_stage_claim,
    save_pipeline_stage_claim,
)
from chipiron.environments.morpion.bootstrap.run_state import (
    load_bootstrap_run_state,
    save_bootstrap_run_state,
)
from chipiron.environments.morpion.bootstrap.worker_result_io import publication_record
from tests.environments.morpion_durable_training_helpers import setup_training
from tests.environments.morpion_fenced_dataset_helpers import local_leaderboard


def test_normal_restart(tmp_path: Path, monkeypatch: Any) -> None:
    """Accept both models, repair status and keep all unrelated authority exact."""
    local_leaderboard(monkeypatch, tmp_path)
    args, paths, _ = setup_training(tmp_path / "workspace")
    before = load_bootstrap_run_state(paths.run_state_path)
    active = paths.pipeline_active_model_path.read_bytes()
    torch.manual_seed(31)
    rng = bytes(torch.get_rng_state().tolist())
    result = run_pipeline_training_stage(args, generation=1)
    assert result.training_status == "done"
    after = load_bootstrap_run_state(paths.run_state_path)
    record = publication_record(after, 1)
    reference = record[publication.ACCEPTED]
    assert (
        bytes.fromhex(
            publication._proposal(paths, reference)["admission"]["reproducibility"][
                "torch_rng_hex"
            ]
        )
        == rng
    )
    assert set(result.model_bundle_paths) == {"linear", "small_mlp"}
    assert paths.pipeline_active_model_path.read_bytes() == active
    assert not paths.pipeline_training_cursor_path.exists()
    record.pop(publication.ACCEPTED)
    assert after == before
    files = {
        p: p.read_bytes()
        for p in (paths.work_dir / publication.NAMESPACE / "objects").rglob("*")
        if p.is_file()
    }
    paths.pipeline_training_status_path_for_generation(1).unlink()
    paths.pipeline_manifest_path_for_generation(1).unlink()
    recovered = publication.accept_training(paths, reference)
    assert recovered == result
    assert files == {p: p.read_bytes() for p in files}
    assert run_pipeline_training_stage(args, generation=1) == result


@pytest.mark.parametrize(
    "damage",
    [
        "weights",
        "missing_weights",
        "missing_config",
        "dataset",
        "dataset_reference",
        "epoch",
        "workspace",
        "claim",
        "admission",
        "selection",
        "descriptor",
        "attempt",
    ],
)
def test_refuse_changed_result(tmp_path: Path, monkeypatch: Any, damage: str) -> None:
    """Every changed dependency fails closed before authoritative acceptance."""
    local_leaderboard(monkeypatch, tmp_path)
    args, paths, _ = setup_training(tmp_path / "workspace")
    accept = publication.accept_training
    authority = None

    def corrupt(paths: Any, reference: Any) -> Any:
        nonlocal authority
        directory = publication._directory(paths, reference["attempt_id"])
        proposal = read_object(directory / "proposal.json")
        model_dir = paths.work_dir / next(iter(proposal["models"].values()))["path"]
        if damage in {"weights", "missing_weights", "missing_config"}:
            target = model_dir / (
                "architecture_args.json" if damage == "missing_config" else "param.pt"
            )
            if damage == "missing_config":
                target = next(
                    p for p in model_dir.glob("*.json") if "manifest" not in p.name
                )
            if damage == "weights":
                target.write_bytes(b"corrupt")
            else:
                target.unlink()
        elif damage == "dataset":
            manifest = publication.dataset.project_dataset(
                paths, load_bootstrap_run_state(paths.run_state_path), 1, None
            )
            (paths.work_dir / manifest.rows_path).write_bytes(b"corrupt")
        elif damage in {"dataset_reference", "epoch", "workspace"}:
            state = load_bootstrap_run_state(paths.run_state_path)
            if damage == "dataset_reference":
                publication_record(state, 1)["accepted_dataset"][
                    "descriptor_sha256"
                ] = "0" * 64
            else:
                state.metadata["atomic_reevaluation"][damage + "_id"] = "0" * 32
            save_bootstrap_run_state(state, paths.run_state_path, durable=True)
        elif damage == "claim":
            claim_path = paths.pipeline_training_claim_path_for_generation(1)
            save_pipeline_stage_claim(
                replace(load_pipeline_stage_claim(claim_path), claim_id="replacement"),
                claim_path,
            )
        elif damage == "admission":
            proposal["admission"]["configuration"]["max_rows"] = 999
            atomic_json(directory / "proposal.json", proposal)
            reference["descriptor_sha256"] = file_digest(directory / "proposal.json")
        elif damage == "selection":
            proposal["status"]["selected_evaluator_name"] = "untrained"
            atomic_json(directory / "proposal.json", proposal)
            reference["descriptor_sha256"] = file_digest(directory / "proposal.json")
        elif damage == "descriptor":
            (directory / "proposal.json").write_text("{}")
        elif damage == "attempt":
            reference["attempt_id"] = "0" * 32
        authority = paths.run_state_path.read_bytes()
        return accept(paths, reference)

    monkeypatch.setattr(publication, "accept_training", corrupt)
    with pytest.raises((ValueError, FileNotFoundError, KeyError)):
        run_pipeline_training_stage(args, generation=1)
    assert paths.run_state_path.read_bytes() == authority
    assert publication.ACCEPTED not in publication_record(
        load_bootstrap_run_state(paths.run_state_path), 1
    )


def test_claim_takeover_and_conflicting_attempt(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """An old live attempt loses after claim replacement, including late cleanup."""
    local_leaderboard(monkeypatch, tmp_path)
    args, paths, _ = setup_training(tmp_path / "workspace")
    accept = publication.accept_training
    loser = None
    winner_bytes = None

    def takeover(paths: Any, reference: Any) -> Any:
        nonlocal loser, winner_bytes
        loser = copy.deepcopy(reference)
        claim_path = paths.pipeline_training_claim_path_for_generation(1)
        claim = load_pipeline_stage_claim(claim_path)
        save_pipeline_stage_claim(
            replace(claim, expires_at_utc="2000-01-01T00:00:00Z"), claim_path
        )
        monkeypatch.setattr(publication, "accept_training", accept)
        run_pipeline_training_stage(args, generation=1)
        winner_bytes = paths.run_state_path.read_bytes()
        return accept(paths, reference)

    monkeypatch.setattr(publication, "accept_training", takeover)
    with pytest.raises(ValueError, match="Conflicting"):
        run_pipeline_training_stage(args, generation=1)
    assert paths.run_state_path.read_bytes() == winner_bytes
    with pytest.raises(ValueError, match="Conflicting"):
        accept(paths, loser)


def test_expiry_without_replacement_can_complete(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """TTL does not revoke ownership until another claim replaces its identity."""
    local_leaderboard(monkeypatch, tmp_path)
    args, _paths, _ = setup_training(tmp_path / "workspace")
    accept = publication.accept_training

    def expire(paths: Any, reference: Any) -> Any:
        claim_path = paths.pipeline_training_claim_path_for_generation(1)
        save_pipeline_stage_claim(
            replace(
                load_pipeline_stage_claim(claim_path),
                expires_at_utc="2000-01-01T00:00:00Z",
            ),
            claim_path,
        )
        return accept(paths, reference)

    monkeypatch.setattr(publication, "accept_training", expire)
    assert run_pipeline_training_stage(args, generation=1).training_status == "done"


@pytest.mark.parametrize("forced", [None, "small_mlp"])
def test_numerical_model_parity(
    tmp_path: Path, monkeypatch: Any, forced: str | None
) -> None:
    """Same rows, RNG, kernels and policy produce exact model tensors and losses."""
    from chipiron.environments.morpion.bootstrap.bootstrap_memory import (
        memory_diagnostics_config_from_args,
    )
    from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
        MorpionBootstrapPaths,
    )
    from chipiron.environments.morpion.bootstrap.control import (
        MorpionBootstrapControl,
        save_bootstrap_control,
    )
    from chipiron.environments.morpion.bootstrap.cycle_training import (
        train_and_select_evaluators_streaming,
    )
    from chipiron.environments.morpion.bootstrap.profiling.memory_diagnostics import (
        MemoryDiagnostics,
    )
    from chipiron.environments.morpion.learning import (
        morpion_supervised_rows_source_from_path,
    )
    from chipiron.environments.morpion.players.evaluators.neural_networks.bundle import (
        load_morpion_model_bundle,
    )

    local_leaderboard(monkeypatch, tmp_path)
    args, paths, _ = setup_training(tmp_path / "workspace")
    before_state = load_bootstrap_run_state(paths.run_state_path)
    manifest = publication.dataset.project_dataset(paths, before_state, 1, None)
    rows = paths.work_dir / manifest.rows_path
    row_bytes = rows.read_bytes()
    control = MorpionBootstrapControl(force_evaluator=forced)
    save_bootstrap_control(control, paths.control_path)
    baseline = MorpionBootstrapPaths.from_work_dir(tmp_path / "reference")
    baseline.ensure_directories()
    # Baseline trainer creates caches alongside input: use a new exact copy too.
    baseline_rows = baseline.work_dir / "rows.jsonl"
    baseline_rows.write_bytes(row_bytes)
    memory = MemoryDiagnostics(memory_diagnostics_config_from_args(args))
    torch.manual_seed(91)
    try:
        expected = train_and_select_evaluators_streaming(
            args=args,
            paths=baseline,
            run_state=before_state,
            rows_path=baseline_rows,
            rows_source=morpion_supervised_rows_source_from_path(baseline_rows),
            generation=1,
            timestamp_utc="2026-10-23T12:00:00Z",
            resolved_evaluators_config=args.resolved_evaluators_config(),
            resolved_control=control,
            memory=memory,
            max_rows=args.training_max_rows,
            chunk_size=args.training_row_chunk_size,
        )
    finally:
        memory.close()
    torch.manual_seed(91)
    actual = run_pipeline_training_stage(args, generation=1)
    assert rows.read_bytes() == row_bytes
    assert actual.selected_evaluator_name == expected.selected_evaluator_name
    proposal = publication._proposal(
        paths,
        publication_record(load_bootstrap_run_state(paths.run_state_path), 1)[
            publication.ACCEPTED
        ],
    )
    status = publication.pipeline_training_status_from_dict(proposal["status"])
    assert status.selection_policy == expected.selection_policy
    for name in expected.model_bundle_paths:
        left = baseline.work_dir / expected.model_bundle_paths[name]
        right = paths.work_dir / actual.model_bundle_paths[name]
        assert (left / "param.pt").read_bytes() == (right / "param.pt").read_bytes()
        lm, la, _ = load_morpion_model_bundle(left)
        rm, ra, _ = load_morpion_model_bundle(right)
        assert la == ra
        assert all(
            torch.equal(t, rm.state_dict()[k]) for k, t in lm.state_dict().items()
        )
        for key in (
            "final_loss",
            "train_loss",
            "validation_loss",
            "train_mae",
            "validation_mae",
            "num_epochs",
            "batch_size",
            "learning_rate",
        ):
            assert getattr(expected.evaluator_results[name], key) == getattr(
                status.evaluator_results[name], key
            )


def test_growth_and_later_old_completion(tmp_path: Path, monkeypatch: Any) -> None:
    """Growth and newer Training can commit while an older admitted Training computes."""
    from tests.environments.morpion_atomic_reevaluation_helpers import publish_proposal
    from tests.environments.morpion_durable_closed_helpers import (
        ClosedPublicationRunner,
        run_closed_publication,
    )
    from tests.environments.morpion_durable_v2_helpers import scientific_args
    from tests.environments.morpion_fenced_dataset_helpers import run_dataset

    local_leaderboard(monkeypatch, tmp_path)
    args, paths, _ = setup_training(tmp_path / "workspace")
    publish = publication.publish_training
    model_before = paths.pipeline_active_model_path.read_bytes()

    def advance(paths: Any, attempt: Any, status: Any) -> Any:
        monkeypatch.setattr(publication, "publish_training", publish)
        publish_proposal(paths)
        run_closed_publication(
            scientific_args(paths.work_dir), ClosedPublicationRunner()
        )
        run_dataset(args, generation=2)
        assert run_pipeline_training_stage(args, generation=2).training_status == "done"
        return publish(paths, attempt, status)

    monkeypatch.setattr(publication, "publish_training", advance)
    assert run_pipeline_training_stage(args, generation=1).training_status == "done"
    state = load_bootstrap_run_state(paths.run_state_path)
    assert state.generation == 2
    assert all(publication.ACCEPTED in publication_record(state, g) for g in (1, 2))
    assert paths.pipeline_active_model_path.read_bytes() == model_before
    assert state.metadata["atomic_reevaluation"]["accepted_cursor"] is not None


@pytest.mark.parametrize(
    "damage", ["weights", "missing_config", "marker", "unknown_version"]
)
def test_accepted_integrity_on_restart(
    tmp_path: Path, monkeypatch: Any, damage: str
) -> None:
    """An accepted identity never licenses changed weights or a protocol downgrade."""
    local_leaderboard(monkeypatch, tmp_path)
    args, paths, _ = setup_training(tmp_path / "workspace")
    result = run_pipeline_training_stage(args, generation=1)
    if damage in {"marker", "unknown_version"}:
        state = load_bootstrap_run_state(paths.run_state_path)
        if damage == "marker":
            state.metadata.pop(publication.KEY)
        else:
            state.metadata[publication.KEY] = {"schema_version": 9}
        save_bootstrap_run_state(state, paths.run_state_path, durable=True)
    else:
        bundle = paths.work_dir / result.model_bundle_paths["linear"]
        if damage == "weights":
            (bundle / "param.pt").write_bytes(b"corrupted after acceptance")
        else:
            next(p for p in bundle.glob("*.json") if "manifest" not in p.name).unlink()
    before = paths.run_state_path.read_bytes()
    with pytest.raises(ValueError):
        run_pipeline_training_stage(args, generation=1)
    assert paths.run_state_path.read_bytes() == before


def test_pins_configuration_and_visibility(tmp_path: Path, monkeypatch: Any) -> None:
    """Actual computation pins its source; visibility is advisory until acceptance."""
    from chipiron.environments.morpion.bootstrap import cycle_training
    from chipiron.environments.morpion.bootstrap.generation_publication_io import (
        capsule_lease,
        growth_receipt_ownership,
    )
    from chipiron.environments.morpion.bootstrap.pipeline_orchestrator import (
        load_available_pipeline_manifests,
    )

    local_leaderboard(monkeypatch, tmp_path)
    args, paths, state = setup_training(tmp_path / "workspace")
    source_id = publication_record(state, 1)["id"]
    train = cycle_training.train_and_select_evaluators_streaming
    accept = publication.accept_training

    def pinned(**kwargs: Any) -> Any:
        with (
            pytest.raises(BlockingIOError),
            capsule_lease(
                paths.work_dir, source_id, version=3, exclusive=True, nonblocking=True
            ),
        ):
            pytest.fail("Source pin missing")
        with growth_receipt_ownership(paths):
            pass
        return train(**kwargs)

    def visible(paths: Any, reference: Any) -> Any:
        assert publication._directory(paths, reference["attempt_id"]).is_dir()
        assert (
            load_available_pipeline_manifests(paths)[1].training_status == "not_started"
        )
        return accept(paths, reference)

    monkeypatch.setattr(cycle_training, "train_and_select_evaluators_streaming", pinned)
    monkeypatch.setattr(publication, "accept_training", visible)
    run_pipeline_training_stage(args, generation=1)
    before = paths.run_state_path.read_bytes()
    with pytest.raises(ValueError, match="configuration"):
        run_pipeline_training_stage(
            replace(args, validation_seed=args.validation_seed + 1), generation=1
        )
    assert paths.run_state_path.read_bytes() == before


def test_nonfinite_loser_keeps_existing_rejection(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The existing result type still rejects nonfinite candidate metrics."""
    from chipiron.environments.morpion.bootstrap import cycle_training
    from chipiron.environments.morpion.players.evaluators.neural_networks.bundle import (
        MORPION_MANIFEST_FILE_NAME,
    )
    from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
        service,
    )

    local_leaderboard(monkeypatch, tmp_path)
    args, paths, _ = setup_training(tmp_path / "workspace")
    train = service.train_morpion_regressor_streaming

    def invalid_loser(spec: Any) -> Any:
        model, metrics = train(spec)
        if spec.training_args.model_kind == "linear":
            for key in ("final_loss", "train_loss", "validation_loss"):
                metrics[key] = float("nan")
            path = Path(spec.training_args.output_dir) / MORPION_MANIFEST_FILE_NAME
            payload = read_object(path)
            payload["metadata"].update({
                key: metrics[key]
                for key in ("final_loss", "train_loss", "validation_loss")
            })
            atomic_json(path, payload)
        return model, metrics

    monkeypatch.setattr(
        cycle_training, "train_morpion_regressor_streaming", invalid_loser
    )
    with pytest.raises(ValueError, match="finite"):
        run_pipeline_training_stage(args, generation=1)
    assert publication.ACCEPTED not in publication_record(
        load_bootstrap_run_state(paths.run_state_path), 1
    )


def test_tie_keeps_admitted_candidate_order(tmp_path: Path, monkeypatch: Any) -> None:
    """Sorted JSON object keys must not change tie-breaking candidate order."""
    from chipiron.environments.morpion.bootstrap import cycle_training
    from chipiron.environments.morpion.bootstrap.evaluator_config import (
        MorpionEvaluatorsConfig,
    )
    from chipiron.environments.morpion.players.evaluators.neural_networks.bundle import (
        MORPION_MANIFEST_FILE_NAME,
    )

    local_leaderboard(monkeypatch, tmp_path)
    args, _paths, _ = setup_training(tmp_path / "workspace")
    args = replace(
        args,
        evaluators_config=MorpionEvaluatorsConfig(
            evaluators=dict(
                reversed(list(args.resolved_evaluators_config().evaluators.items()))
            )
        ),
    )
    train = cycle_training.train_morpion_regressor_streaming

    def tied(spec: Any) -> Any:
        model, metrics = train(spec)
        for key in ("final_loss", "validation_loss"):
            metrics[key] = 1.0
        path = Path(spec.training_args.output_dir) / MORPION_MANIFEST_FILE_NAME
        payload = read_object(path)
        payload["metadata"].update({
            key: metrics[key] for key in ("final_loss", "validation_loss")
        })
        atomic_json(path, payload)
        return model, metrics

    monkeypatch.setattr(cycle_training, "train_morpion_regressor_streaming", tied)
    assert (
        run_pipeline_training_stage(args, generation=1).selected_evaluator_name
        == "small_mlp"
    )
