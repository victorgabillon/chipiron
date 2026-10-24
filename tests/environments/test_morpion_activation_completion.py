"""Historical completion is metadata evidence; current recovery still verifies payloads."""

from __future__ import annotations

import copy
import json
import shutil
from typing import TYPE_CHECKING, Any

import pytest

from chipiron.environments.morpion.bootstrap import (
    MorpionBootstrapPaths,
    run_pipeline_growth_stage,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    PUBLICATION_KEY,
    GenerationPublicationError,
    committed_evaluator,
    generation_read_pin,
    publication_records,
)
from chipiron.environments.morpion.bootstrap.model_activation import (
    _runner,
    activate_model,
    activation_completion,
    prepare_model_activation_request,
)
from chipiron.environments.morpion.bootstrap.pipeline.stages import (
    run_pipeline_training_stage,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from tests.environments.morpion_activation_helpers import setup_activation
from tests.environments.morpion_fenced_dataset_helpers import (
    local_leaderboard,
    run_dataset,
)
from tests.environments.test_morpion_model_activation_crashes import run_child
from tests.environments.test_morpion_rolling_recovery_report import analyze

if TYPE_CHECKING:
    from pathlib import Path


def cold_state(paths: MorpionBootstrapPaths) -> Any:
    """Include full scientific Values, selector, topology and both RNG states."""
    from anemone.checkpoints import (
        build_search_checkpoint_payload,
        checkpoint_payload_to_jsonable,
    )

    state = load_bootstrap_run_state(paths.run_state_path)
    runner, effective = _runner(paths, state)
    _, model = committed_evaluator(paths.work_dir, state)
    try:
        runner.load_or_create(
            paths.work_dir / state.latest_runtime_checkpoint_path, model, effective
        )
        runner.validate_model_refresh_certificates()
        return checkpoint_payload_to_jsonable(
            build_search_checkpoint_payload(
                runner._runtime, state_codec=runner._state_codec
            )
        )
    finally:
        runner.discard_uncommitted_runtime()


@pytest.fixture(scope="module")
def history(tmp_path_factory: Any) -> Path:
    """Real accepted models, with Growth separating activation from later Training."""
    root = tmp_path_factory.mktemp("activation-completion")
    with pytest.MonkeyPatch.context() as monkeypatch:
        local_leaderboard(monkeypatch, root)
        args, paths, _ = setup_activation(root / "work")
        requests = []
        for generation in (1, 3):
            if generation == 3:
                run_dataset(args, generation=generation)
                run_pipeline_training_stage(args, generation=generation)
            request = prepare_model_activation_request(
                paths, training_generation=generation
            )
            requests.append(request)
            state = activate_model(paths, request)
            runner, _ = _runner(paths, state)
            producer = publication_records(state.metadata)[-1]["scientific_identity"][
                "producer"
            ]["identity"]
            run_pipeline_growth_stage(
                args,
                runner,
                max_cycles=1,
                checkpoint_bound_receipts=True,
                durable_generation_publication=True,
                scientific_v2_producer=producer,
                closed_scientific_snapshot=True,
            )
        assert requests[0]["candidate"] != requests[1]["candidate"]
        state = load_bootstrap_run_state(paths.run_state_path)
        assert state.generation == 5
        with generation_read_pin(paths.work_dir / state.latest_tree_snapshot_path):
            pass
        (paths.work_dir / "request.json").write_text(json.dumps(requests[0]))
    return paths.work_dir


@pytest.fixture
def workspace(history: Path, tmp_path: Path) -> MorpionBootstrapPaths:
    """Damage is confined to independent newly generated disposable copies."""
    shutil.copytree(history, tmp_path / "copy")
    return MorpionBootstrapPaths.from_work_dir(tmp_path / "copy")


def unavailable_old_payload(paths: MorpionBootstrapPaths) -> dict[str, Any]:
    """Check every observed root first; retain the seal, move only unpinned payloads."""
    report = analyze(paths, retain=(1,))
    assert report["validated"], report
    assert report["conditional_candidates"] == [2], report
    record = publication_records(
        load_bootstrap_run_state(paths.run_state_path).metadata
    )[1]
    assert record["descriptor_path"] in report["retained_commitment_files"]
    saved = paths.work_dir.parent / "unavailable"
    saved.mkdir()
    for name, relative in (
        ("checkpoint", record["checkpoint_path"]),
        ("science", record["tree_snapshot_path"]),
    ):
        path = paths.work_dir / relative
        if name == "science":
            path = path.parent
        path.rename(saved / name)
    return record


@pytest.mark.fast
def test_old_retry_without_old_tree_preserves_current(
    workspace: MorpionBootstrapPaths,
) -> None:
    """Exact M2 acknowledgement cannot reactivate M2 or change any M3 continuation state."""
    before = cold_state(workspace)
    authority = workspace.run_state_path.read_bytes()
    state = load_bootstrap_run_state(workspace.run_state_path)
    old = unavailable_old_payload(workspace)
    request = json.loads((workspace.work_dir / "request.json").read_text())
    assert activation_completion(workspace, state, old)["request"] == request
    assert activate_model(workspace, request) == state
    assert workspace.run_state_path.read_bytes() == authority
    assert cold_state(workspace) == before
    # The analyzer still refuses missing committed payloads: no deletion contract exists.
    assert not analyze(workspace)["validated"]


def test_recovery_death_with_old_completion(workspace: MorpionBootstrapPaths) -> None:
    """Real process death during retry repair cannot change the already committed M3."""
    before = cold_state(workspace)
    authority = workspace.run_state_path.read_bytes()
    unavailable_old_payload(workspace)
    child = run_child(workspace.work_dir, "recovery")
    assert child.returncode == 73, child.stdout[-2000:] + child.stderr[-2000:]
    assert workspace.run_state_path.read_bytes() == authority
    request = json.loads((workspace.work_dir / "request.json").read_text())
    activate_model(workspace, request)
    assert workspace.run_state_path.read_bytes() == authority
    assert cold_state(workspace) == before


@pytest.mark.parametrize(
    "damage",
    [
        "missing-seal",
        "changed-seal",
        "missing-binding",
        "forged-binding",
        "duplicate-binding",
        "conflicting-request",
        "historical-checkpoint-pointer",
        "historical-science-pointer",
        "current-science",
        "current-model",
    ],
)
def test_invalid_evidence_or_current_head_refuses(
    workspace: MorpionBootstrapPaths, damage: str
) -> None:
    """No missing-evidence fallback, content aliasing, old-model restore or new commit."""
    state = load_bootstrap_run_state(workspace.run_state_path)
    records = publication_records(state.metadata)
    request = json.loads((workspace.work_dir / "request.json").read_text())
    old, head = records[1], records[-1]
    seal = workspace.work_dir / old["descriptor_path"]
    if damage == "missing-seal":
        seal.rename(workspace.work_dir.parent / "unavailable-seal")
    elif damage == "changed-seal":
        seal.write_bytes(seal.read_bytes() + b" ")
    elif damage in {"missing-binding", "forged-binding", "duplicate-binding"}:
        raw = json.loads(workspace.run_state_path.read_text())
        entries = raw["metadata"][PUBLICATION_KEY]["generations"]
        if damage == "missing-binding":
            entries[1].pop("activation_request_sha256")
        elif damage == "forged-binding":
            entries[1].pop("activation_request_sha256")
            entries[0]["activation_request_sha256"] = old["activation_request_sha256"]
        else:
            entries[3]["activation_request_sha256"] = old["activation_request_sha256"]
        workspace.run_state_path.write_text(json.dumps(raw))
    elif damage in {"historical-checkpoint-pointer", "historical-science-pointer"}:
        raw = json.loads(workspace.run_state_path.read_text())
        key = (
            "checkpoint_path"
            if damage == "historical-checkpoint-pointer"
            else "tree_snapshot_path"
        )
        raw["metadata"][PUBLICATION_KEY]["generations"][1][key] = head[key]
        workspace.run_state_path.write_text(json.dumps(raw))
    elif damage == "conflicting-request":
        request["candidate"]["sha256"] = "e" * 64
    elif damage == "current-science":
        (workspace.work_dir / head["tree_snapshot_path"]).with_name(
            "updates.jsonl"
        ).write_text("corrupt")
    else:
        _, model = committed_evaluator(workspace.work_dir, state)
        (model / "param.pt").write_bytes(b"corrupt")
    authority = workspace.run_state_path.read_bytes()
    with pytest.raises((ValueError, OSError)):
        activate_model(workspace, request)
    assert workspace.run_state_path.read_bytes() == authority


def test_uncommitted_record_is_not_a_receipt(workspace: MorpionBootstrapPaths) -> None:
    """A sealed descriptor outside the committed ancestry is not completion authority."""
    state = load_bootstrap_run_state(workspace.run_state_path)
    record = copy.deepcopy(publication_records(state.metadata)[1])
    record["id"] = "e" * 32
    with pytest.raises(GenerationPublicationError, match="not committed"):
        activation_completion(workspace, state, record)


def test_current_analyzer_keeps_evidence_and_results(
    workspace: MorpionBootstrapPaths,
) -> None:
    """Current evaluator ownership and accepted results are fully verified, without new authority."""
    before = workspace.run_state_path.read_bytes()
    report = analyze(workspace, retain=(1,))
    assert report["validated"], report
    assert report["conditional_candidates"] == [2]
    assert report["current_contract_reclaimable_bytes"] == 0
    assert report["conditional_reclaimable_file_blocks"] > 0
    assert not report["deletion_authorized"]
    for inode in report["conditional_reclaimable_files"]:
        assert set(inode["paths"]).isdisjoint(report["retained_commitment_files"])
    assert [r["generation"] for r in report["activation_completion_evidence"]] == [2, 4]
    assert workspace.run_state_path.read_bytes() == before


@pytest.mark.parametrize("damage", ["projection", "rows", "accepted-model"])
def test_analyzer_does_not_bypass_current_ownership_or_results(
    workspace: MorpionBootstrapPaths, damage: str
) -> None:
    """Recognizing the new model contract does not waive evidence or accepted inputs."""
    from chipiron.environments.morpion.bootstrap import (
        dataset_publication,
        training_publication,
    )

    state = load_bootstrap_run_state(workspace.run_state_path)
    record = publication_records(state.metadata)[0]
    if damage == "projection":
        path = workspace.pipeline_active_model_path
        data = json.loads(path.read_text())
        data["metadata"]["committed_head"] = "e" * 32
        path.write_text(json.dumps(data))
    elif damage == "rows":
        proposal = dataset_publication._proposal(workspace, record["accepted_dataset"])
        path = workspace.work_dir / proposal["manifest"]["rows_path"]
        path.write_bytes(path.read_bytes() + b" ")
    else:
        proposal = training_publication._proposal(
            workspace, record["accepted_training"]
        )
        path = (
            workspace.work_dir
            / next(iter(proposal["models"].values()))["path"]
            / "param.pt"
        )
        path.write_bytes(b"corrupt")
    before = workspace.run_state_path.read_bytes()
    result = analyze(workspace, retain=(1,))
    assert not result["validated"] and result["blockers"]
    assert result["conditional_reclaimable_file_blocks"] == 0
    assert workspace.run_state_path.read_bytes() == before


def test_old_reader_and_explicit_pin_still_protect_payload(
    workspace: MorpionBootstrapPaths,
) -> None:
    """Removing the retry dependency cannot remove an independent lifetime pin."""
    from chipiron.environments.morpion.bootstrap.generation_publication_io import (
        capsule_lease,
    )

    record = publication_records(
        load_bootstrap_run_state(workspace.run_state_path).metadata
    )[1]
    with capsule_lease(workspace.work_dir, record["id"], version=3):
        report = analyze(workspace, retain=(1,))
        assert report["validated"]
        assert report["conditional_candidates"] == []
        assert report["generations"][1]["observed_reader"]
    report = analyze(workspace, retain=(1, 2))
    assert report["validated"] and report["conditional_candidates"] == []


def test_receipt_model_lookup_still_pins_old_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An alternate surviving M1 bundle does not bypass the actual model-pin scan."""
    from chipiron.environments.morpion.bootstrap import generation_publication
    from tests.environments.test_morpion_model_activation_edges import propose

    local_leaderboard(monkeypatch, tmp_path)
    args, paths, _ = setup_activation(tmp_path / "work")
    request = prepare_model_activation_request(paths, training_generation=1)
    assert propose(args).patch_written
    original_patch = paths.pipeline_reevaluation_patch_path.read_bytes()
    state = activate_model(paths, request)
    old_generation = state.generation
    for _ in range(3):
        runner, _ = _runner(paths, state)
        producer = publication_records(state.metadata)[-1]["scientific_identity"][
            "producer"
        ]["identity"]
        run_pipeline_growth_stage(
            args,
            runner,
            max_cycles=1,
            checkpoint_bound_receipts=True,
            durable_generation_publication=True,
            scientific_v2_producer=producer,
            closed_scientific_snapshot=True,
        )
        state = load_bootstrap_run_state(paths.run_state_path)
    with generation_read_pin(paths.work_dir / state.latest_tree_snapshot_path):
        pass
    report = analyze(paths, retain=(1,))
    assert report["validated"], report
    assert old_generation not in report["conditional_candidates"]
    old = next(g for g in report["generations"] if g["generation"] == old_generation)
    assert "exact-receipt-model-lookup" in old["rolling_reasons"]
    # Trace actual validation, retaining every byte: no simulated receipt or deletion.
    seen = []
    original_validate = generation_publication.validate_capsule

    def observe(root: Path, record: dict[str, Any]) -> dict[str, Any]:
        seen.append(record["generation"])
        return original_validate(root, record)

    monkeypatch.setattr(generation_publication, "validate_capsule", observe)
    authority = paths.run_state_path.read_bytes()
    paths.pipeline_reevaluation_patch_path.write_bytes(original_patch)
    activate_model(paths, request)
    assert old_generation in seen
    assert paths.run_state_path.read_bytes() == authority
    assert not paths.pipeline_reevaluation_patch_path.exists()
