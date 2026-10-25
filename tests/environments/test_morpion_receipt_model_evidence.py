"""Exact old receipts authenticate seals/models without reopening unrelated trees."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from chipiron.environments.morpion.bootstrap import (
    MorpionBootstrapPaths,
    run_pipeline_growth_stage,
)
from chipiron.environments.morpion.bootstrap import checkpoint_patch_receipts as cr
from chipiron.environments.morpion.bootstrap import generation_publication as gp
from chipiron.environments.morpion.bootstrap import generation_publication_io as io
from chipiron.environments.morpion.bootstrap.model_activation import (
    _runner,
    activate_model,
    prepare_model_activation_request,
)
from chipiron.environments.morpion.bootstrap.pipeline.stages import (
    run_pipeline_training_stage,
)
from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
    reevaluation_patch_from_dict,
)
from chipiron.environments.morpion.bootstrap.reevaluation_publication import (
    publish_reevaluation_patch_if_absent,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from tests.environments.morpion_activation_helpers import setup_activation
from tests.environments.morpion_fenced_dataset_helpers import (
    local_leaderboard,
    run_dataset,
)
from tests.environments.test_morpion_activation_completion import cold_state
from tests.environments.test_morpion_model_activation_edges import propose
from tests.environments.test_morpion_rolling_recovery_report import analyze


def acknowledge(paths: MorpionBootstrapPaths) -> bool:
    """Exercise the real owned recovery path without applying any patch."""
    with io.growth_receipt_ownership(paths):
        state = load_bootstrap_run_state(paths.run_state_path)
        producer = io.publication_records(state.metadata)[-1]["scientific_identity"][
            "producer"
        ]["identity"]
        session = cr.GrowthPatchReceiptSession(
            paths,
            object(),
            durable_generation_publication=True,
            scientific_v2_producer=producer,
            closed_scientific_snapshot=True,
        )
        return session.acknowledge_pending(state)


def grow(args: Any, paths: MorpionBootstrapPaths) -> Any:
    """Restore and advance the actual current evaluator through normal Growth."""
    state = load_bootstrap_run_state(paths.run_state_path)
    runner, _ = _runner(paths, state)
    run_pipeline_growth_stage(
        args,
        runner,
        max_cycles=1,
        checkpoint_bound_receipts=True,
        durable_generation_publication=True,
        closed_scientific_snapshot=True,
        scientific_v2_producer=io.publication_records(state.metadata)[-1][
            "scientific_identity"
        ]["producer"]["identity"],
    )
    return load_bootstrap_run_state(paths.run_state_path)


@pytest.fixture(scope="module")
def history(tmp_path_factory: Any) -> Any:
    """Two actual Training acceptances, activations and neural receipt commits."""
    root = tmp_path_factory.mktemp("receipt-model")
    with pytest.MonkeyPatch.context() as monkeypatch:
        local_leaderboard(monkeypatch, root)
        args, paths, _ = setup_activation(root / "work")
        for index, generation in enumerate((1, 4)):
            if index:
                run_dataset(args, generation=generation)
                run_pipeline_training_stage(args, generation=generation)
            request = prepare_model_activation_request(
                paths, training_generation=generation
            )
            assert propose(args).patch_written
            (root / f"patch-{index}.json").write_bytes(
                paths.pipeline_reevaluation_patch_path.read_bytes()
            )
            (root / f"request-{index}.json").write_text(json.dumps(request))
            activate_model(paths, request)
            grow(args, paths)
        state = load_bootstrap_run_state(paths.run_state_path)
        assert state.generation == 7 and len(cr.receipt_ledger(state)) == 2
        with io.generation_read_pin(paths.work_dir / state.latest_tree_snapshot_path):
            pass
    return args, paths, root


@pytest.fixture
def workspace(history: Any, tmp_path: Path) -> Any:
    """Only independently generated copies may be damaged."""
    args, original, evidence = history
    root = tmp_path / "copy"
    shutil.copytree(original.work_dir, root)
    return (
        replace(args, work_dir=root),
        MorpionBootstrapPaths.from_work_dir(root),
        evidence,
    )


def publish(paths: MorpionBootstrapPaths, evidence: Path, index: int = 0) -> None:
    """Republish exactly the producer's original bytes through its create-only API."""
    raw = (evidence / f"patch-{index}.json").read_bytes()
    assert publish_reevaluation_patch_if_absent(
        reevaluation_patch_from_dict(json.loads(raw)),
        paths.pipeline_reevaluation_patch_path,
    )
    assert paths.pipeline_reevaluation_patch_path.read_bytes() == raw


@pytest.mark.parametrize("index", [0, 1])
def test_old_receipt_repeated_cold_recovery(workspace: Any, index: int) -> None:
    """M1/M2 acknowledgements at M3 change neither scientific state nor authority."""
    _, paths, evidence = workspace
    authority = paths.run_state_path.read_bytes()
    science = cold_state(paths)
    for _ in range(3):
        publish(paths, evidence, index)
        assert acknowledge(paths)
        assert paths.run_state_path.read_bytes() == authority
        assert cold_state(paths) == science
    request = json.loads((evidence / "request-0.json").read_bytes())
    activate_model(paths, request)
    assert paths.run_state_path.read_bytes() == authority


def test_only_sealed_model_evidence_is_required(
    workspace: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preserve chosen physical weights and descriptors; remove only an audited edge."""
    args, paths, evidence = workspace
    report = analyze(paths, retain=(1,))
    assert report["validated"], report
    assert 3 in report["conditional_candidates"]
    old = next(g for g in report["generations"] if g["generation"] == 3)
    capsule = io.capsule_path(paths.work_dir, old["id"], 3)
    saved = paths.work_dir.parent / "preserved"
    saved.mkdir()
    for name in ("checkpoint", "scientific_snapshot"):
        (capsule / name).rename(saved / name)
    before, science = paths.run_state_path.read_bytes(), cold_state(paths)
    seen = []
    original = gp.validate_capsule

    def validate(root: Path, record: dict[str, Any]) -> Any:
        seen.append(record["generation"])
        return original(root, record)

    monkeypatch.setattr(gp, "validate_capsule", validate)
    publish(paths, evidence)
    assert acknowledge(paths)
    assert seen == [7]
    assert paths.run_state_path.read_bytes() == before and cold_state(paths) == science
    activate_model(paths, json.loads((evidence / "request-0.json").read_bytes()))
    assert grow(args, paths).generation == 8
    assert not analyze(paths)["validated"]  # No retirement protocol has been enabled.


@pytest.mark.parametrize(
    "damage",
    [
        "model-missing",
        "model-corrupt",
        "descriptor-missing",
        "descriptor-corrupt",
        "current-model",
        "current-checkpoint",
        "current-science",
        "receipt-digest",
        "receipt-cursor",
        "duplicate-receipt",
        "conflicting-bytes",
    ],
)
def test_corruption_refuses_acknowledgement(workspace: Any, damage: str) -> None:
    """Neither an older surviving model copy nor an exact UUID bypasses integrity."""
    _, paths, evidence = workspace
    publish(paths, evidence)
    state = load_bootstrap_run_state(paths.run_state_path)
    records = io.publication_records(state.metadata)
    old, head = records[2], records[-1]
    if damage.startswith("receipt") or damage == "duplicate-receipt":
        raw = json.loads(paths.run_state_path.read_bytes())
        ledger = raw["metadata"][cr.RECEIPTS_METADATA_KEY]["receipts"]
        if damage == "duplicate-receipt":
            ledger.append(ledger[0])
        else:
            ledger[0][
                "content_sha256"
                if damage == "receipt-digest"
                else "accepted_cursor_sha256"
            ] = "0" * 64
        paths.run_state_path.write_text(json.dumps(raw))
    elif damage == "conflicting-bytes":
        paths.pipeline_reevaluation_patch_path.write_bytes(
            paths.pipeline_reevaluation_patch_path.read_bytes() + b" "
        )
    else:
        record = head if damage.startswith("current") else old
        capsule = io.capsule_path(paths.work_dir, record["id"], 3)
        descriptor = io.read_committed_descriptor(paths.work_dir, record)
        if damage.startswith("descriptor"):
            file = paths.work_dir / record["descriptor_path"]
        elif damage == "current-checkpoint":
            file = next((capsule / "checkpoint").rglob("*.zst"))
        elif damage == "current-science":
            file = capsule / "scientific_snapshot/updates.jsonl"
        else:
            pin = descriptor["model_pins"][0]
            file = next((capsule / pin["path"]).glob("*.pt"))
        if damage.endswith("missing"):
            file.rename(paths.work_dir.parent / "preserved-file")
        else:
            file.write_bytes(b"damaged independent synthetic copy")
    authority, pending = (
        paths.run_state_path.read_bytes(),
        paths.pipeline_reevaluation_patch_path.read_bytes(),
    )
    with pytest.raises((ValueError, FileNotFoundError)):
        acknowledge(paths)
    assert paths.run_state_path.read_bytes() == authority
    assert paths.pipeline_reevaluation_patch_path.read_bytes() == pending


def test_new_m3_admission_and_stale_new_m1_refusal(workspace: Any) -> None:
    """New work still uses full source/model/cursor checks, unlike receipt ACK."""
    args, paths, evidence = workspace
    assert propose(args).patch_written
    patch = reevaluation_patch_from_dict(
        json.loads(paths.pipeline_reevaluation_patch_path.read_bytes())
    )
    assert not acknowledge(paths)
    after = grow(args, paths)
    assert patch.patch_id in cr.receipt_ledger(after)
    old = json.loads((evidence / "patch-0.json").read_bytes())
    old["patch_id"] = "new-stale-M1"
    publish_reevaluation_patch_if_absent(
        reevaluation_patch_from_dict(old), paths.pipeline_reevaluation_patch_path
    )
    authority = paths.run_state_path.read_bytes()
    assert not acknowledge(paths)
    with pytest.raises(ValueError):
        grow(args, paths)
    assert paths.run_state_path.read_bytes() == authority


def test_missing_receipt_is_not_acknowledged_or_reapplied(workspace: Any) -> None:
    """An absent old receipt cannot prove commitment or permit stale new work."""
    args, paths, evidence = workspace
    raw = json.loads(paths.run_state_path.read_bytes())
    raw["metadata"][cr.RECEIPTS_METADATA_KEY]["receipts"].pop(0)
    paths.run_state_path.write_text(json.dumps(raw))
    authority = paths.run_state_path.read_bytes()
    publish(paths, evidence)
    assert not acknowledge(paths)
    with pytest.raises(ValueError):
        grow(args, paths)
    assert paths.run_state_path.read_bytes() == authority
    assert paths.pipeline_reevaluation_patch_path.exists()


def test_reader_and_explicit_roots_survive_edge_removal(workspace: Any) -> None:
    """Only the model lookup edge changes; an actual reader still protects G3."""
    _, paths, _ = workspace
    state = load_bootstrap_run_state(paths.run_state_path)
    record = io.publication_records(state.metadata)[2]
    with io.generation_read_pin(paths.work_dir / record["tree_snapshot_path"]):
        report = analyze(paths, retain=(1,))
        assert report["validated"] and 3 not in report["conditional_candidates"]
        item = next(g for g in report["generations"] if g["generation"] == 3)
        assert item["observed_reader"]
    report = analyze(paths, retain=(1, 3))
    assert report["validated"] and 3 not in report["conditional_candidates"]


def receipt_child(root: Path, phase: str) -> None:
    """Real death/unlink/fsync/lease seams, confined to a disposable child."""
    paths = MorpionBootstrapPaths.from_work_dir(root)
    original_pin = gp.DurableGenerationPublication.model_pin

    def pin(*args: Any, **kwargs: Any) -> Any:
        result = original_pin(*args, **kwargs)
        if phase == "model":
            os._exit(73)
        if phase == "barrier":
            print("owned", flush=True)
            input()
        return result

    gp.DurableGenerationPublication.model_pin = pin
    original_unlink = Path.unlink

    def unlink(path: Path, *args: Any, **kwargs: Any) -> None:
        if path == paths.pipeline_reevaluation_patch_path and phase == "before-unlink":
            os._exit(73)
        original_unlink(path, *args, **kwargs)
        if path == paths.pipeline_reevaluation_patch_path and phase == "after-unlink":
            os._exit(73)

    Path.unlink = unlink
    original_sync = cr.sync_directory

    def sync(path: Path) -> None:
        original_sync(path)
        if path == paths.pipeline_dir and phase == "after-fsync":
            os._exit(73)

    cr.sync_directory = sync
    assert acknowledge(paths)


def child_command(paths: MorpionBootstrapPaths, phase: str) -> list[str]:
    """Use the current Python and installed source path in another process."""
    return [
        sys.executable,
        "-c",
        "import sys; from pathlib import Path; from tests.environments.test_morpion_receipt_model_evidence import receipt_child; receipt_child(Path(sys.argv[1]),sys.argv[2])",
        str(paths.work_dir),
        phase,
    ]


@pytest.mark.parametrize(
    "phase", ["model", "before-unlink", "after-unlink", "after-fsync"]
)
def test_process_death_then_exact_retry(workspace: Any, phase: str) -> None:
    """Death leaves either the same pending proposal or its durable acknowledgement."""
    _, paths, evidence = workspace
    publish(paths, evidence)
    authority, science = paths.run_state_path.read_bytes(), cold_state(paths)
    result = subprocess.run(
        child_command(paths, phase), capture_output=True, text=True, timeout=45
    )
    assert result.returncode == 73, result.stderr
    assert paths.run_state_path.read_bytes() == authority
    if paths.pipeline_reevaluation_patch_path.exists():
        assert acknowledge(paths)
    publish(paths, evidence)
    assert acknowledge(paths)
    assert (
        paths.run_state_path.read_bytes() == authority and cold_state(paths) == science
    )


def test_concurrent_activation_uses_existing_growth_lease(
    workspace: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Activation cannot enter while old receipt acknowledgement owns Growth."""
    from tests.environments.test_morpion_patch_receipt_concurrency import _line

    args, paths, evidence = workspace
    local_leaderboard(monkeypatch, paths.work_dir.parent)
    run_dataset(args, generation=7)
    run_pipeline_training_stage(args, generation=7)
    request = prepare_model_activation_request(paths, training_generation=7)
    publish(paths, evidence)
    authority = paths.run_state_path.read_bytes()
    child = subprocess.Popen(
        child_command(paths, "barrier"),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert _line(child) == "owned"
        with pytest.raises(BlockingIOError):
            activate_model(paths, request)
        assert child.stdin is not None
        child.stdin.write("continue\n")
        child.stdin.flush()
        assert child.wait(timeout=45) == 0
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
    assert paths.run_state_path.read_bytes() == authority
    after = activate_model(paths, request)
    assert after.generation == 8
    assert io.committed_evaluator(paths.work_dir, after)[0] == request["candidate"]
