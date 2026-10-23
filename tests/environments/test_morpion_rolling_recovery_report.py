"""Read-only root analysis on newly generated closed scientific capsules only."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from chipiron.environments.morpion.bootstrap import (
    MorpionBootstrapPaths,
    load_bootstrap_run_state,
    save_bootstrap_run_state,
)
from chipiron.environments.morpion.bootstrap.checkpoint_patch_receipts import (
    RECEIPTS_METADATA_KEY,
    receipt_ledger,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    capsule_lease,
    capsule_path,
    file_digest,
    generation_read_pin,
    growth_receipt_ownership,
    publication_records,
)
from chipiron.environments.morpion.bootstrap.pipeline_claims import claim_pipeline_stage
from tests.environments.morpion_durable_closed_helpers import (
    ClosedPublicationRunner,
    publish_closed_patch,
    run_closed_publication,
    setup_closed_baseline,
)
from tests.environments.morpion_patch_receipt_helpers import MODEL

_SPEC = importlib.util.spec_from_file_location(
    "report_morpion_rolling_recovery",
    Path(__file__).resolve().parents[2] / "scripts/report_morpion_rolling_recovery.py",
)
assert _SPEC is not None and _SPEC.loader is not None
report_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(report_module)
analyze = report_module.analyze


@pytest.fixture(scope="module")
def history(tmp_path_factory: Any) -> Path:
    """Create eight actual closed capsules, without search or workers."""
    root = tmp_path_factory.mktemp("closed-history")
    args, paths, state = setup_closed_baseline(root)
    for generation in range(2, 9):
        publish_closed_patch(paths, f"arbitrary-uuid-{generation}")
        state = run_closed_publication(args, ClosedPublicationRunner())
    # Existing readers create their stable lock files. The analyzer never does.
    with generation_read_pin(root / state.latest_tree_snapshot_path):
        pass
    return root


@pytest.fixture
def workspace(history: Path, tmp_path: Path) -> MorpionBootstrapPaths:
    """Independent copies let corruption tests touch only disposable fixture data."""
    shutil.copytree(history, tmp_path / "work")
    return MorpionBootstrapPaths.from_work_dir(tmp_path / "work")


def records(paths: MorpionBootstrapPaths) -> list[dict[str, Any]]:
    """Read the authoritative committed records."""
    return publication_records(load_bootstrap_run_state(paths.run_state_path).metadata)


def fingerprint(root: Path) -> dict[str, Any]:
    """Include names, bytes, inode, mtime, ctime and hardlinks; exclude read atime."""
    return {
        str(p.relative_to(root)): (
            p.stat().st_ino,
            p.stat().st_mtime_ns,
            p.stat().st_ctime_ns,
            p.stat().st_nlink,
            file_digest(p) if p.is_file() else None,
        )
        for p in [root, *root.rglob("*")]
    }


def test_roots_read_only_and_receipts_separate(
    workspace: MorpionBootstrapPaths,
) -> None:
    """Three rolling roots do not discard the exact ledger or change current authority."""
    before = fingerprint(workspace.work_dir)
    result = analyze(workspace)
    assert result["validated"], result
    assert result["conditional_candidates"] == [1, 2, 3, 4, 5]
    assert result["receipt_count"] == 7
    assert result["conditional_reclaimable_file_blocks"] > 0
    assert result["current_contract_reclaimable_bytes"] == 0
    assert not result["deletion_authorized"]
    assert all(
        g["current_contract_required"] and not g["rollback_executable"]
        for g in result["generations"]
    )
    assert fingerprint(workspace.work_dir) == before


def test_reader_pin_and_explicit_retention(workspace: MorpionBootstrapPaths) -> None:
    """A real shared flock and an explicit scientific root both protect old capsules."""
    record = records(workspace)[1]
    with capsule_lease(workspace.work_dir, record["id"], version=3):
        result = analyze(workspace, retain=(4,))
        assert result["validated"]
        assert result["conditional_candidates"] == [1, 3, 5]
        assert result["generations"][1]["observed_reader"]
    assert analyze(workspace)["conditional_candidates"] == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("stage", ["dataset", "training"])
def test_expired_claim_is_not_reader_death(
    workspace: MorpionBootstrapPaths, stage: str
) -> None:
    """An ancient expired claim still retains the complete generation."""
    path = workspace.pipeline_generation_dir_for_generation(2) / f"{stage}_claim.json"
    claim_pipeline_stage(
        generation=2, stage=stage, claim_path=path, now_unix_s=1, ttl_seconds=1
    )
    result = analyze(workspace)
    assert result["validated"]
    assert 2 not in result["conditional_candidates"]
    assert (
        f"{stage}-claim-even-if-expired" in result["generations"][1]["rolling_reasons"]
    )


def test_pending_old_patch_and_exact_retry(workspace: MorpionBootstrapPaths) -> None:
    """An acknowledged retry keeps its old source and compares exact bytes."""
    # Recreate a source-G2 patch, then use its exact bytes as the recorded receipt
    # identity in this disposable fixture; no application/recovery is simulated.
    from tests.environments.morpion_patch_receipt_helpers import (
        fixture_patch,
        publish_fixture_patch,
    )

    record = records(workspace)[1]
    patch = replace(
        fixture_patch("arbitrary-uuid-3"),
        tree_generation=2,
        metadata={
            "scientific_publication": {
                "capsule_id": record["id"],
                "scientific_identity": record["scientific_identity"],
            }
        },
    )
    publish_fixture_patch(workspace, patch)
    raw = json.loads(workspace.run_state_path.read_bytes())
    receipt = raw["metadata"][RECEIPTS_METADATA_KEY]["receipts"][1]
    receipt["content_sha256"] = file_digest(workspace.pipeline_reevaluation_patch_path)
    workspace.run_state_path.write_text(json.dumps(raw))
    result = analyze(workspace)
    assert result["validated"], result
    assert 2 not in result["conditional_candidates"]
    assert "pending-patch" in result["generations"][1]["rolling_reasons"]
    receipt["evaluator_name"] = "conflicting-producer"
    workspace.run_state_path.write_text(json.dumps(raw))
    conflict = analyze(workspace)
    assert not conflict["validated"] and "Conflicting" in conflict["blockers"][0]
    receipt["evaluator_name"] = patch.evaluator_name
    workspace.run_state_path.write_text(json.dumps(raw))
    # Same UUID, different exact content: no normalization or watermark fallback.
    workspace.pipeline_reevaluation_patch_path.write_bytes(
        workspace.pipeline_reevaluation_patch_path.read_bytes() + b"\n"
    )
    result = analyze(workspace)
    assert not result["validated"] and "Conflicting" in result["blockers"][0]


@pytest.mark.parametrize("conflicting", [False, True])
def test_duplicate_receipt_id(
    workspace: MorpionBootstrapPaths, conflicting: bool
) -> None:
    """Both identical duplicated ledger IDs and conflicting duplicates fail closed."""
    raw = json.loads(workspace.run_state_path.read_bytes())
    receipts = raw["metadata"][RECEIPTS_METADATA_KEY]["receipts"]
    duplicate = dict(receipts[0])
    if conflicting:
        duplicate["content_sha256"] = "e" * 64
    receipts.append(duplicate)
    workspace.run_state_path.write_text(json.dumps(raw))
    result = analyze(workspace)
    assert not result["validated"] and "Duplicate receipt ID" in result["blockers"][0]
    assert result["conditional_reclaimable_file_blocks"] == 0


@pytest.mark.parametrize(
    "damage",
    [
        "missing-model",
        "corrupt-model",
        "missing-generation",
        "missing-pin",
        "unknown-control",
        "unknown-owner",
        "symlink",
    ],
)
def test_unknown_or_corrupt_dependencies_fail_closed(
    workspace: MorpionBootstrapPaths, damage: str
) -> None:
    """Neither absent evidence nor unknown ownership produces removal candidates."""
    record = records(workspace)[0]
    capsule = capsule_path(workspace.work_dir, record["id"], 3)
    if damage == "missing-model":
        (workspace.work_dir / MODEL / "param.pt").unlink()
    elif damage == "corrupt-model":
        (workspace.work_dir / MODEL / "param.pt").write_bytes(b"corrupt")
    elif damage == "missing-generation":
        shutil.rmtree(capsule)
    elif damage == "missing-pin":
        (capsule.parent.parent / "pins" / f"{record['id']}.lock").unlink()
    elif damage == "unknown-control":
        (workspace.pipeline_dir / "unknown.json").write_text("{}")
    elif damage == "unknown-owner":
        (capsule.parent / ("f" * 32)).mkdir()
    else:
        path = capsule / "bootstrap_config.json"
        path.unlink()
        path.symlink_to(workspace.bootstrap_config_path)
    before = fingerprint(workspace.work_dir)
    result = analyze(workspace)
    assert not result["validated"], result
    assert result["conditional_reclaimable_file_blocks"] == 0
    assert fingerprint(workspace.work_dir) == before


def test_external_hardlink_is_not_reclaimable(
    workspace: MorpionBootstrapPaths, tmp_path: Path
) -> None:
    """Unlocated link owners prevent issuing any conditional removal set."""
    before = analyze(workspace)
    record = records(workspace)[0]
    file = (
        capsule_path(workspace.work_dir, record["id"], 3)
        / "scientific_snapshot"
        / "updates.jsonl"
    )
    os.link(file, tmp_path / "external-reader-artifact")
    after = analyze(workspace)
    assert before["validated"]
    assert not after["validated"]
    assert after["conditional_reclaimable_file_blocks"] == 0
    assert "external hard-link ownership" in after["blockers"][0]


def test_growth_owner_refuses_analysis(workspace: MorpionBootstrapPaths) -> None:
    """A second independent lock descriptor cannot observe half a Growth commit."""
    with growth_receipt_ownership(workspace):
        result = analyze(workspace)
    assert not result["validated"] and "BlockingIOError" in result["blockers"][0]
    assert result["conditional_reclaimable_file_blocks"] == 0


def test_shared_candidate_and_kept_inode(workspace: MorpionBootstrapPaths) -> None:
    """A shared config inode survives because the current capsule needs it."""
    old, head = records(workspace)[0], records(workspace)[-1]
    source = capsule_path(workspace.work_dir, head["id"], 3) / "bootstrap_config.json"
    target = capsule_path(workspace.work_dir, old["id"], 3) / "bootstrap_config.json"
    assert source.read_bytes() == target.read_bytes()
    before = analyze(workspace)
    blocks = target.stat().st_blocks * 512
    target.unlink()
    os.link(source, target)
    result = analyze(workspace)
    assert result["validated"]
    assert (
        result["conditional_reclaimable_file_blocks"]
        == before["conditional_reclaimable_file_blocks"] - blocks
    )


def test_changed_control_fails_closed(
    workspace: MorpionBootstrapPaths, monkeypatch: Any
) -> None:
    """A producer change during inspection invalidates the observed control set."""
    original = report_module._controls
    calls = 0

    def changing(paths: MorpionBootstrapPaths) -> dict[str, str]:
        nonlocal calls
        calls += 1
        value = original(paths)
        if calls == 2:
            value["concurrent-producer"] = "different"
        return value

    monkeypatch.setattr(report_module, "_controls", changing)
    result = analyze(workspace)
    assert (
        not result["validated"] and "changed during analysis" in result["blockers"][0]
    )
    assert result["conditional_reclaimable_file_blocks"] == 0


def test_shared_candidate_inodes_count_once(workspace: MorpionBootstrapPaths) -> None:
    """Two candidate names free one allocation only when all links are candidates."""
    first, second = records(workspace)[:2]
    source = capsule_path(workspace.work_dir, first["id"], 3) / "bootstrap_config.json"
    target = capsule_path(workspace.work_dir, second["id"], 3) / "bootstrap_config.json"
    assert source.read_bytes() == target.read_bytes()
    before = analyze(workspace)
    blocks = target.stat().st_blocks * 512
    target.unlink()
    os.link(source, target)
    result = analyze(workspace)
    assert result["validated"]
    assert (
        result["conditional_reclaimable_file_blocks"]
        == before["conditional_reclaimable_file_blocks"] - blocks
    )
    assert (
        sum(i["blocks"] for i in result["conditional_reclaimable_files"])
        == result["conditional_reclaimable_file_blocks"]
    )


def test_unknown_claim_metadata(workspace: MorpionBootstrapPaths) -> None:
    """Do not guess the ownership meaning of an unrecognized worker extension."""
    claim_pipeline_stage(
        generation=2,
        stage="dataset",
        claim_path=workspace.pipeline_dataset_claim_path_for_generation(2),
        metadata={"another_generation_dependency": 1},
    )
    result = analyze(workspace)
    assert (
        not result["validated"]
        and "Unknown claim dependencies" in result["blockers"][0]
    )


def test_intentional_rollback_is_not_crash_recovery(
    workspace: MorpionBootstrapPaths,
) -> None:
    """Retaining newer acknowledgement receipts with an older head is invalid."""
    state = load_bootstrap_run_state(workspace.run_state_path)
    old = records(workspace)[-2]
    save_bootstrap_run_state(
        replace(
            state,
            generation=old["generation"],
            latest_runtime_checkpoint_path=old["checkpoint_path"],
            latest_tree_snapshot_path=old["tree_snapshot_path"],
        ),
        workspace.run_state_path,
        durable=True,
    )
    result = analyze(workspace)
    assert not result["validated"] and "Head mismatch" in result["blockers"][0]


def test_truncated_v3_authority_is_not_a_rolling_contract(
    workspace: MorpionBootstrapPaths,
) -> None:
    """An absent genesis cannot be disguised by shortening the old authority list."""
    raw = json.loads(workspace.run_state_path.read_bytes())
    roots = raw["metadata"]["durable_generation_publication_v1"]["generations"]
    old = roots.pop(0)
    roots[0]["parent_id"] = None
    shutil.rmtree(capsule_path(workspace.work_dir, old["id"], 3))
    workspace.run_state_path.write_text(json.dumps(raw))
    result = analyze(workspace)
    assert not result["validated"] and "genesis 1" in result["blockers"][0]


@pytest.mark.parametrize("boundary", ["before_state", "after_state"])
def test_crash_authority_and_exact_recovery(tmp_path: Path, boundary: str) -> None:
    """The analyzer never adopts an orphan, applies a patch, or acknowledges one."""
    from tests.environments.morpion_durable_v2_helpers import publish_scientific_patch
    from tests.environments.test_morpion_durable_closed_crashes import run_crash

    args, paths, _ = setup_closed_baseline(tmp_path)
    publish_scientific_patch(paths)
    pending_bytes = paths.pipeline_reevaluation_patch_path.read_bytes()
    run_crash(tmp_path, boundary)
    observed = load_bootstrap_run_state(paths.run_state_path)
    assert observed.generation == (1 if boundary == "before_state" else 2)
    before = fingerprint(tmp_path)
    result = analyze(paths)
    assert result["current_contract_reclaimable_bytes"] == 0
    assert fingerprint(tmp_path) == before
    runner = ClosedPublicationRunner()
    runner.grow_delta = int(boundary == "before_state")
    recovered = run_closed_publication(
        args, runner, max_cycles=int(boundary == "before_state")
    )
    assert recovered.generation == 2 and len(receipt_ledger(recovered)) == 1
    assert runner.apply_calls == int(boundary == "before_state")
    checkpoint = (tmp_path / recovered.latest_runtime_checkpoint_path).read_bytes()
    # Exact post-commit retry: producer cursor remains and only acknowledgement runs.
    paths.pipeline_reevaluation_patch_path.write_bytes(pending_bytes)
    retry = ClosedPublicationRunner()
    assert run_closed_publication(args, retry, max_cycles=0) == recovered
    assert retry.apply_calls == 0
    assert (
        tmp_path / recovered.latest_runtime_checkpoint_path
    ).read_bytes() == checkpoint
