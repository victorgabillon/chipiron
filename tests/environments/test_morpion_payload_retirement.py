"""Explicit retirement changes absence admission, never the scientific authority."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from anemone.checkpoints import (
    build_search_checkpoint_payload,
    checkpoint_payload_to_jsonable,
)

from chipiron.environments.morpion.bootstrap import (
    MorpionBootstrapPaths,
    run_pipeline_growth_stage,
)
from chipiron.environments.morpion.bootstrap.checkpoint_patch_receipts import (
    GrowthPatchReceiptSession,
    receipt_ledger,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    PUBLICATION_KEY,
    RETIRED_PAYLOAD_KEY,
    audit_generation_publications,
    capsule_lease,
    capsule_path,
    capsule_payload_files,
    committed_evaluator,
    generation_read_pin,
    growth_receipt_ownership,
    publication_records,
    validate_capsule,
)
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
from chipiron.environments.morpion.bootstrap.pipeline_claims import claim_pipeline_stage
from chipiron.environments.morpion.bootstrap.pipeline_orchestrator import (
    load_available_pipeline_manifests,
)
from chipiron.environments.morpion.bootstrap.reevaluation_transaction import (
    TRANSACTION_KEY,
    validate_proposal,
)
from chipiron.environments.morpion.bootstrap.reevaluation_worker import (
    run_morpion_reevaluation_worker_once,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from tests.environments.morpion_activation_helpers import setup_activation
from tests.environments.morpion_fenced_dataset_helpers import (
    local_leaderboard,
    run_dataset,
)
from tests.environments.test_morpion_rolling_recovery_report import report_module


def grow(paths: MorpionBootstrapPaths) -> Any:
    """One actual cold Growth invocation, using the committed frozen configuration."""
    from chipiron.environments.morpion.bootstrap.config import load_bootstrap_config
    from chipiron.environments.morpion.bootstrap.launcher import _scientific_args
    from tests.environments.morpion_durable_v2_helpers import scientific_args

    state = load_bootstrap_run_state(paths.run_state_path)
    runner, _ = _runner(paths, state)
    args = _scientific_args(
        scientific_args(paths.work_dir),
        load_bootstrap_config(paths.bootstrap_config_path),
    )
    producer = publication_records(state.metadata)[-1]["scientific_identity"][
        "producer"
    ]["identity"]
    return run_pipeline_growth_stage(
        args,
        runner,
        max_cycles=1,
        checkpoint_bound_receipts=True,
        durable_generation_publication=True,
        scientific_v2_producer=producer,
        closed_scientific_snapshot=True,
    )


def payload(runner: Any) -> Any:
    """Use Anemone's complete existing checkpoint serializer as the oracle."""
    return checkpoint_payload_to_jsonable(
        build_search_checkpoint_payload(
            runner._runtime, state_codec=runner._state_codec
        )
    )


def cold(paths: MorpionBootstrapPaths) -> Any:
    """Complete runtime payload includes DAG order, selector, search and rollout RNG."""
    state = load_bootstrap_run_state(paths.run_state_path)
    runner, effective = _runner(paths, state)
    _, model = committed_evaluator(paths.work_dir, state)
    try:
        runner.load_or_create(
            paths.work_dir / state.latest_runtime_checkpoint_path, model, effective
        )
        runner.validate_model_refresh_certificates()
        return payload(runner)
    finally:
        runner.discard_uncommitted_runtime()


@pytest.fixture(scope="module")
def history(tmp_path_factory: Any) -> Path:
    """Two genuine positive-sample activations, real M1/M2 receipts and M3 pending work."""
    root = tmp_path_factory.mktemp("retirement-real-history") / "workspace"
    with pytest.MonkeyPatch.context() as mp:
        local_leaderboard(mp, root)
        args, paths, _ = setup_activation(
            root, count=60, encoding="jsonl-zstd-1", seed_score=42.0
        )
        for label, source in [("M1", 1), ("M2", 4)]:
            if source == 4:
                run_dataset(args, generation=source)
                run_pipeline_training_stage(args, generation=source)
            result = run_morpion_reevaluation_worker_once(
                args,
                max_nodes_per_patch=7,
                use_streaming_sharded_reader=True,
                allow_scientific_v2=True,
            )
            assert result.patch_written
            (root / f"{label}-patch.json").write_bytes(
                paths.pipeline_reevaluation_patch_path.read_bytes()
            )
            state = grow(paths)
            assert state.generation == source + 1
            request = prepare_model_activation_request(
                paths, training_generation=source
            )
            (root / f"{label}-activation.json").write_text(json.dumps(request))
            state = activate_model(paths, request)
            assert state.generation == source + 2
            grow(paths)
        grow(paths)
        grow(paths)
        state = load_bootstrap_run_state(paths.run_state_path)
        assert state.generation == 9
        assert len(receipt_ledger(state)) == 2
        assert run_morpion_reevaluation_worker_once(
            args,
            max_nodes_per_patch=7,
            use_streaming_sharded_reader=True,
            allow_scientific_v2=True,
        ).patch_written
        audit_generation_publications(root)
    return root


@pytest.fixture
def workspace(history: Path, tmp_path: Path) -> MorpionBootstrapPaths:
    """Every unlink and corruption is confined to this independently copied fixture."""
    return MorpionBootstrapPaths.from_work_dir(
        shutil.copytree(history, tmp_path / "copy")
    )


def records(paths: MorpionBootstrapPaths) -> list[dict[str, Any]]:
    """Read only the existing committed publication records."""
    return publication_records(load_bootstrap_run_state(paths.run_state_path).metadata)


def receipt_retry(paths: MorpionBootstrapPaths, label: str) -> None:
    """Acknowledge exact captured bytes without rewinding model, cursor or authority."""
    pending = paths.pipeline_reevaluation_patch_path
    original = pending.read_bytes()
    pending.write_bytes((paths.work_dir / f"{label}-patch.json").read_bytes())
    state = load_bootstrap_run_state(paths.run_state_path)
    producer = records(paths)[-1]["scientific_identity"]["producer"]["identity"]
    session = GrowthPatchReceiptSession(
        paths,
        object(),
        durable_generation_publication=True,
        scientific_v2_producer=producer,
        closed_scientific_snapshot=True,
    )
    before = paths.run_state_path.read_bytes()
    with growth_receipt_ownership(paths):
        assert session.acknowledge_pending(state)
    assert paths.run_state_path.read_bytes() == before
    assert not pending.exists()
    pending.write_bytes(original)


def test_real_receipts_activation_and_exact_continuation_after_retirement(
    workspace: MorpionBootstrapPaths, tmp_path: Path
) -> None:
    """Retire receipt checkpoints/activation trees while preserving all exact evidence."""
    paths = workspace
    baseline = MorpionBootstrapPaths.from_work_dir(
        shutil.copytree(paths.work_dir, tmp_path / "baseline")
    )
    before = cold(paths)
    authority = json.loads(paths.run_state_path.read_bytes())
    plan = report_module.analyze(paths, retirement_plan=True)
    assert plan["validated"], plan
    assert plan["conditional_candidates"] == [2, 3, 5, 6]
    outcome = report_module.retire(paths, generations=(2, 3, 5, 6))
    assert outcome["removed_files"] and outcome["deletion_authorized"]
    after = json.loads(paths.run_state_path.read_bytes())
    for r in after["metadata"][PUBLICATION_KEY]["generations"]:
        r.pop(RETIRED_PAYLOAD_KEY, None)
    assert after == authority
    assert cold(paths) == before
    audit_generation_publications(paths.work_dir)
    available = load_available_pipeline_manifests(paths)
    assert set(available) == {1, 4, 7, 8, 9}
    for label in ["M1", "M2"]:
        receipt_retry(paths, label)
        state = load_bootstrap_run_state(paths.run_state_path)
        assert (
            activate_model(
                paths,
                json.loads((paths.work_dir / f"{label}-activation.json").read_text()),
            )
            == state
        )
    # Pending current-model inference is exactly preserved and actually consumed.
    assert (
        paths.pipeline_reevaluation_patch_path.read_bytes()
        == baseline.pipeline_reevaluation_patch_path.read_bytes()
    )
    grow(baseline)
    grow(paths)
    assert cold(paths) == cold(baseline)
    for left, right in zip(records(paths)[-1:], records(baseline)[-1:], strict=True):
        assert (
            left["scientific_identity"]["checkpoint_sha256"]
            == right["scientific_identity"]["checkpoint_sha256"]
        )
        for name in ["generation.json", "records.jsonl.zst", "updates.jsonl.zst"]:
            a = (paths.work_dir / left["tree_snapshot_path"]).parent / name
            b = (baseline.work_dir / right["tree_snapshot_path"]).parent / name
            assert a.read_bytes() == b.read_bytes()
    assert not report_module.retire(paths, generations=(2, 3, 5, 6))["removed_files"]


@pytest.mark.parametrize(
    "protection", ["reader", "explicit", "claim", "accepted", "recent"]
)
def test_required_sources_cannot_be_authorized(
    workspace: MorpionBootstrapPaths, protection: str
) -> None:
    """Neither expiry nor process absence nor a publication marker removes a real root."""
    from contextlib import ExitStack

    paths = workspace
    generation = 3
    retain = ()
    with ExitStack() as stack:
        if protection == "reader":
            stack.enter_context(
                capsule_lease(paths.work_dir, records(paths)[2]["id"], version=3)
            )
        elif protection == "explicit":
            retain = (generation,)
        elif protection == "claim":
            claim_pipeline_stage(
                generation=generation,
                stage="dataset",
                claim_path=paths.pipeline_dataset_claim_path_for_generation(generation),
                ttl_seconds=1,
                now_unix_s=1,
            )
        elif protection == "accepted":
            generation = 1
        else:
            generation = 8
        before = paths.run_state_path.read_bytes()
        with pytest.raises(ValueError):
            report_module.retire(paths, generations=(generation,), retain=retain)
        assert paths.run_state_path.read_bytes() == before
        audit_generation_publications(paths.work_dir)


@pytest.mark.parametrize(
    "damage",
    ["unmarked-missing", "marked-model", "current", "marker-digest", "recent-marker"],
)
def test_retirement_does_not_authorize_other_missing_or_corrupt_bytes(
    workspace: MorpionBootstrapPaths, damage: str
) -> None:
    """Missing unmarked payloads and damaged models/current state still fail closed."""
    paths = workspace
    report_module.retire(paths, generations=(2,))
    state = load_bootstrap_run_state(paths.run_state_path)
    rows = records(paths)
    if damage == "unmarked-missing":
        (paths.work_dir / rows[2]["tree_snapshot_path"]).unlink()
    elif damage == "marked-model":
        descriptor = validate_capsule(paths.work_dir, rows[1], retired_evidence=True)
        root = capsule_path(paths.work_dir, rows[1]["id"], 3)
        file = next((root / descriptor["model_pins"][0]["path"]).glob("param.pt"))
        file.write_bytes(b"corrupt")
    elif damage == "current":
        (paths.work_dir / rows[-1]["tree_snapshot_path"]).unlink()
    else:
        raw = json.loads(paths.run_state_path.read_bytes())
        record = raw["metadata"][PUBLICATION_KEY]["generations"][
            1 if damage == "marker-digest" else -1
        ]
        record[RETIRED_PAYLOAD_KEY] = (
            "f" * 64 if damage == "marker-digest" else record["descriptor_sha256"]
        )
        paths.run_state_path.write_text(json.dumps(raw))
    with pytest.raises((ValueError, OSError)):
        audit_generation_publications(paths.work_dir)
    assert not report_module.analyze(paths)["validated"]
    # Reusing the committed old source is refused even if its bytes remain after authorization.
    if damage == "marker-digest":
        assert state.generation == 9


def test_old_uncommitted_proposal_is_stale_not_an_exact_receipt(
    workspace: MorpionBootstrapPaths,
) -> None:
    """A new stale operation cannot borrow an old exact receipt identity."""
    paths = workspace
    report_module.retire(paths, generations=(2, 3, 5, 6))
    patch = json.loads((paths.work_dir / "M1-patch.json").read_text())
    patch["patch_id"] = "new-stale-operation"
    patch["metadata"][TRANSACTION_KEY]["next_cursor"]["last_patch_id"] = patch[
        "patch_id"
    ]
    with pytest.raises(ValueError, match="Stale expected accepted cursor"):
        validate_proposal(
            load_bootstrap_run_state(paths.run_state_path),
            reevaluation_patch_from_dict(patch),
        )


def crash_retirement(root: Path, boundary: str) -> None:
    """Real SIGKILL seams: pre-replace, replaced/not-dir-synced, authorized and every unlink."""
    import chipiron.environments.morpion.bootstrap.derived.provenance as provenance
    from scripts import report_morpion_rolling_recovery as module

    def die() -> None:
        os.kill(os.getpid(), signal.SIGKILL)

    original_save = module.save_bootstrap_run_state
    original_replace = provenance.os.replace

    def replace_authority(source: Any, destination: Any) -> None:
        if Path(destination).name == "run_state.json":
            if boundary == "before-replace":
                die()
            original_replace(source, destination)
            if boundary == "after-replace":
                die()
        else:
            original_replace(source, destination)

    provenance.os.replace = replace_authority

    def save(*args: Any, **kwargs: Any) -> None:
        original_save(*args, **kwargs)
        if boundary == "after-authorization":
            die()

    module.save_bootstrap_run_state = save
    original_unlink = Path.unlink
    counter = 0

    def unlink(path: Path, *args: Any, **kwargs: Any) -> None:
        nonlocal counter
        if "/objects/" in str(path):
            counter += 1
            if boundary == f"before-unlink-{counter}":
                die()
            original_unlink(path, *args, **kwargs)
            if boundary == f"after-unlink-{counter}":
                die()
        else:
            original_unlink(path, *args, **kwargs)

    Path.unlink = unlink
    module.retire(MorpionBootstrapPaths.from_work_dir(root), generations=(2,))


# The generated one-shard checkpoint has seven files plus three scientific files.
@pytest.mark.parametrize(
    "boundary",
    [
        "before-replace",
        "after-replace",
        "after-authorization",
        *[f"{when}-unlink-{n}" for n in range(1, 11) for when in ["before", "after"]],
    ],
)
def test_process_death_across_authorization_and_every_payload_unlink(
    workspace: MorpionBootstrapPaths, boundary: str
) -> None:
    """Every process-death boundary preserves exact current recovery and safe retry."""
    paths = workspace
    before = cold(paths)
    record = records(paths)[1]
    descriptor = validate_capsule(paths.work_dir, record)
    assert len(capsule_payload_files(descriptor)) == 10
    code = "from pathlib import Path; from tests.environments.test_morpion_payload_retirement import crash_retirement; import sys; crash_retirement(Path(sys.argv[1]),sys.argv[2])"
    result = subprocess.run(
        [sys.executable, "-c", code, str(paths.work_dir), boundary],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == -signal.SIGKILL, (
        result.stdout[-2000:] + result.stderr[-2000:]
    )
    assert cold(paths) == before
    audit_generation_publications(paths.work_dir)
    if boundary == "before-replace":
        assert RETIRED_PAYLOAD_KEY not in records(paths)[1]
    else:
        assert records(paths)[1][RETIRED_PAYLOAD_KEY] == record["descriptor_sha256"]
        with (
            pytest.raises(ValueError, match="Retired payload"),
            generation_read_pin(paths.work_dir / record["descriptor_path"]),
        ):
            pass
    report_module.retire(paths, generations=(2,))
    assert cold(paths) == before
    assert report_module.analyze(paths)["validated"]


def test_all_three_recent_checkpoints_restore_and_accepted_inputs_stay_verified(
    workspace: MorpionBootstrapPaths,
) -> None:
    """Audit complete prior search boundaries; ordinary recovery still selects head."""
    from chipiron.environments.morpion.bootstrap import (
        dataset_publication,
        training_publication,
    )
    from chipiron.environments.morpion.bootstrap.generation_publication_io import (
        artifact_digest,
    )

    paths = workspace
    current = load_bootstrap_run_state(paths.run_state_path)
    _, model = committed_evaluator(paths.work_dir, current)
    kept = records(paths)[-3:]

    def offline_payload(record: dict[str, Any]) -> Any:
        runner, _ = _runner(paths, current)
        try:
            runner._runtime = runner._load_runtime_from_checkpoint(
                paths.work_dir / record["checkpoint_path"],
                search_args=runner._args.search_args,
                model_bundle_path=model,
            )
            runner.validate_model_refresh_certificates()
            return payload(runner)
        finally:
            runner.discard_uncommitted_runtime()

    expected = [
        (artifact_digest(paths.work_dir / r["checkpoint_path"]), offline_payload(r))
        for r in kept
    ]
    accepted = {
        g: (
            dataset_publication.accepted_dataset_identity(paths, current, g),
            training_publication.accepted_selected_model(paths, current, g),
        )
        for g in [1, 4]
    }
    report_module.retire(paths, generations=(2, 3, 5, 6))
    for r, (checkpoint_digest, expected_payload) in zip(kept, expected, strict=True):
        descriptor = validate_capsule(paths.work_dir, r)
        assert (
            descriptor["active_evaluator"]
            == committed_evaluator(paths.work_dir, current)[0]
        )
        assert (
            artifact_digest(paths.work_dir / r["checkpoint_path"]) == checkpoint_digest
        )
        assert offline_payload(r) == expected_payload
    for g in [1, 4]:
        assert accepted[g] == (
            dataset_publication.accepted_dataset_identity(paths, current, g),
            training_publication.accepted_selected_model(paths, current, g),
        )
    runner, effective = _runner(paths, current)
    try:
        with pytest.raises(ValueError, match="Only the committed head"):
            runner.load_or_create(
                paths.work_dir / kept[0]["checkpoint_path"], model, effective
            )
    finally:
        runner.discard_uncommitted_runtime()
