"""Complete bounded managed transaction/entry matrix, prepared for installed validation."""

import json
import subprocess
import sys
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest

from chipiron.environments.morpion.bootstrap import managed_sessions as managed
from chipiron.environments.morpion.bootstrap.derived.provenance import atomic_json
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    growth_receipt_ownership,
)
from tests.environments.test_morpion_managed_sessions import controlled_close, snapshot
from tests.environments.test_morpion_managed_sessions import fresh as fresh

if TYPE_CHECKING:
    from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
        MorpionBootstrapPaths,
    )


@pytest.mark.exhaustive
@pytest.mark.parametrize("boundary", ["before-object", "after-object", "after-root"])
def test_grant_publication_crash_boundaries(
    fresh: MorpionBootstrapPaths, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    """An orphan has no authority; a committed grant retries once without new credit."""
    controlled_close(fresh)
    config, state = snapshot(fresh)
    request, _ = managed.plan_additional_time(
        fresh, managed.inspect(fresh, config, state), 4, "durable-request"
    )
    original_put, original_commit = managed._put, managed._commit

    def fail_put(*args: Any, **kwargs: Any) -> Any:
        message = "before object"
        raise RuntimeError(message)

    def fail_commit(*args: Any, **kwargs: Any) -> Any:
        if boundary == "after-root":
            original_commit(*args, **kwargs)
        message = "controlled root boundary"
        raise RuntimeError(message)

    with monkeypatch.context() as patch:
        if boundary == "before-object":
            patch.setattr(managed, "_put", fail_put)
        else:
            patch.setattr(managed, "_commit", fail_commit)
        with growth_receipt_ownership(fresh, create=False), pytest.raises(RuntimeError):
            managed.admit_additional_time_owned(fresh, config, state, request)
    config, state = snapshot(fresh)
    assert len(state.metadata[managed.KEY]["grants"]) == (
        1 if boundary == "after-root" else 0
    )
    with growth_receipt_ownership(fresh, create=False):
        updated, _ = managed.admit_additional_time_owned(fresh, config, state, request)
    assert (
        len(updated.metadata[managed.KEY]["grants"]) == 1
        and updated.metadata[managed.KEY]["authorized_seconds"] == 6
    )
    assert original_put is managed._put


@pytest.mark.exhaustive
@pytest.mark.parametrize("boundary", ["before-object", "after-object", "after-root"])
def test_launch_publication_crash_boundaries(
    fresh: MorpionBootstrapPaths, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    """A committed launch is uncertain even when the coordinator died before spawn."""
    config, state = snapshot(fresh)
    commit = managed._commit

    def fail_put(*args: Any, **kwargs: Any) -> Any:
        message = "before launch object"
        raise RuntimeError(message)

    def fail_commit(*args: Any, **kwargs: Any) -> Any:
        if boundary == "after-root":
            commit(*args, **kwargs)
        message = "launch root boundary"
        raise RuntimeError(message)

    with monkeypatch.context() as patch:
        patch.setattr(
            managed,
            "_put" if boundary == "before-object" else "_commit",
            fail_put if boundary == "before-object" else fail_commit,
        )
        with pytest.raises(RuntimeError):
            managed.begin_owned(fresh, config, "124", time.monotonic(), 1)
    config, state = snapshot(fresh)
    assert (state.metadata[managed.KEY]["active"] is not None) == (
        boundary == "after-root"
    )
    if boundary == "after-root":
        with pytest.raises(ValueError, match="validated close"):
            managed.recover_close_owned(fresh, config)
    else:
        assert managed.inspect(fresh, config, state)["status"] == "ready"


@pytest.mark.exhaustive
@pytest.mark.parametrize(
    "writer",
    [
        "growth",
        "dataset",
        "training",
        "reevaluation",
        "dataset-accept",
        "training-accept",
        "activation",
        "persistent",
        "profile",
    ],
)
def test_all_public_writers_refuse_unmonitored_managed_work(
    fresh: MorpionBootstrapPaths, writer: str
) -> None:
    """No alternate direct API may write before an active native ancestry check."""
    from chipiron.environments.morpion.bootstrap import (
        dataset_publication,
        model_activation,
        persistent_worker,
        reevaluation_worker,
        training_publication,
    )
    from chipiron.environments.morpion.bootstrap.bootstrap_args import (
        MorpionBootstrapArgs,
    )
    from chipiron.environments.morpion.bootstrap.execution_profile import (
        require_args_profile,
    )
    from chipiron.environments.morpion.bootstrap.launcher import launcher_args_from_cli
    from chipiron.environments.morpion.bootstrap.pipeline import stages

    args = MorpionBootstrapArgs(
        work_dir=fresh.work_dir,
        training_device="cpu",
        pipeline_mode="artifact_pipeline",
    )
    calls = {
        "growth": lambda: stages.run_pipeline_growth_stage(args, None),
        "dataset": lambda: stages.run_pipeline_dataset_stage(args, generation=1),
        "training": lambda: stages.run_pipeline_training_stage(args, generation=1),
        "reevaluation": lambda: (
            reevaluation_worker.run_morpion_reevaluation_worker_once(args)
        ),
        "dataset-accept": lambda: dataset_publication.accept_dataset(fresh, {}),
        "training-accept": lambda: training_publication.accept_training(fresh, {}),
        "activation": lambda: model_activation.activate_model(fresh, {}),
        "persistent": lambda: persistent_worker.run_persistent_worker(
            launcher_args_from_cli([
                "--scientific",
                "--pipeline-mode",
                "artifact_pipeline",
                "--pipeline-stage",
                "training_worker",
                "--work-dir",
                str(fresh.work_dir),
            ])
        ),
        "profile": lambda: require_args_profile(args, stage="training"),
    }
    before = {str(p): p.read_bytes() for p in fresh.work_dir.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match=r"Managed|managed"):
        calls[writer]()
    assert {
        str(p): p.read_bytes() for p in fresh.work_dir.rglob("*") if p.is_file()
    } == before


@pytest.mark.exhaustive
@pytest.mark.parametrize(
    "damage",
    [
        "worker-failure",
        "control-failure",
        "forced",
        "grace",
        "bad-exit",
        "future-decision",
        "backward-decision",
        "mismatched-launch",
        "missing-usage",
    ],
)
def test_bad_native_evidence_cannot_renew(
    fresh: MorpionBootstrapPaths, damage: str
) -> None:
    """Hash changes and endpoint/cleanup inconsistency never mint extra authorization."""
    controlled_close(fresh)
    config, state = snapshot(fresh)
    monitor = fresh.work_dir / "resource_monitor"
    if damage == "missing-usage":
        (monitor / "usage.json").unlink()
    else:
        path = monitor / "session_123.json"
        receipt = json.loads(path.read_text())
        decision = receipt["decision"]
        if damage == "worker-failure":
            decision["worker_failures"] = [[1, "persistent_worker_failed"]]
        elif damage == "control-failure":
            decision["control_failure"] = "invalid_supervisor_shutdown"
        elif damage == "forced":
            receipt["forced_group_cleanup"] = True
        elif damage == "grace":
            receipt["shutdown_grace_expired"] = True
        elif damage == "bad-exit":
            receipt["supervisor_exit_code"] = 1
        elif damage == "future-decision":
            decision["monotonic_s"] = time.monotonic() + 10000
        elif damage == "backward-decision":
            decision["budget_elapsed_seconds"] = -1
        else:
            decision["managed_session"] = {"other": "launch"}
        atomic_json(path, receipt)
    before = fresh.run_state_path.read_bytes()
    with pytest.raises((ValueError, OSError)):
        managed.plan_additional_time(fresh, managed.inspect(fresh, config, state), 4)
    assert fresh.run_state_path.read_bytes() == before


@pytest.mark.exhaustive
def test_concurrent_request_owner_and_predecessor_are_fenced(
    fresh: MorpionBootstrapPaths,
) -> None:
    """A real second process cannot acquire the held publication owner."""
    controlled_close(fresh)
    config, state = snapshot(fresh)
    request, _ = managed.plan_additional_time(
        fresh, managed.inspect(fresh, config, state), 4, "client"
    )
    code = """from pathlib import Path
import sys
from chipiron.environments.morpion.bootstrap.bootstrap_paths import MorpionBootstrapPaths
from chipiron.environments.morpion.bootstrap.generation_publication_io import growth_receipt_ownership
try:
 with growth_receipt_ownership(MorpionBootstrapPaths.from_work_dir(Path(sys.argv[1])),create=False):sys.exit(8)
except BlockingIOError:sys.exit(0)
"""
    before = fresh.run_state_path.read_bytes()
    with growth_receipt_ownership(fresh, create=False):
        child = subprocess.run(
            [sys.executable, "-B", "-c", code, str(fresh.work_dir)],
            capture_output=True,
            timeout=15,
        )
        assert child.returncode == 0
    assert before == fresh.run_state_path.read_bytes()
    stale = replace(request, predecessor={"id": "close-999", "sha256": "0" * 64})
    with (
        growth_receipt_ownership(fresh, create=False),
        pytest.raises(ValueError, match="predecessor"),
    ):
        managed.admit_additional_time_owned(fresh, config, state, stale)


@pytest.mark.exhaustive
def test_real_concurrent_admissions_commit_one_grant(
    fresh: MorpionBootstrapPaths,
) -> None:
    """Two independently owned processes race the same ID under actual kernel locks."""
    controlled_close(fresh)
    code = """from pathlib import Path
import sys,fcntl
from chipiron.environments.morpion.bootstrap import managed_sessions as m
from chipiron.environments.morpion.bootstrap.bootstrap_paths import MorpionBootstrapPaths
from chipiron.environments.morpion.bootstrap.config import load_bootstrap_config
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from chipiron.environments.morpion.bootstrap.generation_publication_io import growth_receipt_ownership
p=MorpionBootstrapPaths.from_work_dir(Path(sys.argv[1]));sys.stdin.read(1)
try:
 with (p.work_dir/'resource_monitor/owner.lock').open('a') as owner:
  fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
  with growth_receipt_ownership(p,create=False):
   c=load_bootstrap_config(p.bootstrap_config_path);s=load_bootstrap_run_state(p.run_state_path)
   r,_=m.plan_additional_time(p,m.inspect(p,c,s),4,'same-client')
   m.admit_additional_time_owned(p,c,s,r)
except BlockingIOError:sys.exit(75)
"""
    children = [
        subprocess.Popen(
            [sys.executable, "-B", "-c", code, str(fresh.work_dir)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    try:
        for child in children:
            assert child.stdin is not None
            child.stdin.write("x")
            child.stdin.flush()
        outcomes = [child.communicate(timeout=15) for child in children]
        assert any(child.returncode == 0 for child in children), outcomes
        assert all(child.returncode in (0, 75) for child in children), outcomes
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
    _config, state = snapshot(fresh)
    assert (
        len(state.metadata[managed.KEY]["grants"]) == 1
        and state.metadata[managed.KEY]["authorized_seconds"] == 6
    )


@pytest.mark.exhaustive
@pytest.mark.parametrize(
    "damage",
    ["bool-schema", "grant-overflow", "session-overflow", "scope", "staged-model"],
)
def test_metadata_and_identity_bounds_refuse_before_authorization(
    fresh: MorpionBootstrapPaths, damage: str
) -> None:
    """Small controlled boundary values cannot enlarge the accepted protocol."""
    config, state = snapshot(fresh)
    data = state.metadata[managed.KEY]
    if damage == "bool-schema":
        data = {**data, "schema_version": True}
    elif damage == "grant-overflow":
        data = {**data, "grants": [{}] * (managed.MAX_GRANTS + 1)}
    elif damage == "session-overflow":
        data = {**data, "sessions": [{}] * (managed.MAX_SESSIONS + 1)}
    elif damage == "scope":
        identity = managed._local_identity()
        identity["boot"] = "unknown"
        with pytest.raises(ValueError, match="host/boot"):
            managed._same_scope(identity)
        return
    else:
        # No accepted candidate parent exists at genesis; a grant must still refuse
        # rather than manufacture a model/optimizer lineage from a staged directory.
        folder = fresh.work_dir / "training_publications_v1/staging/unaccepted"
        folder.mkdir(parents=True)
        (folder / "param.pt").write_bytes(b"unaccepted")
        with pytest.raises(ValueError):
            managed.plan_additional_time(
                fresh, managed.inspect(fresh, config, state), 4
            )
        return
    before = fresh.run_state_path.read_bytes()
    with pytest.raises((ValueError, TypeError)):
        managed.validate(
            fresh,
            config,
            replace(state, metadata={**state.metadata, managed.KEY: data}),
        )
    assert fresh.run_state_path.read_bytes() == before


@pytest.mark.exhaustive
def test_direct_native_grant_refuses_fresh_before_monitor_writes(
    fresh: MorpionBootstrapPaths,
) -> None:
    """Direct native admission refuses before creating monitor metadata."""
    from chipiron.environments.morpion.bootstrap.continual_operator import (
        run_monitored_continual_experiment,
    )

    script = fresh.work_dir.parent / "native.sh"
    script.write_text("#!/bin/bash\nexit 0\n")
    before = {str(p): p.read_bytes() for p in fresh.work_dir.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="clean spent"):
        run_monitored_continual_experiment(
            fresh,
            script,
            additional_time=managed.AdditionalTimeRequest(3, "invalid-fresh", None),
        )
    assert before == {
        str(p): p.read_bytes() for p in fresh.work_dir.rglob("*") if p.is_file()
    }
    assert not (fresh.work_dir / "resource_monitor").exists()
