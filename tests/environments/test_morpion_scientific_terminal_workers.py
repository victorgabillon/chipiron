"""Scientific terminal publication is an authenticated wait, never an early exit."""

import json
import os
import signal
import time
from pathlib import Path
from typing import Any

import psutil
import pytest

from chipiron.environments.morpion.bootstrap import continual_operator as operator
from chipiron.environments.morpion.bootstrap import managed_sessions as managed
from chipiron.environments.morpion.bootstrap import persistent_worker as worker
from chipiron.environments.morpion.bootstrap import persistent_worker_state as status
from chipiron.environments.morpion.bootstrap.bootstrap_args import MorpionBootstrapArgs
from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    MorpionBootstrapPaths,
)
from chipiron.environments.morpion.bootstrap.config import load_bootstrap_config
from chipiron.environments.morpion.bootstrap.continuous_contract import Policy
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    publication_records,
)
from chipiron.environments.morpion.bootstrap.learning_policy import (
    MorpionContinualLearningPolicy,
)
from chipiron.environments.morpion.bootstrap.run_state import (
    load_bootstrap_run_state,
    save_bootstrap_run_state,
)
from tests.environments.test_morpion_continual_publication import grow, setup_continual
from tests.environments.test_morpion_continuous_runtime import candidate_family
from tests.environments.test_morpion_continuous_runtime import continuous as continuous
from tests.environments.test_morpion_learning_schedule import monitored as monitored
from tests.environments.test_morpion_persistent_workers import _identity, _launcher


def terminal_fixture(root: Path) -> MorpionBootstrapPaths:
    """Commit a real small-budget Growth terminal head, with real managed identity."""
    args, paths = setup_continual(
        root,
        maximum=2,
        memory_floor_mb=1,
        automatic=True,
        evaluators=candidate_family(),
        learning_schedule=Policy(),
        strict_execution=True,
        managed=True,
    )
    managed.begin_owned(
        paths,
        load_bootstrap_config(paths.bootstrap_config_path),
        "42",
        time.monotonic(),
        60,
    )
    state = grow(args)
    assert state.metadata["experiment_termination"]
    return paths


@pytest.mark.fast
@pytest.mark.usefixtures("continuous")
@pytest.mark.parametrize(
    "damage", [None, "marker", "seal", "missing_binding", "foreign", "inactive"]
)
def test_terminal_guard_authenticates_before_typed_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str | None
) -> None:
    """Only the actual sealed terminal head in the live native launch may wait."""
    paths = terminal_fixture(tmp_path / "run")
    state = load_bootstrap_run_state(paths.run_state_path)
    if damage == "marker":
        state.metadata["experiment_termination"]["reason"] = "invented_stop"
        save_bootstrap_run_state(state, paths.run_state_path, durable=True)
    elif damage in {"seal", "missing_binding"}:
        head = publication_records(state.metadata)[-1]
        seal = paths.work_dir / head["descriptor_path"]
        value = json.loads(seal.read_text())
        value.pop("experiment_termination")
        seal.write_text(json.dumps(value))
        if damage == "missing_binding":
            from chipiron.environments.morpion.bootstrap.generation_publication_io import (
                file_digest,
            )

            head["descriptor_sha256"] = file_digest(seal)
            save_bootstrap_run_state(state, paths.run_state_path, durable=True)
    elif damage == "foreign":
        monkeypatch.setenv("MORPION_RESOURCE_MONITOR_SESSION", "foreign")
    elif damage == "inactive":
        state.metadata[managed.KEY]["active"] = None
        save_bootstrap_run_state(state, paths.run_state_path, durable=True)
    before = paths.run_state_path.read_bytes()
    with pytest.raises(ValueError) as caught:
        managed.require_managed_execution(paths)
    assert isinstance(caught.value, managed.ManagedScientificTerminationError) == (
        damage is None
    )
    assert paths.run_state_path.read_bytes() == before


@pytest.mark.fast
@pytest.mark.usefixtures("continuous")
@pytest.mark.parametrize("stage", sorted(status.STAGES))
@pytest.mark.parametrize("race", [False, True])
def test_terminal_workers_wait_without_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str, race: bool
) -> None:
    """Real guard at boundary/dispatch, then idle heartbeat until supervised signal."""
    from contextlib import contextmanager

    from chipiron.environments.morpion.bootstrap import launcher

    paths = terminal_fixture(tmp_path / "run")
    args = _launcher(paths.work_dir, pipeline_stage=stage)
    monkeypatch.setattr(
        worker, "monitored_identity", lambda: {**_identity(), "session": os.getpid()}
    )
    guard = managed.require_managed_execution
    checks = 0

    def boundary(p: MorpionBootstrapPaths, state: Any = None) -> None:
        nonlocal checks
        checks += 1
        if race and checks <= 2:
            return  # Commit races this read; the unchanged dispatch guard sees it.
        guard(p, state)

    @contextmanager
    def dispatch(configured: Any) -> Any:
        guard(paths)
        yield configured

    monkeypatch.setattr(managed, "require_managed_execution", boundary)
    monkeypatch.setattr(worker, "launcher_execution", dispatch)
    monkeypatch.setattr(worker, "_cleanup", lambda: None)
    monkeypatch.setattr(
        launcher,
        "_run_launcher_cli",
        lambda *_a, **_k: pytest.fail("terminal job dispatched"),
    )
    before = paths.run_state_path.read_bytes()
    waits = []

    def stop(seconds: float) -> None:
        waits.append(seconds)
        record = json.loads(
            (paths.pipeline_dir / f"worker_processes/{stage}.json").read_text()
        )
        assert record["phase"] == "idle"
        assert paths.run_state_path.read_bytes() == before
        worker._interrupt(signal.SIGTERM, None)

    monkeypatch.setattr(worker.time, "sleep", stop)
    with pytest.raises(SystemExit) as caught:
        worker.run_persistent_worker(args)
    assert caught.value.code == 143 and waits == [1.0]
    assert paths.run_state_path.read_bytes() == before


@pytest.mark.usefixtures("continuous")
def test_native_terminal_four_workers_finalize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Actual shell, four installed workers, Growth terminal commit and managed close."""
    root = tmp_path / "native"
    policy = MorpionContinualLearningPolicy(
        automatic_activation=True,
        max_tree_nodes=2,
        max_elapsed_seconds=45,
        min_available_ram_mb=1,
        min_free_disk_mb=1,
        shutdown_grace_seconds=8,
    )
    args = MorpionBootstrapArgs(
        work_dir=root,
        training_device="cpu",
        evaluators_config=candidate_family(),
        learning_schedule=Policy(),
        execution_profile="strict_math",
        torch_intra_op_threads=1,
        torch_inter_op_threads=1,
    )
    operator.prepare_continual_experiment(args, policy, managed_sessions=True)
    paths = MorpionBootstrapPaths.from_work_dir(root)
    monkeypatch.setenv("MORPION_PERSISTENT_WORKERS", "1")
    monkeypatch.setenv("CHIPIRON_CANDIDATE_WORKERS", "1")
    original = operator.load_bootstrap_run_state
    observed = []
    saved = []

    def observe(path: Path) -> Any:
        state = original(path)
        if state.metadata.get("experiment_termination") and not observed:
            # Hold only the test's observer, never Growth ownership. Give each
            # actual five-second poll a chance to encounter the terminal state.
            detected = time.monotonic()
            deadline = detected + 12
            before = path.read_bytes()
            while time.monotonic() < deadline:
                records = [
                    json.loads(f.read_text())
                    for f in (paths.pipeline_dir / "worker_processes").glob("*.json")
                ]
                if len(records) == 4 and all(
                    r["phase"] == "idle" and r["heartbeat_monotonic_s"] > detected + 5
                    for r in records
                ):
                    assert all(psutil.pid_exists(r["identity"]["pid"]) for r in records)
                    observed.extend(records)
                    saved.append(before)
                    assert path.read_bytes() == before
                    break
                time.sleep(0.1)
            assert observed, "four terminal workers did not remain alive and idle"
        return state

    monkeypatch.setattr(operator, "load_bootstrap_run_state", observe)
    script = (
        Path(__file__).resolve().parents[2] / "scripts/launch_morpion_gnome_cluster.sh"
    )
    final = operator.run_monitored_continual_experiment(paths, script)
    assert observed and all(
        not psutil.pid_exists(r["identity"]["pid"]) for r in observed
    )
    before = json.loads(saved[0])
    assert (
        final.metadata["experiment_termination"]
        == before["metadata"]["experiment_termination"]
    )
    assert (
        final.latest_runtime_checkpoint_path == before["latest_runtime_checkpoint_path"]
    )
    assert publication_records(final.metadata) == publication_records(
        before["metadata"]
    )
    assert not final.metadata["continuous_pipeline_v1"]["admission_open"]
    assert final.metadata[managed.KEY]["active"] is None
    usage = json.loads((root / "resource_monitor/usage.json").read_text())
    receipt = json.loads(
        (root / f"resource_monitor/session_{usage['session']}.json").read_text()
    )
    assert receipt["reason"] == before["metadata"]["experiment_termination"]["reason"]
    assert receipt["decision"]["worker_failures"] == []
    assert receipt["shutdown_confirmed"] and receipt["admission_closed"]
    assert not receipt["forced_group_cleanup"] and not receipt["shutdown_grace_expired"]
    assert (
        0
        < receipt["decision"]["budget_elapsed_seconds"]
        <= usage["elapsed_seconds"]
        == final.metadata[managed.KEY]["consumed_seconds"]
        < 45
    )


@pytest.mark.fast
@pytest.mark.usefixtures("continuous")
def test_terminal_wait_has_finite_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-signalling monitor cannot turn terminal idling into an unlimited wait."""
    from dataclasses import replace

    paths = terminal_fixture(tmp_path / "run")
    launcher = _launcher(paths.work_dir)
    launcher = replace(
        launcher,
        bootstrap_args=replace(
            launcher.bootstrap_args,
            continual_learning=MorpionContinualLearningPolicy(
                max_elapsed_seconds=1, shutdown_grace_seconds=1
            ),
        ),
    )
    monkeypatch.setattr(
        worker, "monitored_identity", lambda: {**_identity(), "session": os.getpid()}
    )
    clock = [time.monotonic()]
    waits = []

    def advance(seconds: float) -> None:
        waits.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(worker.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(worker.time, "sleep", advance)
    before = paths.run_state_path.read_bytes()
    with pytest.raises(TimeoutError, match="supervised lifetime"):
        worker.run_persistent_worker(launcher)
    assert waits == [1.0, 1.0]
    record = json.loads(
        (paths.pipeline_dir / "worker_processes/training_worker.json").read_text()
    )
    assert record["phase"] == "stopped" and record["exit_code"] == 1
    assert paths.run_state_path.read_bytes() == before
