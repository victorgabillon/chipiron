"""Native closure contention preserves decisions, deadlines and managed accounting."""

import json
import subprocess
import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import psutil
import pytest

from chipiron.environments.morpion.bootstrap import continual_operator as operator
from chipiron.environments.morpion.bootstrap import continuous_lifecycle as lifecycle
from chipiron.environments.morpion.bootstrap import (
    continuous_publication as publication,
)
from chipiron.environments.morpion.bootstrap.bootstrap_args import MorpionBootstrapArgs
from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    MorpionBootstrapPaths,
)
from chipiron.environments.morpion.bootstrap.continuous_contract import Policy
from chipiron.environments.morpion.bootstrap.learning_policy import (
    MorpionContinualLearningPolicy,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from tests.environments.test_morpion_continuous_runtime import candidate_family, fixture
from tests.environments.test_morpion_continuous_runtime import continuous as continuous
from tests.environments.test_morpion_learning_schedule import monitored as monitored
from tests.environments.test_morpion_persistent_shutdown import _owned_deadline


@pytest.mark.fast
@pytest.mark.parametrize("worker_exit", [0, 1])
def test_ack_does_not_wait_for_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, worker_exit: int
) -> None:
    """Reuse actual supervisor intent/reap; closure may only retry after worker exit."""
    calls = []

    def close(_paths: MorpionBootstrapPaths, *, deadline: float | None = None) -> bool:
        calls.append(deadline)
        decision = next((tmp_path / "resource_monitor").glob("decision_*.json"))
        assert json.loads(decision.read_text())["reason"] == (
            "supervisor_deadline" if worker_exit == 0 else "persistent_worker_failed"
        )
        if deadline is None:
            return False
        assert not psutil.pid_exists(int((tmp_path / "ready").read_text()))
        return True

    monkeypatch.setattr(lifecycle, "close_admission", close)
    receipt = _owned_deadline(
        tmp_path, monkeypatch, worker_exit, cleanup_elapsed=7200.5
    )
    assert len(calls) == 2 and calls[0] is None and calls[1] is not None
    assert receipt["admission_closed"]
    assert receipt["budget_elapsed_seconds"] == pytest.approx(7200.5)
    assert receipt["decision"]["budget_elapsed_seconds"] == pytest.approx(7199.8)
    assert receipt["test_group_failure"] == (worker_exit != 0)


@pytest.mark.fast
@pytest.mark.usefixtures("continuous")
def test_native_contention_closes_and_finalizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real monitor stops a lock-owning child before closing context4 and charging time."""
    root = tmp_path / "native"
    args = MorpionBootstrapArgs(
        work_dir=root,
        training_device="cpu",
        evaluators_config=candidate_family(),
        learning_schedule=Policy(),
        max_growth_steps_per_cycle=2,
        execution_profile="strict_math",
        torch_intra_op_threads=1,
        torch_inter_op_threads=1,
    )
    policy = MorpionContinualLearningPolicy(
        automatic_activation=True,
        max_elapsed_seconds=2,
        min_available_ram_mb=1,
        shutdown_grace_seconds=3,
    )
    operator.prepare_continual_experiment(args, policy, managed_sessions=True)
    paths = MorpionBootstrapPaths.from_work_dir(root)
    script = tmp_path / "lock-supervisor.sh"
    script.write_text("""#!/bin/bash
exec "$PYTHON_BIN" -I -B -c '
import fcntl,os,signal,time
from pathlib import Path
p=Path(os.environ["MORPION_WORK_DIR"])
f=(p/"pipeline/.growth_patch_consumer.lock").open("rb")
fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
(p/"lock-ready").write_text(str(os.getpid()))
signal.signal(signal.SIGTERM,lambda *_:exit(143))
while True:time.sleep(.05)
'
""")
    monkeypatch.setenv("MORPION_PERSISTENT_WORKERS", "0")
    original = lifecycle.close_admission
    attempts = []

    def observed(p: MorpionBootstrapPaths, *, deadline: float | None = None) -> bool:
        answer = original(p, deadline=deadline)
        attempts.append(answer)
        return answer

    monkeypatch.setattr(lifecycle, "close_admission", observed)
    state = operator.run_monitored_continual_experiment(paths, script)
    assert attempts == [False, True]
    assert not psutil.pid_exists(int((root / "lock-ready").read_text()))
    frame = publication.load_frame(paths, state)
    assert not frame.state.admission_open
    closed = paths.run_state_path.read_bytes()
    assert original(paths) and paths.run_state_path.read_bytes() == closed
    data = state.metadata["managed_sessions_v1"]
    assert data["active"] is None and len(data["sessions"]) == 1
    usage = json.loads((root / "resource_monitor/usage.json").read_text())
    receipt = json.loads(
        (root / f"resource_monitor/session_{usage['session']}.json").read_text()
    )
    assert receipt["reason"] == receipt["decision"]["reason"] == "maximum_elapsed_time"
    assert data["consumed_seconds"] == usage["elapsed_seconds"] >= 2
    assert receipt["admission_closed"] and receipt["shutdown_confirmed"]
    assert state.tree_size_at_last_save == 1
    assert publication.dispatch_generation(paths, state, training=False) is None


@pytest.mark.fast
@pytest.mark.usefixtures("continuous")
def test_foreign_lock_expires_closure_without_mutation(tmp_path: Path) -> None:
    """A live unrelated owner is neither killed nor treated as positive closure."""
    _, paths = fixture(tmp_path / "run")
    code = 'import fcntl,sys; f=open(sys.argv[1],"rb"); fcntl.flock(f,fcntl.LOCK_EX); print("ready",flush=True); sys.stdin.read()'
    with subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            code,
            str(paths.pipeline_dir / ".growth_patch_consumer.lock"),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    ) as owner:
        assert owner.stdout is not None and owner.stdout.readline().strip() == "ready"
        before = paths.run_state_path.read_bytes()
        try:
            start = time.monotonic()
            assert not lifecycle.close_admission(paths, deadline=start + 0.1)
            assert time.monotonic() - start < 1
            assert paths.run_state_path.read_bytes() == before
            assert owner.poll() is None
        finally:
            assert owner.stdin is not None
            owner.stdin.close()
            owner.wait(timeout=3)
    assert lifecycle.close_admission(paths)
    closed = paths.run_state_path.read_bytes()
    assert (
        lifecycle.close_admission(paths) and paths.run_state_path.read_bytes() == closed
    )


@pytest.mark.fast
def test_unproven_closure_retains_native_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even permanent contention keeps the decisive reason and reaps native groups."""
    monkeypatch.setattr(lifecycle, "close_admission", lambda *_a, **_k: False)
    result = _owned_deadline(tmp_path, monkeypatch, 0)
    assert result["test_group_failure"]
    receipt = next(
        json.loads(p.read_text())
        for p in (tmp_path / "resource_monitor").glob("session_*.json")
        if not p.name.endswith(".native.json")
    )
    assert receipt["shutdown_confirmed"] and not receipt["admission_closed"]
    assert receipt["reason"] == receipt["decision"]["reason"] == "supervisor_deadline"
    assert (
        receipt["budget_elapsed_seconds"]
        >= receipt["decision"]["budget_elapsed_seconds"]
        > 0
    )
    assert not psutil.pid_exists(int((tmp_path / "ready").read_text()))


@pytest.mark.fast
@pytest.mark.usefixtures("continuous")
def test_closure_fences_pending_admission_without_erasing_work(tmp_path: Path) -> None:
    """A closed gate preserves Source/debt and refuses its next actual dispatch."""
    from tests.environments.test_morpion_continual_publication import grow

    args, paths = fixture(tmp_path / "run")
    state = grow(args)
    before = publication.load_frame(paths, state)
    assert publication.dispatch_generation(paths, state, training=False) is not None
    assert not lifecycle.close_admission(paths, deadline=time.monotonic() - 1)
    assert (
        publication.load_frame(paths, load_bootstrap_run_state(paths.run_state_path))
        == before
    )
    assert lifecycle.close_admission(paths)
    state = load_bootstrap_run_state(paths.run_state_path)
    after = publication.load_frame(paths, state)
    assert after.state.pending_source == before.state.pending_source
    assert after.state.learning_debt == before.state.learning_debt
    assert publication.dispatch_generation(paths, state, training=False) is None
