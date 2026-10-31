"""Real deaths, conservative worker fencing and historical-context isolation."""

import ctypes
import json
import os
import signal
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from chipiron.environments.morpion.bootstrap import (
    checkpoint_patch_receipts as receipts,
)
from chipiron.environments.morpion.bootstrap import learning_worker
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    publication_records,
)
from chipiron.environments.morpion.bootstrap.learning_policy import (
    validate_continual_context,
)
from chipiron.environments.morpion.bootstrap.learning_schedule import (
    WINDOW_KEY,
    BoundedGrowthWindow,
    validate_supported,
    window,
)
from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
    load_pipeline_stage_claim,
    save_pipeline_stage_claim,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from tests.environments.morpion_fenced_dataset_helpers import run_dataset
from tests.environments.test_morpion_continual_publication import grow, setup_continual
from tests.environments.test_morpion_learning_schedule import (
    monitored as monitored,
)
from tests.environments.test_morpion_learning_schedule import (
    run_growth_cli,
    start_blocked,
)
from tests.environments.test_morpion_value_certainty import endgame as endgame


def crash_growth_worker(root: str, phase: str) -> None:
    """Die at the real authoritative replacement, not a mocked exception unwind."""
    from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
        MorpionBootstrapPaths,
    )

    save = receipts.save_bootstrap_run_state

    def die(state: Any, *args: Any, **kwargs: Any) -> None:
        if state.generation == 2 and state.metadata[WINDOW_KEY]["active"] is not None:
            if phase == "after":
                save(state, *args, **kwargs)
            os._exit(73)
        save(state, *args, **kwargs)

    receipts.save_bootstrap_run_state = die
    run_growth_cli(MorpionBootstrapPaths.from_work_dir(root), 1)


@pytest.mark.fast
@pytest.mark.parametrize("phase", ["before", "after"])
@pytest.mark.usefixtures("monitored")
def test_reservation_is_atomic_across_real_process_death(
    tmp_path: Path, phase: str
) -> None:
    """Recover either the old head or a complete source reservation after death."""
    args, paths = setup_continual(
        tmp_path / "run",
        automatic=True,
        memory_floor_mb=1,
        learning_schedule=BoundedGrowthWindow(1),
    )
    log = tmp_path / "crash.log"
    code = "from tests.environments.test_morpion_learning_schedule_recovery import crash_growth_worker; import sys; crash_growth_worker(*sys.argv[1:])"
    with log.open("wb") as output:
        child = subprocess.run(
            [sys.executable, "-c", code, str(paths.work_dir), phase],
            stdout=output,
            stderr=subprocess.STDOUT,
            timeout=60,
            start_new_session=True,
            check=False,
        )
    assert child.returncode == 73, log.read_text()
    state = load_bootstrap_run_state(paths.run_state_path)
    if phase == "before":
        assert state.generation == 1
        assert window(state)["active"] is None
        state = grow(args)
    else:
        assert state.generation == 2
    slot = window(state)["active"]
    assert slot["source"]["generation"] == 2
    resumed = run_growth_cli(paths, 20)
    assert resumed.cycle_index == slot["target_cycle"]
    assert window(resumed)["active"] == slot
    authority = paths.run_state_path.read_bytes()
    grow(args)
    assert paths.run_state_path.read_bytes() == authority


@pytest.mark.parametrize("stage", ["dataset", "training"])
@pytest.mark.usefixtures("monitored")
def test_dead_learning_retry_requires_empty_session_and_keeps_reserved_source(
    tmp_path: Path,
    stage: str,
) -> None:
    """Fence an expired live claim and retry the same source after real death."""
    args, paths = setup_continual(
        tmp_path / "run",
        automatic=True,
        memory_floor_mb=1,
        learning_schedule=BoundedGrowthWindow(1),
    )
    source = grow(args)
    from chipiron.environments.morpion.bootstrap.pipeline.stages import (
        run_pipeline_training_stage,
    )

    execute = run_dataset if stage == "dataset" else run_pipeline_training_stage
    claim_for = (
        paths.pipeline_dataset_claim_path_for_generation
        if stage == "dataset"
        else paths.pipeline_training_claim_path_for_generation
    )
    if stage == "training":
        run_dataset(args, generation=source.generation)
    process, _markers, output = start_blocked(paths, stage, tmp_path)
    try:
        claim_path = claim_for(source.generation)
        claim = load_pipeline_stage_claim(claim_path)
        expired = replace(claim, expires_at_utc="1970-01-01T00:00:00Z")
        save_pipeline_stage_claim(expired, claim_path)
        # TTL expiration never permits another computation while the old worker lives.
        with pytest.raises(ValueError, match="not quiescent"):
            execute(args, generation=source.generation)
        assert load_pipeline_stage_claim(claim_path).claim_id == claim.claim_id
        grow(args)
        process.kill()
        process.wait(timeout=20)
        output.close()
        # A cold retry may now compute, still against S rather than the latest head.
        result = execute(args, generation=source.generation)
        assert result.generation == source.generation
        state = load_bootstrap_run_state(paths.run_state_path)
        assert f"accepted_{stage}" in publication_records(state.metadata)[1]
        assert f"accepted_{stage}" not in publication_records(state.metadata)[-1]
        assert window(state)["active"]["source"]["generation"] == source.generation
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        output.close()


@pytest.mark.usefixtures("monitored")
def test_dead_coordinator_does_not_prove_child_quiescence(tmp_path: Path) -> None:
    """Refuse takeover while a real orphaned child survives in the old session."""
    current = learning_worker.monitored_identity()
    # Child remains in the dead coordinator's session, as a spawned trainer would.
    code = "import os,time; pid=os.fork(); print(pid,flush=True) if pid else time.sleep(60); os._exit(0)"
    libc = ctypes.CDLL(None, use_errno=True)
    old_subreaper = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(old_subreaper), 0, 0, 0) == 0
    assert libc.prctl(36, 1, 0, 0, 0) == 0
    output = tmp_path / "orphan.pid"
    with output.open("w") as handle:
        parent = subprocess.Popen(
            [sys.executable, "-c", code], stdout=handle, start_new_session=True
        )
        parent.wait(timeout=10)
    child_pid = int(output.read_text().strip())
    previous = {**current, "pid": parent.pid, "session": parent.pid}
    try:
        assert Path(f"/proc/{child_pid}").exists()
        with pytest.raises(ValueError, match="not quiescent"):
            learning_worker.require_quiescent(previous, current)
    finally:
        os.killpg(parent.pid, signal.SIGKILL)
        os.waitpid(child_pid, 0)
        assert libc.prctl(36, old_subreaper.value, 0, 0, 0) == 0


@pytest.mark.usefixtures("monitored")
def test_policy_change_downgrade_and_unknown_worker_evidence_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject changed context and missing supervisor evidence before mutation."""
    args, paths = setup_continual(
        tmp_path / "run",
        automatic=True,
        memory_floor_mb=1,
        learning_schedule=BoundedGrowthWindow(1),
    )
    state = grow(args)
    authority = paths.run_state_path.read_bytes()
    for schedule in (None, BoundedGrowthWindow(2)):
        with pytest.raises(ValueError):
            validate_continual_context(paths, state, args.continual_learning, schedule)
    with pytest.raises(ValueError, match="Missing prior"):
        learning_worker.require_quiescent(None, learning_worker.monitored_identity())
    monkeypatch.delenv("MORPION_RESOURCE_MONITOR_PID")
    with pytest.raises(ValueError, match="resource monitor"):
        grow(args)
    assert paths.run_state_path.read_bytes() == authority


@pytest.mark.usefixtures("monitored")
def test_reserved_source_cannot_retire_after_reader_death(tmp_path: Path) -> None:
    """Retain the durable source even when no process holds a read pin."""
    args, paths = setup_continual(
        tmp_path / "run",
        automatic=True,
        memory_floor_mb=1,
        learning_schedule=BoundedGrowthWindow(4),
    )
    grow(args)
    state = run_growth_cli(paths, 20)
    metadata = json.loads(json.dumps(state.metadata))
    record = metadata["durable_generation_publication_v1"]["generations"][1]
    record["retired_payload_v1"] = record["descriptor_sha256"]
    with pytest.raises(ValueError, match="reserved"):
        publication_records(metadata)


@pytest.mark.usefixtures("monitored")
def test_budget_termination_never_extends_window(tmp_path: Path) -> None:
    """Keep a node-budget termination authoritative before the fixed gate."""
    args, paths = setup_continual(
        tmp_path / "run",
        automatic=True,
        memory_floor_mb=1,
        maximum=2,
        learning_schedule=BoundedGrowthWindow(2),
    )
    state = grow(args)
    assert state.metadata.get("experiment_termination") is not None
    authority = paths.run_state_path.read_bytes()
    run_growth_cli(paths, 20)
    assert paths.run_state_path.read_bytes() == authority


@pytest.mark.usefixtures("monitored")
def test_legacy_default_keeps_v1_context_and_barrier(tmp_path: Path) -> None:
    """Keep historical default identity and its learning boundary unchanged."""
    args, paths = setup_continual(tmp_path / "run", automatic=True, memory_floor_mb=1)
    state = grow(args)
    assert state.metadata["continual_learning"] == {"schema_version": 1}
    assert WINDOW_KEY not in state.metadata
    authority = paths.run_state_path.read_bytes()
    grow(args)
    assert paths.run_state_path.read_bytes() == authority
    with pytest.raises(ValueError, match="requires continual"):
        validate_supported(
            replace(
                args,
                learning_schedule=BoundedGrowthWindow(2),
                continual_learning=replace(
                    args.continual_learning, automatic_activation=False
                ),
            )
        )


def crash_activation_worker(root: str, phase: str) -> None:
    """Exit at real activation/drain commits while leaving the existing request."""
    from chipiron.environments.morpion.bootstrap import model_activation
    from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
        MorpionBootstrapPaths,
    )

    save = receipts.save_bootstrap_run_state
    paths = MorpionBootstrapPaths.from_work_dir(root)
    original_generation = load_bootstrap_run_state(paths.run_state_path).generation

    def die(state: Any, *args: Any, **kwargs: Any) -> None:
        data = window(state)
        activated = data["active"] is None
        drained = state.generation > original_generation and not activated
        if (phase == "drain" and drained) or (phase != "drain" and activated):
            if phase != "before_activation":
                save(state, *args, **kwargs)
            os._exit(74)
        save(state, *args, **kwargs)

    receipts.save_bootstrap_run_state = die
    model_activation.activate_selected_round_automatically(paths)


@pytest.mark.parametrize("phase", ["drain", "before_activation", "after_activation"])
@pytest.mark.usefixtures("monitored")
def test_activation_recovery_preserves_gate_and_single_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    endgame: tuple[Any, Any],
    phase: str,
) -> None:
    """Recover an old-model drain or exact committed activation without extra Growth."""
    from chipiron.environments.morpion.bootstrap import model_activation
    from chipiron.environments.morpion.bootstrap.launcher import (
        launcher_args_from_cli,
        run_morpion_bootstrap_experiment,
    )
    from chipiron.environments.morpion.bootstrap.runtime import runner as runners
    from tests.environments.test_morpion_learning_schedule import learn

    position, _ = endgame
    with monkeypatch.context() as genesis:
        genesis.setattr(runners, "initial_state", lambda: position)
        args, paths = setup_continual(
            tmp_path / "run",
            automatic=True,
            proof=True,
            memory_floor_mb=1,
            learning_schedule=BoundedGrowthWindow(1),
        )
    source = grow(args)
    learn(args, source.generation)
    gate = grow(args)
    run_morpion_bootstrap_experiment(
        launcher_args_from_cli([
            "--work-dir",
            str(paths.work_dir),
            "--scientific",
            "--pipeline-stage",
            "reevaluation",
            "--reevaluation-max-nodes-per-patch",
            "2",
            "--no-print-startup-summary",
        ])
    )
    assert paths.pipeline_reevaluation_patch_path.is_file()
    code = "from tests.environments.test_morpion_learning_schedule_recovery import crash_activation_worker; import sys; crash_activation_worker(*sys.argv[1:])"
    with (tmp_path / "activation-crash.log").open("w", encoding="utf-8") as log:
        child = subprocess.run(
            [sys.executable, "-c", code, str(paths.work_dir), phase],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=60,
            start_new_session=True,
            check=False,
        )
    assert child.returncode == 74, (tmp_path / "activation-crash.log").read_text(
        encoding="utf-8"
    )
    state = load_bootstrap_run_state(paths.run_state_path)
    assert state.cycle_index == gate.cycle_index
    request_path = (
        paths.pipeline_dir
        / "automatic_activation"
        / f"generation_{source.generation:06d}.json"
    )
    if phase == "after_activation":
        assert window(state)["active"] is None

        def forbidden(*args: Any, **kwargs: Any) -> Any:
            pytest.fail("Committed activation replay refreshed the tree")

        monkeypatch.setattr(
            runners.AnemoneMorpionSearchRunner, "refresh_model_estimates", forbidden
        )
        model_activation.activate_model(
            paths, json.loads(request_path.read_text(encoding="utf-8"))
        )
    else:
        assert window(state)["active"] is not None
        model_activation.activate_selected_round_automatically(paths)
    result = load_bootstrap_run_state(paths.run_state_path)
    assert result.cycle_index == gate.cycle_index
    assert window(result)["active"] is None
    assert len(window(result)["completed"]) == 1
    assert window(result)["completed"][0]["outcome"] == "activated"
    authority = paths.run_state_path.read_bytes()
    model_activation.activate_selected_round_automatically(paths)
    assert paths.run_state_path.read_bytes() == authority


@pytest.mark.usefixtures("monitored")
def test_worker_identity_rejects_unknown_namespace_and_boot() -> None:
    """Never infer process death from a different namespace or malformed boot id."""
    current = learning_worker.monitored_identity()
    for changed in ({"pid_namespace": "pid:[999999999]"}, {"boot": "unknown"}):
        with pytest.raises(ValueError):
            learning_worker.require_quiescent({**current, **changed}, current)
