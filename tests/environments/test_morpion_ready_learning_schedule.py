"""Ready-result windows preserve acceptance authority and serialize cutover races."""

import copy
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from chipiron.environments.morpion.bootstrap import model_activation as activation
from chipiron.environments.morpion.bootstrap import training_publication as training
from chipiron.environments.morpion.bootstrap.bootstrap_args import MorpionBootstrapArgs
from chipiron.environments.morpion.bootstrap.config import bootstrap_config_from_args
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    growth_receipt_ownership,
    publication_records,
)
from chipiron.environments.morpion.bootstrap.launcher import launcher_args_from_cli
from chipiron.environments.morpion.bootstrap.learning_policy import (
    validate_continual_context,
)
from chipiron.environments.morpion.bootstrap.learning_schedule import (
    READY_WINDOW_KEY,
    WINDOW_KEY,
    BoundedGrowthWindow,
    ReadyBoundedGrowthWindow,
    context_marker,
    initial_window,
    schedule_from_metadata,
    wait_reason,
)
from chipiron.environments.morpion.bootstrap.pipeline import stages
from chipiron.environments.morpion.bootstrap.run_state import (
    load_bootstrap_run_state,
    save_bootstrap_run_state,
)
from tests.environments.morpion_fenced_dataset_helpers import run_dataset
from tests.environments.test_morpion_continual_publication import grow, setup_continual
from tests.environments.test_morpion_learning_schedule import (
    finish_blocked,
    learn,
    run_growth_cli,
    start_blocked,
    window,
)
from tests.environments.test_morpion_learning_schedule import (
    monitored as monitored,
)


@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, None])
def test_ready_limit_requires_positive_integer(value: Any) -> None:
    """Do not infer an unlimited window or normalize an ambiguous bound."""
    with pytest.raises(ValueError):
        ReadyBoundedGrowthWindow(value)


def test_explicit_policy_and_legacy_bytes(tmp_path: Path) -> None:
    """Pin historical serialized fields while adding a distinct fresh context."""
    legacy = {
        "schema_version": 1,
        "kind": "bounded_growth_window_v1",
        "growth_lookahead_cycles": 4,
        "queue_capacity": 1,
        "cutover": "window_end",
        "refresh": "owned_full",
    }
    assert BoundedGrowthWindow(4).to_metadata() == legacy
    assert context_marker(None) == {"schema_version": 1}
    assert context_marker(BoundedGrowthWindow(4))["schema_version"] == 2
    assert initial_window(None) == {}
    assert (
        "learning_schedule"
        not in bootstrap_config_from_args(
            MorpionBootstrapArgs(work_dir=tmp_path)
        ).metadata
    )
    ready = ReadyBoundedGrowthWindow(4)
    assert schedule_from_metadata({"learning_schedule": ready.to_metadata()}) == ready
    assert context_marker(ready)["schema_version"] == 3
    assert set(initial_window(ready)) == {READY_WINDOW_KEY}
    assert set(initial_window(BoundedGrowthWindow(4))) == {WINDOW_KEY}
    for field, value in [
        ("schema_version", True),
        ("queue_capacity", True),
        ("kind", "unknown"),
        ("max_growth_cycles", True),
        ("cutover", "window_end"),
        ("extra", None),
    ]:
        with pytest.raises(ValueError):
            schedule_from_metadata({
                "learning_schedule": {**ready.to_metadata(), field: value}
            })
    args = launcher_args_from_cli([
        "--work-dir",
        str(tmp_path),
        "--ready-growth-max-cycles",
        "4",
    ])
    assert args.bootstrap_args.learning_schedule == ready
    with pytest.raises(SystemExit):
        launcher_args_from_cli([
            "--ready-growth-max-cycles",
            "4",
            "--growth-lookahead-cycles",
            "4",
        ])


@pytest.mark.usefixtures("monitored")
@pytest.mark.parametrize("metrics_failure", [False, True])
def test_ready_acceptance_resolves_early_and_keeps_each_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, metrics_failure: bool
) -> None:
    """A retained early result closes once; even losing candidates retain lineage."""
    if metrics_failure:
        from chipiron.environments.morpion.bootstrap import runtime_performance

        def unavailable(*_args: Any, **_kwargs: Any) -> None:
            message = "injected observational write failure"
            raise OSError(message)

        monkeypatch.setattr(runtime_performance, "_write", unavailable)
    args, paths = setup_continual(
        tmp_path / "run",
        automatic=True,
        memory_floor_mb=1,
        learning_schedule=ReadyBoundedGrowthWindow(4),
    )
    source = grow(args)
    reference = learn(args, source.generation)
    accepted = load_bootstrap_run_state(paths.run_state_path)
    assert wait_reason(accepted) == "await_learning_resolution"
    activation.activate_selected_round_automatically(paths)
    closed = load_bootstrap_run_state(paths.run_state_path)
    assert closed.cycle_index == source.cycle_index
    data = window(closed)
    assert data["active"] is None and len(data["completed"]) == 1
    assert data["completed"][0]["outcome"] == "retained"
    assert data["decisions"][-1]["accepted_training"] == reference
    assert data["decisions"][-1]["action"] == "retained"
    authority = paths.run_state_path.read_bytes()
    activation.activate_selected_round_automatically(paths)
    assert paths.run_state_path.read_bytes() == authority
    with monkeypatch.context() as patch:

        def forbidden(*_args: Any, **_kwargs: Any) -> Any:
            pytest.fail("Accepted replay computed again")

        patch.setattr(training, "reproducibility", forbidden)
        stages.run_pipeline_training_stage(args, generation=source.generation)
    next_source = grow(args)
    next_ref = learn(args, next_source.generation)
    assert all(
        p["accepted_training"] == reference
        for p in training._proposal(paths, next_ref)["admission"]["parents"].values()
    )


@pytest.mark.usefixtures("monitored")
@pytest.mark.parametrize("maximum", [1, 4])
def test_published_result_is_not_ready_and_cap_is_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, maximum: int
) -> None:
    """Real private/published output cannot authorize resolution before acceptance."""
    args, paths = setup_continual(
        tmp_path / "run",
        automatic=True,
        memory_floor_mb=1,
        learning_schedule=ReadyBoundedGrowthWindow(maximum),
    )
    source = grow(args)
    capped = run_growth_cli(paths, 20)
    assert capped.cycle_index == source.cycle_index + maximum
    assert wait_reason(capped) == "await_dataset"
    run_dataset(args, generation=source.generation)
    assert (
        wait_reason(load_bootstrap_run_state(paths.run_state_path)) == "await_training"
    )
    original = training.accept_training

    def inspect_before_accept(actual_paths: Any, reference: Any) -> Any:
        before = paths.run_state_path.read_bytes()
        assert (
            wait_reason(load_bootstrap_run_state(paths.run_state_path))
            == "await_training"
        )
        assert grow(args).cycle_index == capped.cycle_index
        assert paths.run_state_path.read_bytes() == before
        return original(actual_paths, reference)

    monkeypatch.setattr(training, "accept_training", inspect_before_accept)
    stages.run_pipeline_training_stage(args, generation=source.generation)
    activation.activate_selected_round_automatically(paths)
    data = window(load_bootstrap_run_state(paths.run_state_path))
    assert len(data["completed"]) == 1
    growth = [d for d in data["decisions"] if d["action"] == "grow"]
    assert len(growth) == maximum + 1  # reservation plus permitted ordinary cycles
    assert all(
        d["after_cycle"] == d["before_cycle"] + 1 and d["accepted_training"] is None
        for d in growth
    )


@pytest.mark.usefixtures("monitored")
def test_acceptance_between_preflight_and_owned_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Complete real Training after preflight, then prove owned admission grows zero."""
    args, paths = setup_continual(
        tmp_path / "run",
        automatic=True,
        memory_floor_mb=1,
        learning_schedule=ReadyBoundedGrowthWindow(4),
    )
    source = grow(args)
    original = activation.activate_selected_round_automatically
    calls = []

    def accept_after_preflight(actual_paths: Any) -> None:
        original(actual_paths)
        calls.append(1)
        if len(calls) == 1:
            learn(args, source.generation)

    monkeypatch.setattr(
        activation, "activate_selected_round_automatically", accept_after_preflight
    )

    @contextmanager
    def guarded(actual_paths: Any) -> Any:
        with growth_receipt_ownership(actual_paths):
            yield

    monkeypatch.setattr(stages, "growth_receipt_ownership", guarded)
    result = grow(args)
    assert len(calls) == 2
    assert result.cycle_index == source.cycle_index
    assert window(result)["active"] is None
    with growth_receipt_ownership(paths):
        pass  # No leaked or nested lease.


@pytest.mark.usefixtures("monitored")
def test_termination_wins_acceptance_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An owned stop after preflight prevents both Growth and resolution."""
    args, paths = setup_continual(
        tmp_path / "run",
        automatic=True,
        memory_floor_mb=1,
        learning_schedule=ReadyBoundedGrowthWindow(2),
    )
    source = grow(args)

    def stop_after_preflight(_paths: Any) -> None:
        learn(args, source.generation)
        with growth_receipt_ownership(paths):
            state = load_bootstrap_run_state(paths.run_state_path)
            save_bootstrap_run_state(
                replace(
                    state,
                    metadata={
                        **state.metadata,
                        "experiment_termination": {"reason": "test_budget"},
                    },
                ),
                paths.run_state_path,
                durable=True,
            )

    monkeypatch.setattr(
        activation, "activate_selected_round_automatically", stop_after_preflight
    )
    result = grow(args)
    assert result.cycle_index == source.cycle_index
    assert window(result)["completed"] == []
    assert result.metadata["experiment_termination"]["reason"] == "test_budget"


@pytest.mark.usefixtures("monitored")
def test_format_and_context_changes_are_read_only(tmp_path: Path) -> None:
    """Reject mixed/adopted state and changed declared policies without rewriting it."""
    args, paths = setup_continual(
        tmp_path / "run",
        automatic=True,
        memory_floor_mb=1,
        learning_schedule=ReadyBoundedGrowthWindow(4),
    )
    state = grow(args)
    before = paths.run_state_path.read_bytes()
    for schedule in [None, BoundedGrowthWindow(4), ReadyBoundedGrowthWindow(3)]:
        with pytest.raises(ValueError):
            validate_continual_context(paths, state, args.continual_learning, schedule)
    for corrupt in ["mixed", "bound", "unknown", "key", "history", "null_legacy"]:
        metadata = copy.deepcopy(state.metadata)
        if corrupt == "null_legacy":
            metadata[READY_WINDOW_KEY] = None
            metadata["continual_learning"] = {"schema_version": 1}
        if corrupt == "history":
            metadata[READY_WINDOW_KEY]["decisions"] = []
        if corrupt == "mixed":
            metadata[WINDOW_KEY] = metadata[READY_WINDOW_KEY]
        if corrupt == "bound":
            metadata[READY_WINDOW_KEY]["active"]["limit_cycle"] += 1
        if corrupt == "unknown":
            metadata[READY_WINDOW_KEY]["extra"] = None
        if corrupt == "key":
            metadata[WINDOW_KEY] = metadata.pop(READY_WINDOW_KEY)
        with pytest.raises(ValueError):
            window(replace(state, metadata=metadata))
    assert paths.run_state_path.read_bytes() == before


@pytest.mark.usefixtures("monitored")
def test_acceptance_waits_for_already_owned_cycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real Training acceptance cannot interrupt an admitted cycle under the lease."""
    # The production supervisor declares this bounded existing wait allowance.
    monkeypatch.setenv("MORPION_WORKER_OWNERSHIP_WAIT_SECONDS", "20")
    args, paths = setup_continual(
        tmp_path / "run",
        automatic=True,
        memory_floor_mb=1,
        learning_schedule=ReadyBoundedGrowthWindow(4),
    )
    source = grow(args)
    run_dataset(args, generation=source.generation)
    child, markers, output = start_blocked(paths, "training_accept", tmp_path)
    original = stages._run_pipeline_growth_stage_owned

    def release_under_ownership(*actual: Any, **kwargs: Any) -> Any:
        (markers / "release").touch()
        deadline = time.monotonic() + 15
        while not (markers / "attempting").exists():
            assert child.poll() is None and time.monotonic() < deadline
            time.sleep(0.01)
        state = load_bootstrap_run_state(paths.run_state_path)
        assert "accepted_training" not in publication_records(state.metadata)[1]
        return original(*actual, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(
                stages, "_run_pipeline_growth_stage_owned", release_under_ownership
            )
            result = grow(args)
        assert result.cycle_index == source.cycle_index + 1
    finally:
        finish_blocked(child, markers, output)
    accepted = load_bootstrap_run_state(paths.run_state_path)
    assert accepted.cycle_index == source.cycle_index + 1
    assert wait_reason(accepted) == "await_learning_resolution"
    activation.activate_selected_round_automatically(paths)
    assert window(load_bootstrap_run_state(paths.run_state_path))["active"] is None


@pytest.mark.usefixtures("monitored")
def test_memory_refusal_precedes_ready_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The existing RAM floor can refuse early resolution without changing authority."""
    args, paths = setup_continual(
        tmp_path / "run",
        automatic=True,
        memory_floor_mb=1,
        learning_schedule=ReadyBoundedGrowthWindow(4),
    )
    source = grow(args)
    learn(args, source.generation)
    before = paths.run_state_path.read_bytes()
    monkeypatch.setattr(stages, "log_available_ram_guard", lambda **_: False)
    assert grow(args).cycle_index == source.cycle_index
    assert paths.run_state_path.read_bytes() == before


@pytest.mark.usefixtures("monitored")
def test_no_save_growth_after_ready_promotion_keeps_decision_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real no-save Growth attempt still advances the durable cycle bound."""
    from chipiron.environments.morpion.bootstrap.runtime import runner as runner_module
    from tests.environments.test_morpion_value_certainty import endgame

    position, _ = endgame.__wrapped__()
    monkeypatch.setattr(runner_module, "initial_state", lambda: position)
    args, paths = setup_continual(
        tmp_path / "run",
        automatic=True,
        proof=True,
        memory_floor_mb=1,
        learning_schedule=ReadyBoundedGrowthWindow(2),
    )
    source = grow(args)
    learn(args, source.generation)
    activation.activate_selected_round_automatically(paths)
    closed = load_bootstrap_run_state(paths.run_state_path)
    assert window(closed)["completed"][-1]["outcome"] == "activated"
    attempted = grow(args)
    assert attempted.generation == closed.generation
    assert attempted.cycle_index == closed.cycle_index + 1
    data = window(attempted)
    assert data["active"] is None
    assert data["decisions"][-1]["action"] == "no_save"
    assert (
        data["decisions"][-1]["publication"]
        == data["decisions"][-1]["parent_publication"]
    )
    assert window(load_bootstrap_run_state(paths.run_state_path)) == data
    before = paths.run_state_path.read_bytes()
    monkeypatch.setattr(stages, "log_available_ram_guard", lambda **_: False)
    refused = grow(args)
    assert refused.cycle_index == attempted.cycle_index
    assert paths.run_state_path.read_bytes() == before


def persistent_ready_worker(root: str) -> None:
    """Run two bounded real worker jobs in a supervised CPU fixture session."""
    import json

    from chipiron.environments.morpion.bootstrap.launcher import main
    from chipiron.learning.torch_runtime import (
        configure_torch_process_threads,
        torch_execution_profile,
    )

    configure_torch_process_threads(intra_op=1, inter_op=1)
    with torch_execution_profile("strict_math"):
        args, paths = setup_continual(
            Path(root),
            automatic=True,
            memory_floor_mb=1,
            strict_execution=True,
            learning_schedule=ReadyBoundedGrowthWindow(1),
        )
        source = grow(args)
        learn(args, source.generation)
        assert (
            main([
                "--work-dir",
                root,
                "--scientific",
                "--pipeline-stage",
                "growth",
                "--persistent-worker",
                "--worker-max-jobs",
                "2",
                "--worker-poll-seconds",
                "1",
                "--no-print-startup-summary",
            ])
            == 0
        )
        state = load_bootstrap_run_state(paths.run_state_path)
        data = window(state)
        assert len(data["completed"]) == 1
        assert data["completed"][0]["source_cycle"] == source.cycle_index
        assert state.cycle_index == source.cycle_index + 2
        assert state.cycle_index == data["active"]["limit_cycle"]
        (paths.work_dir.parent / "persistent-result.json").write_text(json.dumps(data))


@pytest.mark.usefixtures("monitored")
def test_persistent_jobs_reconstruct_ready_policy(tmp_path: Path) -> None:
    """The real persistent launcher rereads v2 authority and stops exactly at H."""
    import subprocess
    import sys

    code = "from tests.environments.test_morpion_ready_learning_schedule import persistent_ready_worker; import sys; persistent_ready_worker(sys.argv[1])"
    with (tmp_path / "persistent.log").open("w") as log:
        child = subprocess.run(
            [sys.executable, "-c", code, str(tmp_path / "run")],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=60,
            start_new_session=True,
            check=False,
        )
    assert child.returncode == 0, (tmp_path / "persistent.log").read_text()
    assert (tmp_path / "persistent-result.json").is_file()


def crash_retained_worker(root: str, phase: str) -> None:
    """Exit at the existing retained-decision save, without a new commit seam."""
    import os

    from chipiron.environments.morpion.bootstrap import run_state as authority
    from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
        MorpionBootstrapPaths,
    )

    original = authority.save_bootstrap_run_state

    def die(state: Any, *args: Any, **kwargs: Any) -> None:
        if window(state)["completed"]:
            if phase == "after":
                original(state, *args, **kwargs)
            os._exit(75)
        original(state, *args, **kwargs)

    authority.save_bootstrap_run_state = die
    activation.activate_selected_round_automatically(
        MorpionBootstrapPaths.from_work_dir(root)
    )


@pytest.mark.parametrize("phase", ["before", "after"])
@pytest.mark.usefixtures("monitored")
def test_retained_decision_recovers_once_after_process_death(
    tmp_path: Path, phase: str
) -> None:
    """Accepted authority survives either side of the unchanged retained save."""
    import subprocess
    import sys

    args, paths = setup_continual(
        tmp_path / "run",
        automatic=True,
        memory_floor_mb=1,
        learning_schedule=ReadyBoundedGrowthWindow(4),
    )
    source = grow(args)
    accepted = learn(args, source.generation)
    code = "from tests.environments.test_morpion_ready_learning_schedule import crash_retained_worker; import sys; crash_retained_worker(*sys.argv[1:])"
    with (tmp_path / "retain-crash.log").open("w") as log:
        child = subprocess.run(
            [sys.executable, "-c", code, str(paths.work_dir), phase],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=60,
            start_new_session=True,
            check=False,
        )
    assert child.returncode == 75, (tmp_path / "retain-crash.log").read_text()
    state = load_bootstrap_run_state(paths.run_state_path)
    assert bool(window(state)["completed"]) == (phase == "after")
    activation.activate_selected_round_automatically(paths)
    state = load_bootstrap_run_state(paths.run_state_path)
    assert state.cycle_index == source.cycle_index
    assert len(window(state)["completed"]) == 1
    assert window(state)["completed"][0]["accepted_training"] == accepted
    before = paths.run_state_path.read_bytes()
    activation.activate_selected_round_automatically(paths)
    assert paths.run_state_path.read_bytes() == before
