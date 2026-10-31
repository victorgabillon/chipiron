"""Real scientific overlap and fixed gates, with no production experiment fixture."""

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from chipiron.environments.morpion.bootstrap import model_activation as activation
from chipiron.environments.morpion.bootstrap import training_publication as training
from chipiron.environments.morpion.bootstrap.bootstrap_args import MorpionBootstrapArgs
from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    MorpionBootstrapPaths,
)
from chipiron.environments.morpion.bootstrap.config import bootstrap_config_from_args
from chipiron.environments.morpion.bootstrap.dashboard.performance_view import (
    load_performance_summary,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    committed_evaluator,
    growth_receipt_ownership,
    publication_records,
)
from chipiron.environments.morpion.bootstrap.launcher import (
    launcher_args_from_cli,
    run_morpion_bootstrap_experiment,
)
from chipiron.environments.morpion.bootstrap.learning_schedule import (
    WINDOW_KEY,
    BoundedGrowthWindow,
    schedule_from_metadata,
    window,
)
from chipiron.environments.morpion.bootstrap.pipeline.stages import (
    run_pipeline_training_stage,
)
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from tests.environments.morpion_fenced_dataset_helpers import (
    local_leaderboard,
    run_dataset,
)
from tests.environments.test_morpion_continual_publication import grow, setup_continual
from tests.environments.test_morpion_value_certainty import endgame as endgame


@pytest.fixture
def monitored(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Use this actual fixture process as the small test's resource supervisor."""
    ticks = Path("/proc/self/stat").read_text().rsplit(") ", 1)[1].split()[19]
    monkeypatch.setenv("MORPION_RESOURCE_MONITOR_PID", str(os.getpid()))
    monkeypatch.setenv("MORPION_RESOURCE_MONITOR_START_TICKS", ticks)
    monkeypatch.setenv("CHIPIRON_CANDIDATE_WORKERS", "1")
    # The production scientific launcher replaces this process-wide default.
    # Register it here so fixture teardown restores the worker's original scope.
    monkeypatch.setenv("MORPION_LEADERBOARD_PATH", str(tmp_path / "leaderboard.jsonl"))
    local_leaderboard(monkeypatch, tmp_path)


def run_growth_cli(paths: MorpionBootstrapPaths, cycles: int) -> Any:
    """Exercise the public multi-cycle API on persisted scientific configuration."""
    return run_morpion_bootstrap_experiment(
        launcher_args_from_cli([
            "--work-dir",
            str(paths.work_dir),
            "--scientific",
            "--pipeline-stage",
            "growth",
            "--max-cycles",
            str(cycles),
            "--no-print-startup-summary",
        ])
    )


def learn(args: MorpionBootstrapArgs, source: int) -> dict[str, Any]:
    """Use the real immutable Dataset and nine/selected-fixture candidate path."""
    run_dataset(args, generation=source)
    run_pipeline_training_stage(args, generation=source)
    state = load_bootstrap_run_state(
        MorpionBootstrapPaths.from_work_dir(args.work_dir).run_state_path
    )
    return next(
        r for r in publication_records(state.metadata) if r["generation"] == source
    )["accepted_training"]


def blocked_worker(root: str, stage: str, markers: str) -> None:
    """Subprocess entry: pause real work outside ownership, before its measurement ends."""
    from chipiron.environments.morpion.bootstrap.pipeline import stages

    path = Path(markers)
    module = stages if stage == "dataset" else training
    name = "publish_dataset" if stage == "dataset" else "_compare_incumbent"
    original = getattr(module, name)

    def blocked(*args: Any, **kwargs: Any) -> Any:
        (path / "ready").write_text("ready")
        deadline = time.monotonic() + 60
        while not (path / "release").exists():
            if time.monotonic() > deadline:
                message = "Test controller did not release the worker"
                raise TimeoutError(message)
            time.sleep(0.02)
        return original(*args, **kwargs)

    setattr(module, name, blocked)
    run_morpion_bootstrap_experiment(
        launcher_args_from_cli([
            "--work-dir",
            root,
            "--scientific",
            "--pipeline-stage",
            stage + "_worker",
            "--no-print-startup-summary",
            "--skip-evaluator-diagnostics",
        ])
    )


def start_blocked(
    paths: MorpionBootstrapPaths, stage: str, tmp_path: Path
) -> tuple[subprocess.Popen[bytes], Path, Any]:
    """Inherit the source import path and give each worker its own session."""
    markers = tmp_path / stage
    markers.mkdir()
    output = (markers / "worker.log").open("wb")
    code = "from tests.environments.test_morpion_learning_schedule import blocked_worker; import sys; blocked_worker(*sys.argv[1:])"
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(paths.work_dir), stage, str(markers)],
        stdout=output,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    deadline = time.monotonic() + 60
    while not (markers / "ready").exists():
        if process.poll() is not None or time.monotonic() > deadline:
            process.kill()
            process.wait()
            output.close()
            pytest.fail((markers / "worker.log").read_text())
        time.sleep(0.02)
    return process, markers, output


def finish_blocked(
    process: subprocess.Popen[bytes], markers: Path, output: Any
) -> None:
    """Reap our own worker so failed assertions cannot leak benchmark work."""
    (markers / "release").touch()
    try:
        assert process.wait(timeout=60) == 0, (markers / "worker.log").read_text()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        output.close()


@pytest.mark.fast
@pytest.mark.usefixtures("monitored")
def test_real_dataset_and_training_overlap_growth_at_exact_gate(tmp_path: Path) -> None:
    """Commit useful Growth during both real stages, then yield exactly at L."""
    args, paths = setup_continual(
        tmp_path / "run",
        automatic=True,
        memory_floor_mb=1,
        learning_schedule=BoundedGrowthWindow(2),
    )
    source = grow(args)
    slot = source.metadata[WINDOW_KEY]["active"]
    assert slot["source"]["generation"] == source.generation == 2
    incumbent = committed_evaluator(paths.work_dir, source)[0]
    process, markers, output = start_blocked(paths, "dataset", tmp_path)
    try:
        after_dataset_overlap = grow(args)
        assert after_dataset_overlap.cycle_index == source.cycle_index + 1
        assert (
            "accepted_dataset"
            not in publication_records(after_dataset_overlap.metadata)[1]
        )
    finally:
        finish_blocked(process, markers, output)
    process, markers, output = start_blocked(paths, "training", tmp_path)
    try:
        gate = run_growth_cli(paths, 20)
        assert gate.cycle_index == slot["target_cycle"]
        assert committed_evaluator(paths.work_dir, gate)[0] == incumbent
        authority = paths.run_state_path.read_bytes()
        assert grow(args).cycle_index == gate.cycle_index
        assert paths.run_state_path.read_bytes() == authority
        with growth_receipt_ownership(paths):
            pass  # Waiting Growth returned without owning the tree lease.
    finally:
        finish_blocked(process, markers, output)
    accepted = load_bootstrap_run_state(paths.run_state_path)
    assert committed_evaluator(paths.work_dir, accepted)[0] == incumbent
    assert "accepted_training" in publication_records(accepted.metadata)[1]
    report = load_performance_summary(paths.work_dir)
    # J1 consumes the actual accepted/nested evidence; no parallel metrics path.
    for stage in ("dataset", "training"):
        observed = [
            r for r in report.timeline if r["Stage"] == stage and r["Generation"] == 2
        ]
        growth = [
            r for r in report.timeline if r["Stage"] == "growth" and r["Generation"] > 2
        ]
        assert any(
            max(a["Start (Unix s)"], b["Start (Unix s)"])
            < min(a["End (Unix s)"], b["End (Unix s)"])
            for a in observed
            for b in growth
        )
    activation.activate_selected_round_automatically(paths)
    resolved = load_bootstrap_run_state(paths.run_state_path)
    assert window(resolved)["active"] is None
    assert len(window(resolved)["completed"]) == 1
    authority = paths.run_state_path.read_bytes()
    activation.activate_selected_round_automatically(paths)
    assert paths.run_state_path.read_bytes() == authority


@pytest.mark.fast
@pytest.mark.usefixtures("monitored")
def test_early_acceptance_waits_for_fixed_gate_and_preserves_loser_parents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep early acceptance inert and continue each candidate from its own parent."""
    args, paths = setup_continual(
        tmp_path / "run",
        automatic=True,
        memory_floor_mb=1,
        learning_schedule=BoundedGrowthWindow(2),
    )
    source = grow(args)
    reference = learn(args, source.generation)
    authority = paths.run_state_path.read_bytes()
    activation.activate_selected_round_automatically(paths)
    assert paths.run_state_path.read_bytes() == authority
    with pytest.raises(ValueError, match="gate"):
        activation.prepare_model_activation_request(
            paths, training_generation=source.generation
        )
    assert paths.run_state_path.read_bytes() == authority
    run_growth_cli(paths, 2)
    activation.activate_selected_round_automatically(paths)
    state = load_bootstrap_run_state(paths.run_state_path)
    assert window(state)["active"] is None

    # Exact accepted replay after closure still uses no computation.
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        message = "accepted retry computed"
        raise AssertionError(message)

    with monkeypatch.context() as patch:
        patch.setattr(training, "reproducibility", forbidden)
        run_pipeline_training_stage(args, generation=source.generation)
    second = grow(args)
    second_source = window(second)["active"]["source"]["generation"]
    second_ref = learn(args, second_source)
    proposal = training._proposal(paths, second_ref)
    assert all(
        parent["accepted_training"] == reference
        for parent in proposal["admission"]["parents"].values()
    )
    assert proposal["admission"]["round_number"] == 2


@pytest.mark.usefixtures("monitored")
def test_cadence_skips_and_retention_resolution(tmp_path: Path) -> None:
    """Record skipped cuts without admitting a newer source or losing the slot."""
    args, paths = setup_continual(
        tmp_path / "run",
        automatic=True,
        memory_floor_mb=1,
        training_cadence=2,
        dataset_cadence=2,
        learning_schedule=BoundedGrowthWindow(3),
    )
    source = grow(args)
    run_growth_cli(paths, 20)
    state = load_bootstrap_run_state(paths.run_state_path)
    assert state.cycle_index == source.cycle_index + 3
    data = window(state)
    assert [r["source"]["generation"] for r in data["skipped"]] == [4]
    assert data["skipped"][0]["reason"] == "skipped_by_active_window"
    # Even a perfectly valid newer closed source is not admitted.
    with pytest.raises(ValueError, match="reserved"):
        run_dataset(args, generation=4)
    assert (
        "accepted_dataset"
        not in publication_records(
            load_bootstrap_run_state(paths.run_state_path).metadata
        )[3]
    )
    learn(args, 2)
    activation.activate_selected_round_automatically(paths)
    assert window(load_bootstrap_run_state(paths.run_state_path))["active"] is None


@pytest.mark.parametrize("value", [0, -1, True, 1.5, None])
def test_invalid_lookahead(value: Any) -> None:
    """Reject zero, negative, non-integral and ambiguous cycle limits."""
    with pytest.raises(ValueError):
        BoundedGrowthWindow(value)


def test_policy_serialization_is_explicit_and_strict(tmp_path: Path) -> None:
    """Preserve omitted metadata and reject unsupported explicit queue sizes."""
    args = MorpionBootstrapArgs(work_dir=tmp_path)
    assert "learning_schedule" not in bootstrap_config_from_args(args).metadata
    policy = BoundedGrowthWindow(2)
    assert schedule_from_metadata({"learning_schedule": policy.to_metadata()}) == policy
    assert schedule_from_metadata({}) is None
    raw = policy.to_metadata()
    raw["queue_capacity"] = 2
    with pytest.raises(ValueError):
        schedule_from_metadata({"learning_schedule": raw})


@pytest.mark.fast
@pytest.mark.usefixtures("monitored")
def test_winning_activation_drains_old_patch_and_commits_closure_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, endgame: tuple[Any, Any]
) -> None:
    """Real proof-bearing tree and winning learner exercise full owned activation."""
    from chipiron.environments.morpion.bootstrap.runtime import runner as runners

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
    reference = learn(args, source.generation)
    proposal = training._proposal(paths, reference)
    assert proposal["status"]["metadata"]["continual_selection"]["activation_eligible"]
    run_growth_cli(paths, 1)
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
    refresh = runners.AnemoneMorpionSearchRunner.refresh_model_estimates
    calls = []

    def observed(self: Any, model: Path) -> Any:
        calls.append(str(model))
        return refresh(self, model)

    monkeypatch.setattr(
        runners.AnemoneMorpionSearchRunner, "refresh_model_estimates", observed
    )
    request = activation.prepare_model_activation_request(
        paths, training_generation=source.generation
    )
    result = activation.activate_model(paths, request)
    assert len(calls) == 1
    assert not paths.pipeline_reevaluation_patch_path.exists()
    data = window(result)
    assert data["active"] is None
    assert data["completed"][0]["outcome"] == "activated"
    assert data["completed"][0]["accepted_training"] == reference
    assert any(s["reason"] == "non_growth_publication" for s in data["skipped"])
    authority = paths.run_state_path.read_bytes()
    activation.activate_model(paths, request)
    assert paths.run_state_path.read_bytes() == authority
    assert len(calls) == 1


@pytest.mark.usefixtures("monitored")
def test_noop_does_not_consume_lookahead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resource refusal cannot be counted as committed lookahead."""
    from chipiron.environments.morpion.bootstrap.pipeline import stages

    args, _paths = setup_continual(
        tmp_path / "run",
        automatic=True,
        memory_floor_mb=1,
        learning_schedule=BoundedGrowthWindow(1),
    )
    source = grow(args)
    with monkeypatch.context() as refused:
        refused.setattr(stages, "log_available_ram_guard", lambda **_: False)
        state = grow(args)
    assert state.cycle_index == source.cycle_index
    assert window(state)["active"] == window(source)["active"]
    assert grow(args).cycle_index == window(source)["active"]["target_cycle"]
