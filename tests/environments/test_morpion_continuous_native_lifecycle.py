"""Actual native stop/resume and settled continuous fork; no synthetic close receipts."""

import shlex
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from chipiron.environments.morpion.bootstrap import (
    continuous_lifecycle,
    continuous_publication,
    fork,
    managed_sessions,
    training_publication,
)
from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    MorpionBootstrapPaths,
)
from chipiron.environments.morpion.bootstrap.continuous_contract import Policy
from chipiron.environments.morpion.bootstrap.fork_origin import seeds
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
from tests.environments import test_morpion_managed_native as native
from tests.environments.test_morpion_fork import cli


def test_actual_native_resume_and_settled_fork(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Close children before suspension, preserve both own lineages and fork at epoch zero."""
    monkeypatch.setenv("CHIPIRON_SHARED_PREPARATION", "1")
    monkeypatch.delenv("CHIPIRON_SHARED_PREPARATION_REUSE", raising=False)
    original = native.prepare_continual_experiment

    def prepare(args: Any, policy: Any, **kwargs: Any) -> Any:
        return original(
            replace(args, learning_schedule=Policy()),
            replace(policy, automatic_activation=True),
            **kwargs,
        )

    with monkeypatch.context() as setup:
        setup.setattr(native, "prepare_continual_experiment", prepare)
        args = native.initialize(tmp_path / "parent")
    paths = MorpionBootstrapPaths.from_work_dir(args.work_dir)
    script = tmp_path / "supervisor.sh"
    script.write_text(
        "#!/bin/bash\nexec "
        + shlex.quote(sys.executable)
        + " -B "
        + shlex.quote(
            str(Path(native.__file__).with_name("managed_session_workload.py"))
        )
        + ' "$MORPION_WORK_DIR"\n'
    )
    result = native.invoke(paths.work_dir, script, 1)
    assert result.returncode == 0, result.stdout + result.stderr
    state = load_bootstrap_run_state(paths.run_state_path)
    first = continuous_publication.load_frame(paths, state)
    assert not first.state.admission_open and first.state.pending_promotion is not None
    assert state.metadata[managed_sessions.KEY]["active"] is None
    with pytest.raises(ValueError, match=r"[Uu]nsettled"):
        fork.plan_fork(paths.work_dir, tmp_path / "busy", 300, 10000)
    names = list(args.resolved_evaluators_config().evaluators)
    parents = training_publication.latest_candidate_parents(paths, state, names)
    result = native.invoke(
        paths.work_dir,
        script,
        1,
        "--add-time",
        str(native.SESSION_SECONDS),
        "--request-id",
        "continuous-second-session",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    state = load_bootstrap_run_state(paths.run_state_path)
    frame = continuous_publication.load_frame(paths, state)
    assert not frame.state.admission_open and frame.state.last_accepted_round == 2
    data = state.metadata[managed_sessions.KEY]
    assert data["active"] is None and len(data["sessions"]) == 2
    latest = training_publication.latest_candidate_parents(paths, state, names)
    for name in names:
        proposal = training_publication._proposal(
            paths, latest[name]["accepted_training"]
        )
        assert proposal["admission"]["parents"][name] == parents[name]
    continuous_lifecycle.settled_discard(paths)
    state = load_bootstrap_run_state(paths.run_state_path)
    continuous_lifecycle.require_settled(paths, state)
    child_root = tmp_path / "child"
    cli(paths.work_dir, child_root)
    child_paths = MorpionBootstrapPaths.from_work_dir(child_root)
    child = load_bootstrap_run_state(child_paths.run_state_path)
    child_frame = continuous_publication.load_frame(child_paths, child)
    assert (
        child_frame.state.incumbent_epoch == 0
        and child_frame.state.last_accepted_round == 0
    )
    imported = seeds(child_paths, child, verify=True)
    assert imported.keys() == latest.keys()
    for name in names:
        assert imported[name]["model"]["sha256"] == latest[name]["model"]["sha256"]
        assert imported[name]["weights_sha256"] == latest[name]["weights_sha256"]
