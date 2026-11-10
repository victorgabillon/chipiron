"""Time-grant contracts with tiny fresh genesis and controlled native receipts.

Controlled receipts are unit evidence only. The separate installed integration
fixtures exercise the actual sampler/shell, scientific continuation and reaping.
"""

import subprocess
import sys
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest

from chipiron.environments.morpion.bootstrap import managed_sessions as managed
from chipiron.environments.morpion.bootstrap.bootstrap_args import MorpionBootstrapArgs
from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    MorpionBootstrapPaths,
)
from chipiron.environments.morpion.bootstrap.config import load_bootstrap_config
from chipiron.environments.morpion.bootstrap.continual_operator import (
    prepare_continual_experiment,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    growth_receipt_ownership,
)
from chipiron.environments.morpion.bootstrap.learning_policy import (
    MorpionContinualLearningPolicy,
)
from chipiron.environments.morpion.bootstrap.run_state import (
    load_bootstrap_run_state,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def fresh(tmp_path: Path) -> MorpionBootstrapPaths:
    """Use actual genesis; no scientific payload or historical fixture is imported."""
    root = tmp_path / "normal"
    prepare_continual_experiment(
        MorpionBootstrapArgs(work_dir=root, training_device="cpu"),
        MorpionContinualLearningPolicy(max_elapsed_seconds=2, min_available_ram_mb=1),
        managed_sessions=True,
    )
    return MorpionBootstrapPaths.from_work_dir(root)


def snapshot(paths: MorpionBootstrapPaths) -> tuple[Any, Any]:
    """Read config/state through the normal persistence primitives."""
    return load_bootstrap_config(paths.bootstrap_config_path), load_bootstrap_run_state(
        paths.run_state_path
    )


def controlled_close(
    paths: MorpionBootstrapPaths,
    *,
    crash: str | None = None,
    identifier: str = "123",
    early: bool = False,
) -> None:
    """Use a real owned/reaped child identity, with explicitly controlled clocks.

    The child is a unit coordinator creating synthetic monitor telemetry; it is
    not represented as an actual native-monitor or scientific execution receipt.
    """
    code = """import json,sys,time
from pathlib import Path
from chipiron.environments.morpion.bootstrap import managed_sessions as m
from chipiron.environments.morpion.bootstrap.bootstrap_paths import MorpionBootstrapPaths
from chipiron.environments.morpion.bootstrap.config import load_bootstrap_config
from chipiron.environments.morpion.bootstrap.derived.provenance import atomic_json
p=MorpionBootstrapPaths.from_work_dir(Path(sys.argv[1]));config=load_bootstrap_config(p.bootstrap_config_path)
started=time.monotonic()-3
from chipiron.environments.morpion.bootstrap.run_state import load_bootstrap_run_state
data=load_bootstrap_run_state(p.run_state_path).metadata[m.KEY]
identifier=sys.argv[3];ceiling=data['authorized_seconds'];base=data['consumed_seconds']
link=m.begin_owned(p,config,identifier,started,2)
if sys.argv[2]=='launch':sys.exit(0)
early=sys.argv[4]=='early';reason='supervisor_deadline' if early else 'maximum_elapsed_time'
decisive=base+1 if early else ceiling;cleanup=decisive+(.25 if early else .75)
decision={'reason':reason,'monotonic_s':started+decisive-base,'budget_elapsed_seconds':float(decisive),'worker_failures':[],'control_failure':None,'managed_session':link}
if early:decision['supervisor_shutdown']={'authenticated':True}
usage={'session':identifier,'elapsed_seconds':cleanup,'reason':reason,'managed_session':link}
receipt={'reason':reason,'decision':decision,'budget_elapsed_seconds':cleanup,'shutdown_confirmed':True,'forced_group_cleanup':False,'shutdown_grace_expired':False,'supervisor_exit_code':0,'managed_owned_groups':[],'managed_session':link}
monitor=p.work_dir/'resource_monitor'
for name,value in [(f'decision_{identifier}.json',decision),(f'session_{identifier}.json',receipt),('usage.json',usage)]:atomic_json(monitor/name,value)
if sys.argv[2]=='native':sys.exit(0)
if sys.argv[2]=='close':
 original=m._commit
 def stop(*a,**k):raise RuntimeError('controlled close-before-root crash')
 m._commit=stop
 try:m.finalize_managed_session_owned(p,config,identifier)
 except RuntimeError:sys.exit(0)
else:m.finalize_managed_session_owned(p,config,identifier)
"""
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            code,
            str(paths.work_dir),
            crash or "none",
            identifier,
            "early" if early else "spent",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.fast
@pytest.mark.parametrize(
    "raw", ["0", "-1", "1.0", "nan", "inf", "True", "43201", "9999999999999999999"]
)
def test_seconds_refuse_nonpositive_or_nonintegral(raw: str) -> None:
    """No numeric coercion can create authority from malformed input."""
    with pytest.raises(ValueError):
        managed.positive_seconds(raw)


@pytest.mark.fast
@pytest.mark.parametrize("value", [True, -1, float("nan"), float("inf"), 10**1000])
def test_invalid_clocks(value: Any) -> None:
    """Invalid clocks never become zero remaining consumption."""
    with pytest.raises(ValueError):
        managed._number(value)


@pytest.mark.fast
def test_genesis_has_one_enrollment_and_no_model_changes(
    fresh: MorpionBootstrapPaths,
) -> None:
    """Managed metadata is committed alongside the same scientific baseline."""
    config, state = snapshot(fresh)
    data = managed.validate(fresh, config, state)
    assert (
        data is not None
        and data["authorized_seconds"] == 2
        and data["consumed_seconds"] == 0
    )
    assert state.generation == 1 and data["active"] is None and data["grants"] == []
    assert managed.inspect(fresh, config, state)["status"] == "ready"
    with pytest.raises(ValueError):
        managed.plan_additional_time(fresh, managed.inspect(fresh, config, state), 3)
    with pytest.raises(ValueError):
        managed.require_managed_execution(fresh)


@pytest.mark.fast
def test_grant_debits_cleanup_and_is_idempotent(fresh: MorpionBootstrapPaths) -> None:
    """A real reaped identity and controlled receipt bind duplicate grant admission."""
    controlled_close(fresh)
    config, state = snapshot(fresh)
    oldscientific = {k: v for k, v in state.metadata.items() if k != managed.KEY}
    summary = managed.inspect(fresh, config, state)
    request, duplicate = managed.plan_additional_time(fresh, summary, 3, "client-1")
    assert (
        not duplicate
        and request.predecessor == state.metadata[managed.KEY]["sessions"][-1]
    )
    before = fresh.run_state_path.read_bytes()
    assert managed.plan_additional_time(fresh, summary, 3, "client-1")[0] == request
    assert fresh.run_state_path.read_bytes() == before
    with growth_receipt_ownership(fresh, create=False):
        updated, work = managed.admit_additional_time_owned(
            fresh, config, state, request
        )
    assert work and updated.metadata[managed.KEY]["authorized_seconds"] == 5
    assert (
        updated.metadata[managed.KEY]["authorized_seconds"]
        - updated.metadata[managed.KEY]["consumed_seconds"]
        == 2.25
    )
    assert {
        k: v for k, v in updated.metadata.items() if k != managed.KEY
    } == oldscientific
    before = fresh.run_state_path.read_bytes()
    with growth_receipt_ownership(fresh, create=False):
        again, work = managed.admit_additional_time_owned(
            fresh, config, updated, request
        )
    assert work and again == updated and fresh.run_state_path.read_bytes() == before
    with pytest.raises(ValueError, match="Conflicting"):
        managed.plan_additional_time(
            fresh, managed.inspect(fresh, config, again), 4, "client-1"
        )


@pytest.mark.fast
def test_completed_latest_request_does_not_restart_with_credit(
    fresh: MorpionBootstrapPaths,
) -> None:
    """A completed request is idempotent even when an early stop leaves credit."""
    from tests.environments.test_morpion_lifecycle_continuation import inert_supervisor

    controlled_close(fresh)
    config, state = snapshot(fresh)
    request, _ = managed.plan_additional_time(
        fresh, managed.inspect(fresh, config, state), 10, "client-early"
    )
    with growth_receipt_ownership(fresh, create=False):
        managed.admit_additional_time_owned(fresh, config, state, request)
    controlled_close(fresh, identifier="124", early=True)
    config, state = snapshot(fresh)
    summary = managed.inspect(fresh, config, state)
    assert summary["authorized_seconds"] - summary["consumed_seconds"] > 1
    before = fresh.run_state_path.read_bytes()
    with growth_receipt_ownership(fresh, create=False):
        unchanged, work = managed.admit_additional_time_owned(
            fresh, config, state, request
        )
    assert not work and unchanged == state
    assert fresh.run_state_path.read_bytes() == before
    script = inert_supervisor(fresh.work_dir.parent / "supervisor.sh")
    from chipiron.environments.morpion.bootstrap.lifecycle import plan_start

    duplicate = plan_start(
        fresh.work_dir,
        cluster_script=script,
        additional_seconds=10,
        request_id="client-early",
    )
    assert duplicate["action"] == "no-work"
    assert "already completed" in duplicate["no_work_reason"]
    assert plan_start(fresh.work_dir, cluster_script=script)["action"] == "resume"
    assert fresh.run_state_path.read_bytes() == before


@pytest.mark.fast
@pytest.mark.parametrize("crash", ["launch", "native"])
def test_incomplete_launch_never_uses_prior_credit(
    fresh: MorpionBootstrapPaths, crash: str
) -> None:
    """Even dead/reaped owners cannot prove an unvalidated native close succeeded."""
    controlled_close(fresh, crash=crash)
    config, state = snapshot(fresh)
    before = fresh.run_state_path.read_bytes()
    assert managed.inspect(fresh, config, state)["status"] == "blocked/uncertain"
    with pytest.raises(ValueError, match="validated close"):
        managed.recover_close_owned(fresh, config)
    assert fresh.run_state_path.read_bytes() == before


@pytest.mark.fast
def test_positive_close_root_recovery_is_exact(fresh: MorpionBootstrapPaths) -> None:
    """Only the exact positively validated close can finish a missed root commit."""
    controlled_close(fresh, crash="close")
    config, state = snapshot(fresh)
    assert (
        managed.inspect(fresh, config, state)["status"]
        == "accounting-finalization-required"
    )
    managed.recover_close_owned(fresh, config)
    config, state = snapshot(fresh)
    assert state.metadata[managed.KEY]["consumed_seconds"] == 2.75
    before = fresh.run_state_path.read_bytes()
    managed.recover_close_owned(fresh, config)
    assert fresh.run_state_path.read_bytes() == before


@pytest.mark.fast
@pytest.mark.parametrize(
    "damage", ["marker", "config", "total", "native", "node", "clock", "new-attempt"]
)
def test_no_write_on_invalid_or_terminal_renewal(
    fresh: MorpionBootstrapPaths, damage: str
) -> None:
    """No enrollment, identity, time or terminal fence can be repaired by a grant."""
    controlled_close(fresh)
    config, state = snapshot(fresh)
    if damage == "marker":
        config = replace(
            config,
            metadata={k: v for k, v in config.metadata.items() if k != managed.KEY},
        )
    elif damage == "config":
        config = replace(config, metadata={**config.metadata, "changed": True})
    elif damage == "total":
        state = replace(
            state,
            metadata={
                **state.metadata,
                managed.KEY: {**state.metadata[managed.KEY], "authorized_seconds": 20},
            },
        )
    elif damage == "native":
        (fresh.work_dir / "resource_monitor/decision_123.json").write_text("{}")
    elif damage == "node":
        state = replace(
            state,
            metadata={
                **state.metadata,
                "experiment_termination": {"reason": "maximum_tree_nodes"},
            },
        )
    elif damage == "clock":
        state = replace(
            state,
            metadata={
                **state.metadata,
                managed.KEY: {**state.metadata[managed.KEY], "consumed_seconds": True},
            },
        )
    else:
        (fresh.work_dir / "resource_monitor/session_124.log").write_text(
            "incomplete attempt"
        )
    before = {str(p): p.read_bytes() for p in fresh.work_dir.rglob("*") if p.is_file()}
    with pytest.raises((ValueError, KeyError)):
        summary = managed.inspect(fresh, config, state)
        managed.plan_additional_time(fresh, summary, 3)
    assert before == {
        str(p): p.read_bytes() for p in fresh.work_dir.rglob("*") if p.is_file()
    }


@pytest.mark.fast
def test_scientific_terminal_label_survives_elapsed_history(
    fresh: MorpionBootstrapPaths,
) -> None:
    """A node terminal is never presented as merely an exhausted time session."""
    controlled_close(fresh)
    config, state = snapshot(fresh)
    state = replace(
        state,
        metadata={
            **state.metadata,
            "experiment_termination": {"reason": "maximum_tree_nodes"},
        },
    )
    assert managed.inspect(fresh, config, state)["status"] == "scientific-terminal"


@pytest.mark.fast
def test_old_request_preview_does_not_launch_a_newer_grant(
    fresh: MorpionBootstrapPaths,
) -> None:
    """An old stable client ID returns no-work while a newer grant has credit."""
    from chipiron.environments.morpion.bootstrap.lifecycle import plan_start

    controlled_close(fresh)
    config, state = snapshot(fresh)
    first, _ = managed.plan_additional_time(
        fresh, managed.inspect(fresh, config, state), 3, "first"
    )
    with growth_receipt_ownership(fresh, create=False):
        managed.admit_additional_time_owned(fresh, config, state, first)
    controlled_close(fresh, identifier="124")
    config, state = snapshot(fresh)
    second, _ = managed.plan_additional_time(
        fresh, managed.inspect(fresh, config, state), 4, "second"
    )
    with growth_receipt_ownership(fresh, create=False):
        managed.admit_additional_time_owned(fresh, config, state, second)
    script = fresh.work_dir.parent / "supervisor.sh"
    script.write_text("#!/bin/bash\nexit 0\n")
    before = fresh.run_state_path.read_bytes()
    plan = plan_start(
        fresh.work_dir, cluster_script=script, additional_seconds=3, request_id="first"
    )
    assert (
        plan["action"] == "no-work"
        and "already completed" in plan["no_work_reason"]
        and plan["remaining_seconds"] > 0
    )
    assert fresh.run_state_path.read_bytes() == before


@pytest.mark.fast
def test_enrollment_does_not_follow_copied_workspace_identity(
    fresh: MorpionBootstrapPaths,
) -> None:
    """The same config/workspace IDs cannot enroll a different filesystem root."""
    config, state = snapshot(fresh)
    other = MorpionBootstrapPaths.from_work_dir(fresh.work_dir.parent / "foreign")
    with pytest.raises(ValueError, match=r"enrollment|workspace"):
        managed.validate(other, config, state)
    assert not other.work_dir.exists()


@pytest.mark.fast
def test_growth_config_reconstruction_keeps_enrollment_identity(
    fresh: MorpionBootstrapPaths,
) -> None:
    """Growth must not erase operational enrollment when rebuilding canonical args."""
    from chipiron.environments.morpion.bootstrap.config import (
        bootstrap_config_from_args,
        bootstrap_config_sha256,
    )
    from chipiron.environments.morpion.bootstrap.launcher import _scientific_args

    config, state = snapshot(fresh)
    args = _scientific_args(MorpionBootstrapArgs(work_dir=fresh.work_dir), config)
    reconstructed = bootstrap_config_from_args(args)
    assert managed.KEY not in reconstructed.metadata
    restored = managed.preserve_capability(fresh, state, config, reconstructed)
    assert bootstrap_config_sha256(restored) == bootstrap_config_sha256(config)
    with pytest.raises(ValueError, match="recipe"):
        managed.preserve_capability(
            fresh,
            state,
            config,
            replace(reconstructed, validation_seed=config.validation_seed + 1),
        )
