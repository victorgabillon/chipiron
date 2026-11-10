"""Bounded metadata previews and native delegation; long parity uses real fixtures."""

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
import torch

from chipiron.environments.morpion.bootstrap import lifecycle as life
from chipiron.environments.morpion.bootstrap import operator_cli
from chipiron.environments.morpion.bootstrap.bootstrap_args import MorpionBootstrapArgs
from chipiron.environments.morpion.bootstrap.bootstrap_paths import (
    MorpionBootstrapPaths,
)
from chipiron.environments.morpion.bootstrap.continual_operator import (
    prepare_continual_experiment,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    growth_receipt_ownership,
)
from chipiron.environments.morpion.bootstrap.learning_policy import (
    MorpionContinualLearningPolicy,
)

if TYPE_CHECKING:
    from pathlib import Path


def inventory(root: Path) -> dict[str, bytes]:
    """Keep no-write assertions independent of metadata interpretation."""
    return {
        str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }


@pytest.fixture
def prepared(tmp_path: Path) -> tuple[Path, Path]:
    """Actual native fresh genesis, not a fabricated committed head."""
    root = tmp_path / "normal"
    prepare_continual_experiment(
        MorpionBootstrapArgs(work_dir=root), MorpionContinualLearningPolicy()
    )
    script = tmp_path / "supervisor.sh"
    script.write_text("#!/bin/bash\nexit 0\n")
    return root, script


@pytest.mark.fast
def test_preview_is_observational_and_does_not_load_payloads(
    prepared: tuple[Path, Path],
) -> None:
    """Real source identity and genesis work without Torch/GPU policy mutation."""
    root, script = prepared
    before = inventory(root)
    with patch.object(
        torch, "load", side_effect=AssertionError("preview loaded tensors")
    ):
        plan = life.plan_start(root, cluster_script=script)
    assert plan["action"] == "resume"
    assert plan["head"]["generation"] == 1
    assert len(plan["candidate_parents"]) == 9
    assert all(p is None for p in plan["candidate_parents"].values())
    assert plan["accepted_datasets"] == [] and plan["consumed_seconds"] == 0
    assert inventory(root) == before and not torch.cuda.is_initialized()


@pytest.mark.fast
@pytest.mark.parametrize("mode", ["auto", "resume", "fresh"])
def test_partial_never_falls_back_to_fresh(tmp_path: Path, mode: Any) -> None:
    """Partial metadata refuses before writes or process registration."""
    root = tmp_path / "partial"
    root.mkdir()
    (root / "bootstrap_config.json").write_text("{}")
    before = inventory(root)
    with pytest.raises(ValueError):
        life.plan_start(root, mode)
    assert inventory(root) == before


@pytest.mark.fast
def test_fresh_and_sealed_destinations(
    tmp_path: Path, prepared: tuple[Path, Path]
) -> None:
    """Fresh only creates an observational plan and refuses sealed/occupied roots."""
    root, script = prepared
    new = tmp_path / "new"
    assert life.plan_start(new, "fresh", cluster_script=script)["action"] == "create"
    assert not new.exists() and not new.with_name("new-inputs").exists()
    with pytest.raises(ValueError, match="Fresh requires"):
        life.plan_start(root, "fresh", cluster_script=script)
    (root.parent / "LAUNCH_READY.json").write_text("{}")
    before = inventory(root.parent)
    with pytest.raises(ValueError, match="Sealed"):
        life.plan_start(root, cluster_script=script)
    assert inventory(root.parent) == before


@pytest.mark.fast
def test_explicit_conflict_and_owner_refuse_without_writes(
    prepared: tuple[Path, Path],
) -> None:
    """Native owners and persisted fields cannot be bypassed by the normal entry."""
    root, script = prepared
    before = inventory(root)
    with pytest.raises(ValueError, match="conflicts"):
        life.plan_start(root, recipe=["--max-rows", "17"], cluster_script=script)
    assert inventory(root) == before
    paths = MorpionBootstrapPaths.from_work_dir(root)
    with growth_receipt_ownership(paths, create=False), pytest.raises(BlockingIOError):
        life.plan_start(root, cluster_script=script)
    assert inventory(root) == before
    assert (
        life.plan_start(
            root, recipe=["--max-tree-nodes", "200000"], cluster_script=script
        )["action"]
        == "resume"
    )


@pytest.mark.parametrize("damage", ["config", "head", "source", "checkpoint", "legacy"])
def test_corrupt_or_incompatible_metadata_fails_closed(
    prepared: tuple[Path, Path], damage: str
) -> None:
    """Never repair/configure/initialize a failed resume."""
    root, script = prepared
    path = root / "run_state.json"
    state = json.loads(path.read_text())
    if damage == "config":
        (root / "bootstrap_config.json").write_text("{")
    elif damage == "head":
        state["latest_runtime_checkpoint_path"] = "missing"
    elif damage == "checkpoint":
        (root / state["latest_runtime_checkpoint_path"] / "manifest.json").unlink()
    elif damage == "legacy":
        state["metadata"].pop("durable_training")
    else:
        # Inject the mismatch at the existing source fence, without editing sealed payloads.
        with (
            patch.object(
                life, "_source_identity", side_effect=ValueError("source changed")
            ),
            pytest.raises(ValueError, match="source changed"),
        ):
            life.plan_start(root, cluster_script=script)
        return
    if damage in {"head", "legacy"}:
        path.write_text(json.dumps(state))
    before = inventory(root)
    with pytest.raises((ValueError, KeyError)):
        life.plan_start(root, cluster_script=script)
    assert inventory(root) == before


def receipt(
    root: Path,
    *,
    consumed: object = 4.0,
    reason: str = "supervisor_exit",
    failed: bool = False,
) -> None:
    """Write controlled monitor metadata without claiming real process execution."""
    directory = root / "resource_monitor"
    directory.mkdir()
    decision = {
        "monotonic_s": 1.0,
        "reason": reason,
        "budget_elapsed_seconds": consumed,
        "worker_failures": [1] if failed else [],
        "control_failure": None,
    }
    values = {
        "usage.json": {"session": "123", "elapsed_seconds": consumed, "reason": reason},
        "decision_123.json": decision,
        "session_123.json": {
            "reason": reason,
            "decision": decision,
            "budget_elapsed_seconds": consumed,
            "shutdown_confirmed": True,
            "forced_group_cleanup": False,
            "shutdown_grace_expired": False,
            "supervisor_exit_code": 0,
            "environment_owned_processes": {},
        },
    }
    for name, data in values.items():
        (directory / name).write_text(json.dumps(data))


@pytest.mark.fast
@pytest.mark.parametrize("consumed", [True, -1.0, float("inf")])
def test_uncertain_usage_never_becomes_zero(
    prepared: tuple[Path, Path], consumed: object
) -> None:
    """Invalid clock evidence cannot renew authorization."""
    root, script = prepared
    receipt(root, consumed=consumed)
    before = inventory(root)
    with pytest.raises(ValueError):
        life.plan_start(root, cluster_script=script)
    assert inventory(root) == before


@pytest.mark.fast
def test_spent_and_failure_status_preserve_decision(
    prepared: tuple[Path, Path],
) -> None:
    """Elapsed does not erase a coincident failure; no-work never clears usage."""
    root, script = prepared
    receipt(root, consumed=43001.0, reason="maximum_elapsed_time")
    before = inventory(root)
    plan = life.plan_start(root, cluster_script=script)
    assert plan["action"] == "no-work" and plan["remaining_seconds"] == 0
    assert plan["last_endpoint"]["reason"] == "maximum_elapsed_time"
    assert inventory(root) == before
    path = root / "resource_monitor/session_123.json"
    data = json.loads(path.read_text())
    data["forced_group_cleanup"] = True
    path.write_text(json.dumps(data))
    before = inventory(root)
    with pytest.raises(ValueError, match="failed/uncertain"):
        life.plan_start(root, cluster_script=script)
    assert inventory(root) == before


@pytest.mark.fast
def test_operator_preview_and_exec_are_distinct(prepared: tuple[Path, Path]) -> None:
    """The normal path never delegates to a prepared/sealed script."""
    root, script = prepared
    before = inventory(root)
    with patch.object(operator_cli.os, "execv", side_effect=AssertionError):
        assert (
            operator_cli.main([
                "start",
                "--work-dir",
                str(root),
                "--cluster-script",
                str(script),
                "--dry-run",
            ])
            == 0
        )
    with patch.object(operator_cli.os, "execv", return_value=0) as execute:
        assert (
            operator_cli.main([
                "start",
                "--work-dir",
                str(root),
                "--cluster-script",
                str(script),
            ])
            == 0
        )
    command = execute.call_args.args[1]
    assert life.MODULE in command and "run_bootstrap.sh" not in command
    assert inventory(root) == before


@pytest.mark.fast
def test_fresh_delegates_preparation_then_replaces_process(tmp_path: Path) -> None:
    """Configure native preparation once and exec before runtime thread setup."""
    from chipiron.environments.morpion.bootstrap import continual_operator

    root = tmp_path / "new"
    script = tmp_path / "supervisor.sh"
    script.write_text("#!/bin/bash\nexit 0\n")
    with (
        patch.object(
            continual_operator, "prepare_continual_experiment", return_value=None
        ) as prepare,
        patch.object(life.os, "execv", return_value=0) as execute,
    ):
        assert life.execute_start(root, "auto", (), script) == 0
    assert prepare.call_args.kwargs["managed_sessions"] is True
    command = execute.call_args.args[1]
    assert life.MODULE in command and command[command.index("--mode") + 1] == "resume"
    assert not root.exists()


@pytest.mark.fast
@pytest.mark.parametrize(
    "tokens",
    [["--new-session-seconds", "10"], ["--max-rows", "99"], ["--reset-models"]],
)
def test_unsupported_and_conflicting_options_write_nothing(
    prepared: tuple[Path, Path], tokens: list[str]
) -> None:
    """Reject grants, selective resets and changed scientific inputs."""
    root, script = prepared
    before = inventory(root)
    with pytest.raises(ValueError):
        life.plan_start(root, recipe=tokens, cluster_script=script)
    assert inventory(root) == before


@pytest.mark.fast
@pytest.mark.parametrize("incomplete", ["decision_124.json", "session_124.log"])
def test_incomplete_monitor_attempt_cannot_reuse_old_usage(
    prepared: tuple[Path, Path], incomplete: str
) -> None:
    """Accept native sidecars, but refuse an attempt without final cleanup/usage."""
    root, script = prepared
    receipt(root)
    monitor = root / "resource_monitor"
    (monitor / "session_123.native.json").write_text("{}")
    assert life.plan_start(root, cluster_script=script)["action"] == "resume"
    (monitor / incomplete).write_text("{}")
    before = inventory(root)
    with pytest.raises(ValueError, match="Incomplete or conflicting"):
        life.plan_start(root, cluster_script=script)
    assert inventory(root) == before
