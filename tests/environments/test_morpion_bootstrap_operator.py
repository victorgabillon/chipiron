"""Operator resolution, foreground launch safety and truthful lightweight summaries."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from chipiron.environments.morpion.bootstrap import operator_cli as cli
from chipiron.environments.morpion.bootstrap.dashboard import view_model as vm
from chipiron.environments.morpion.bootstrap.dashboard.navigation import (
    refresh_seconds,
    selected_page,
)

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.monkeypatch import MonkeyPatch


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: MonkeyPatch) -> Path:
    """Every preference and run artifact lives under the test's temporary root."""
    monkeypatch.setattr(
        cli, "current_run_config_path", lambda: tmp_path / "preferences/current.json"
    )
    monkeypatch.delenv("MORPION_WORK_DIR", raising=False)
    path = tmp_path / "run with spaces"
    path.mkdir()
    (path / "bootstrap_config.json").write_text('{"pipeline_mode":"artifact_pipeline"}')
    monkeypatch.chdir(tmp_path)
    return path


def test_resolution_precedence_and_atomic_selection(
    workspace: Path, tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Explicit/env/saved/cwd order never silently falls through an invalid choice."""
    other = tmp_path / "other"
    other.mkdir()
    (other / "bootstrap_config.json").write_text("{}")
    cli.save_current_run(workspace)
    assert cli.resolve_work_dir() == workspace
    assert json.loads(cli.current_run_config_path().read_text()) == {
        "version": 1,
        "work_dir": str(workspace),
    }
    assert list(cli.current_run_config_path().parent.iterdir()) == [
        cli.current_run_config_path()
    ]
    monkeypatch.setenv("MORPION_WORK_DIR", str(other))
    assert cli.resolve_work_dir() == other
    assert cli.resolve_work_dir(workspace) == workspace
    monkeypatch.setenv("MORPION_WORK_DIR", str(tmp_path / "absent"))
    with pytest.raises(cli.BootstrapOperatorError, match="Not a bootstrap"):
        cli.resolve_work_dir()
    monkeypatch.delenv("MORPION_WORK_DIR")
    cli.current_run_config_path().unlink()
    monkeypatch.chdir(workspace)
    assert cli.resolve_work_dir() == workspace
    monkeypatch.chdir(tmp_path)
    with pytest.raises(cli.BootstrapOperatorError, match="Choose a run"):
        cli.resolve_work_dir()


@pytest.mark.parametrize(
    "payload",
    [
        "{",
        "[]",
        '{"version":2,"work_dir":"/tmp"}',
        '{"version":1,"work_dir":"relative"}',
    ],
)
def test_invalid_saved_preference_is_actionable(workspace: Path, payload: str) -> None:
    """Broken preferences are reported rather than quietly choosing another run."""
    config = cli.current_run_config_path()
    config.parent.mkdir()
    config.write_text(payload)
    with pytest.raises(cli.BootstrapOperatorError, match="config is invalid"):
        cli.resolve_work_dir()


def test_zero_argument_dashboard_and_installed_command(
    workspace: Path, monkeypatch: MonkeyPatch
) -> None:
    """Default opening uses this interpreter and a real packaged entry script."""
    cli.save_current_run(workspace)
    commands = []
    monkeypatch.setattr(
        cli.os,
        "execv",
        lambda executable, command: commands.append((executable, command)),
    )
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: object())
    assert cli.main([]) is None  # Only the mocked exec returns.
    executable, command = commands[0]
    assert executable == sys.executable
    assert command[:4] == (sys.executable, "-m", "streamlit", "run")
    assert Path(command[4]).is_file()
    assert command[-3:] == ("--", "--work-dir", str(workspace))
    assert "--server.headless=false" in command
    assert (
        cli.main([
            "dashboard",
            "--work-dir",
            str(workspace),
            "--port",
            "8619",
            "--no-browser",
        ])
        is None
    )
    assert "--server.port=8619" in commands[-1][1]
    assert "--server.headless=true" in commands[-1][1]
    assert "--server.address=127.0.0.1" in commands[-1][1]


def test_missing_extra_has_install_help(
    workspace: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    """Missing Streamlit is an actionable optional-dependency error."""
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: None)
    assert cli.main(["--work-dir", str(workspace)]) == 2
    assert "pip install 'chipiron[dashboard]'" in capsys.readouterr().err


def test_current_use_status_and_prepared_dry_run(
    workspace: Path, capsys: CaptureFixture[str], monkeypatch: MonkeyPatch
) -> None:
    """Dry-run resolves the canonical script without executing even its preflight."""
    (workspace / "run_bootstrap.sh").write_text("echo NEVER_EXECUTE")
    (workspace / "manifest.json").write_text(
        json.dumps({
            "status": "READY_TO_LAUNCH_CANONICAL_BOOTSTRAP",
            "scientific_configuration": {"runtime_limits": {"active_seconds": 42900}},
        })
    )

    def forbidden(*args: object) -> None:
        pytest.fail("Dry run must never execute a launcher")

    monkeypatch.setattr(cli.os, "execvp", forbidden)
    assert cli.main(["use", str(workspace)]) == 0
    assert cli.main(["current"]) == 0
    assert cli.main(["status"]) == 0
    assert cli.main(["run", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "Status: prepared" in output
    assert "11h55" in output
    assert str(workspace / "run_bootstrap.sh") in output
    assert "nothing launched" in output


def test_missing_launcher_fails_closed(
    workspace: Path, capsys: CaptureFixture[str]
) -> None:
    """Never synthesize a scientific experiment from defaults."""
    assert cli.main(["run", "--work-dir", str(workspace), "--dry-run"]) == 2
    assert "No prepared run_bootstrap.sh" in capsys.readouterr().err


def test_noninteractive_launch_requires_confirmation(
    workspace: Path, monkeypatch: MonkeyPatch
) -> None:
    """Redirected stdin cannot silently launch a prepared experiment."""
    (workspace / "run_bootstrap.sh").write_text("exit 0")
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    assert cli.main(["run", "--work-dir", str(workspace)]) == 2


def test_foreground_exec_delegates_to_fake_launcher(
    workspace: Path, monkeypatch: MonkeyPatch
) -> None:
    """A deliberately confirmed fake script becomes the foreground process."""
    (workspace / "run_bootstrap.sh").write_text("exit 7")
    calls = []
    monkeypatch.setattr(
        cli.os,
        "execvp",
        lambda executable, args: calls.append((executable, args, Path.cwd())),
    )
    assert cli.main(["run", "--work-dir", str(workspace), "--yes"]) is None
    assert calls == [("bash", ("bash", str(workspace / "run_bootstrap.sh")), workspace)]


def test_status_does_not_import_streamlit_or_torch(workspace: Path) -> None:
    """A fresh status process remains lightweight and read-only."""
    code = "from pathlib import Path; from chipiron.environments.morpion.bootstrap.dashboard.view_model import load_dashboard_snapshot; import sys; load_dashboard_snapshot(Path(sys.argv[1])); assert 'streamlit' not in sys.modules; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code, str(workspace)], check=True, timeout=15)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (None, "unknown"),
        ({"exit_code": 0, "finished_at_utc": "2026-01-01T10:00:00Z"}, "finished"),
        ({"exit_code": 143, "finished_at_utc": "2026-01-01T10:00:00Z"}, "stopped"),
        ({"exit_code": 1, "finished_at_utc": "2026-01-01T10:00:00Z"}, "failed"),
    ],
)
def test_lifecycle_is_evidence_based(
    workspace: Path, payload: dict[str, object] | None, expected: str
) -> None:
    """Old files alone never mean a healthy running experiment."""
    if payload is not None:
        (workspace / "launch_finished.json").write_text(json.dumps(payload))
    snapshot = vm.load_dashboard_snapshot(workspace)
    assert snapshot.status == expected
    assert all(worker.state == "unknown" for worker in snapshot.workers)


def test_stale_finished_and_unrelated_pid_are_not_live(workspace: Path) -> None:
    """PID reuse and an older finish event cannot give a false current status."""
    (workspace / "launch_finished.json").write_text(
        '{"exit_code":0,"finished_at_utc":"2026-01-01T10:00:00Z"}'
    )
    (workspace / "launch_started.json").write_text(
        '{"started_at_utc":"2026-01-02T10:00:00Z"}'
    )
    (workspace / "launcher_process_state.json").write_text(
        json.dumps({"pid": os.getpid()})
    )
    assert vm.load_dashboard_snapshot(workspace).status == "unknown"


def test_corrupt_artifacts_visible_and_uncertified_values_withheld(
    workspace: Path,
) -> None:
    """Malformed status is distinct from missing data; frontier is never promoted."""
    (workspace / "pipeline").mkdir()
    (workspace / "pipeline/active_model.json").write_text("[]")
    (workspace / "latest_status.json").write_text(
        json.dumps({
            "latest_event": {
                "record": {
                    "current_best_total_points": 130,
                    "current_best_is_exact": False,
                    "current_best_is_terminal": False,
                },
                "frontier": {"current_best_total_points": 140},
            }
        })
    )
    snapshot = vm.load_dashboard_snapshot(workspace)
    assert snapshot.errors and snapshot.record == {}
    assert snapshot.frontier["current_best_total_points"] == 140
    assert "expected a JSON object" in snapshot.errors[0]


def test_pipeline_metrics_and_model_provenance(workspace: Path) -> None:
    """Active external-seed provenance stays separate from later trained metrics."""
    p = workspace / "pipeline/generation_000004"
    p.mkdir(parents=True)
    (p / "dataset_status.json").write_text(
        '{"generation":4,"metadata":{"dataset_rows":1234},"record_status":{"current_best_total_points":99,"current_best_is_exact":true}}'
    )
    (p / "training_status.json").write_text(
        '{"generation":4,"status":"done","updated_at_utc":"2026-01-01T10:00:00Z","evaluator_results":{"tiny":{"validation_loss":3.2}}}'
    )
    (workspace / "pipeline/active_model.json").write_text(
        '{"evaluator_name":"seed","source":"external_seed","generation":0,"model_bundle_path":"models/seed"}'
    )
    model = workspace / "models/seed"
    model.mkdir(parents=True)
    (model / "morpion_manifest.json").write_text(
        '{"input_representation":"entity_tokens","metadata":{"parameter_count":106049,"training_config":{"train_count":80000,"validation_count":20000,"config":{"num_epochs":20,"learning_rate":0.001}}}}'
    )
    (model / "morpion_regressor_args.json").write_text(
        '{"model_kind":"transformer","relation_bias_scale":0.25}'
    )
    snapshot = vm.load_dashboard_snapshot(workspace)
    summary = vm.active_model_summary(snapshot)
    assert snapshot.dataset_rows == 1234
    assert snapshot.evaluator_name == "seed"
    assert snapshot.record["current_best_total_points"] == 99
    assert summary["Parameter count"] == 106049 and summary["Epochs"] == 20
    assert summary["Training rows"] == 80000 and summary["Validation rows"] == 20000
    assert len(snapshot.training_history) == 1


@pytest.mark.parametrize(
    ("page", "interval"),
    [
        ("Overview", 15),
        ("Tree", None),
        ("Evaluator", None),
        ("Operations", None),
        ("Record", None),
    ],
)
def test_navigation_refresh_defaults(page: str, interval: int | None) -> None:
    """Unknown navigation is safe and heavy views do not auto-refresh by default."""
    assert selected_page(page) == page
    assert selected_page("removed-view") == "Overview"
    assert refresh_seconds(page) == interval
    assert refresh_seconds(page, "5s") == 5
    assert refresh_seconds(page, "Off") is None


@pytest.mark.parametrize(
    ("script", "expected"), [("exit 7", 7), ("kill -TERM $$", -15)]
)
def test_fake_prepared_launcher_preserves_exit_and_signal(
    workspace: Path, script: str, expected: int
) -> None:
    """Exec preserves real exit and termination semantics for a harmless fake script."""
    (workspace / "run_bootstrap.sh").write_text(script)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            cli.__name__,
            "run",
            "--work-dir",
            str(workspace),
            "--yes",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == expected
    assert "Command:" in result.stdout


@pytest.mark.parametrize("stopped", ["broken", "2026-01-01T10:00:00Z"])
def test_old_or_malformed_stop_does_not_override_a_new_launch(
    workspace: Path, stopped: str
) -> None:
    """Historical single-process metadata cannot certify the current launch as stopped."""
    status, _ = vm.classify_run(
        workspace,
        manifest={},
        started={"started_at_utc": "2026-01-02T10:00:00Z"},
        finished={},
        process={"stopped_at_utc": stopped},
        has_progress=True,
    )
    assert status == "unknown"
