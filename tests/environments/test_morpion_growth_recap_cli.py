"""The recap's Parsley adoption keeps the CLI and inspected files read-only."""

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from chipiron.environments.morpion.bootstrap import growth_recap
from chipiron.environments.morpion.bootstrap.run_state import (
    MorpionBootstrapRunState,
    save_bootstrap_run_state,
)


def _snapshot(root: Path) -> dict[str, tuple[int, int, str | None]]:
    """Record content, modes and modification times, including directories."""
    return {
        str(path.relative_to(root)): (
            path.stat().st_mode,
            path.stat().st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
        )
        for path in [root, *sorted(root.rglob("*"))]
    }


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Create a small real state using the unchanged artifact writer."""
    work = tmp_path / "inspected"
    work.mkdir()
    save_bootstrap_run_state(
        MorpionBootstrapRunState(
            generation=7,
            cycle_index=9,
            latest_tree_snapshot_path=None,
            latest_rows_path=None,
            latest_model_bundle_paths=None,
            active_evaluator_name="mlp_41",
            tree_size_at_last_save=123,
            last_save_unix_s=0.0,
        ),
        work / "run_state.json",
    )
    (work / "sentinel.bin").write_bytes(b"unchanged\x00artifact")
    return work


@pytest.mark.parametrize(
    ("options", "code", "message"),
    [
        (["--help"], 0, "--work-dir"),
        ([], 2, "--work-dir"),
        (["--work-dir"], 2, "expected one argument"),
        (["--work-dir", "{work}", "--worker-max-cycles"], 2, "expected one argument"),
        (
            ["--work-dir", "{work}", "--unknown-option"],
            2,
            "unrecognized arguments: --unknown-option",
        ),
        (["--work-dir", "{work}", "--no-rich"], 0, "latest_generation=7 cycle=9"),
        (
            ["--work-dir", "{work}", "--rich", "--no-rich"],
            0,
            "latest_generation=7 cycle=9",
        ),
        (
            ["--work-dir", "{work}", "--no-rich", "--rich"],
            0,
            "latest_generation=7 cycle=9",
        ),
    ],
)
def test_recap_subprocess_never_modifies_inspected_workspace(
    workspace: Path, options: list[str], code: int, message: str
) -> None:
    """Help, usage failures and real recap execution only read the workspace."""
    # Enforce read-only access as well as checking bytes and directory metadata.
    for path in workspace.iterdir():
        path.chmod(0o444)
    workspace.chmod(0o555)
    before = _snapshot(workspace.parent)
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                growth_recap.__name__,
                *[arg.replace("{work}", str(workspace)) for arg in options],
            ],
            cwd=workspace,
            env={
                **os.environ,
                "PYTHONDONTWRITEBYTECODE": "1",
                "MORPION_GROWTH_WORKER_MAX_CYCLES": "17",
            },
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == code, result.stderr
        assert message in (result.stdout if code == 0 else result.stderr)
        assert _snapshot(workspace.parent) == before
        if "--no-rich" in options:
            expected = growth_recap.render_growth_recap(
                growth_recap.collect_growth_recap(workspace, worker_max_cycles="17"),
                force_plain=True,
            )
            assert result.stdout == expected + "\n"
            assert result.stderr == ""
    finally:
        workspace.chmod(0o755)
        for path in workspace.iterdir():
            path.chmod(0o644)


def test_explicit_argv_path_and_invocation_environment(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Explicit argv ignores process arguments and samples the environment per call."""
    original = growth_recap.collect_growth_recap
    received: list[tuple[Path, str | None]] = []

    def collect(work_dir: Path, *, worker_max_cycles: str | None = None) -> Any:
        assert isinstance(work_dir, Path)
        received.append((work_dir, worker_max_cycles))
        return original(work_dir, worker_max_cycles=worker_max_cycles)

    monkeypatch.setattr(growth_recap, "collect_growth_recap", collect)
    monkeypatch.setattr(sys, "argv", ["host", "--unrelated-host-option"])
    before = _snapshot(workspace)
    args = ["--work-dir", str(workspace), "--no-rich"]
    for value in ("3", "8"):
        monkeypatch.setenv("MORPION_GROWTH_WORKER_MAX_CYCLES", value)
        assert growth_recap.main(args) == 0
        assert f"worker_max_cycles={value}" in capsys.readouterr().out
    assert growth_recap.main([*args, "--worker-max-cycles", "0"]) == 0
    assert "worker_max_cycles=0" in capsys.readouterr().out
    monkeypatch.delenv("MORPION_GROWTH_WORKER_MAX_CYCLES")
    assert growth_recap.main(args) == 0
    assert "worker_max_cycles=n/a" in capsys.readouterr().out
    assert received == [(workspace, value) for value in ("3", "8", "0", None)]
    assert _snapshot(workspace) == before


def test_missing_workspace_is_not_created(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tolerant recap of an absent workspace must not create output directories."""
    monkeypatch.delenv("MORPION_GROWTH_WORKER_MAX_CYCLES", raising=False)
    missing = tmp_path / "absent"
    before = _snapshot(tmp_path)
    assert growth_recap.main(["--work-dir", str(missing), "--no-rich"]) == 0
    assert "latest_generation=n/a" in capsys.readouterr().out
    assert _snapshot(tmp_path) == before


def test_none_argv_reads_process_arguments(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The standard module entry point still consumes sys.argv."""
    monkeypatch.setattr(
        sys, "argv", ["recap", "--work-dir", str(workspace), "--no-rich"]
    )
    assert growth_recap.main() == 0
    assert "latest_generation=7 cycle=9" in capsys.readouterr().out


def test_rich_flag_reaches_existing_renderer(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The rich-only switch keeps its existing renderer contract."""
    received: list[tuple[bool, bool]] = []

    def render(
        recap: growth_recap.MorpionGrowthRecap,
        *,
        force_plain: bool = False,
        force_rich: bool = False,
    ) -> str:
        assert recap.latest_generation == 7
        received.append((force_plain, force_rich))
        return "rich recap"

    monkeypatch.setattr(growth_recap, "render_growth_recap", render)
    assert growth_recap.main(["--work-dir", str(workspace), "--rich"]) == 0
    assert capsys.readouterr().out == "rich recap\n"
    assert received == [(False, True)]
