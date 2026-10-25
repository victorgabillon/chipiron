"""Bounded foreground hosting uses the existing stages and stops their descendants."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts/launch_morpion_gnome_cluster.sh"
)


def _process_has_exited(pid: int) -> bool:
    """Read procfs atomically enough to tolerate a concurrently reaped process."""
    try:
        status = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return True
    return status.rsplit(") ", 1)[1].startswith(("Z", "X"))


@pytest.mark.parametrize("interrupt", [False, True])
def test_headless_cluster_stops_all_four_worker_groups(
    tmp_path: Path, *, interrupt: bool
) -> None:
    """Real shell supervision terminates even descendants that ignore TERM."""
    stub = tmp_path / "python_stub"
    stub.write_text("""#!/usr/bin/env python3
import json, os, signal, subprocess, sys, time
from pathlib import Path
if "--pipeline-stage" not in sys.argv:
    if "-c" in sys.argv:
        if "latest_tree_snapshot_path" in sys.argv[2]:
            print("fixed-source")
            sys.exit(0)
        sys.exit(1)
    sys.exit(0)
stage=sys.argv[sys.argv.index("--pipeline-stage")+1]
if "--reader-cache-dir" in sys.argv:
    cache=Path(sys.argv[sys.argv.index("--reader-cache-dir")+1])
    cache.mkdir(parents=True,exist_ok=True)
    (cache/"owned.sqlite3").write_bytes(b"private lifecycle fixture")
child=subprocess.Popen([sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"])
Path(os.environ["MORPION_WORK_DIR"], stage+".json").write_text(json.dumps({"pid":os.getpid(),"child":child.pid}))
time.sleep(60)
""")
    stub.chmod(0o755)
    env = dict(
        os.environ,
        MORPION_WORK_DIR=str(tmp_path),
        PYTHON_BIN=str(stub),
        MORPION_CLUSTER_HEADLESS="1",
        MORPION_CLUSTER_MAX_SECONDS="20" if interrupt else "3",
        MORPION_CLUSTER_STOP_GRACE_SECONDS="1",
        MORPION_CLUSTER_OPEN_STATUS="0",
        MORPION_CLUSTER_SHOW_RECAP="0",
        MORPION_GROWTH_SHOW_RECAP="0",
        MORPION_CLUSTER_WORKERS="all",
        MORPION_SEED_MODELS_FROM_WORK_DIR="",
    )
    process = subprocess.Popen(
        ["bash", str(_SCRIPT)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 8
        stages = ("growth", "dataset_worker", "training_worker", "reevaluation")
        while time.monotonic() < deadline and not all(
            (tmp_path / f"{stage}.json").exists() for stage in stages
        ):
            time.sleep(0.05)
        assert all((tmp_path / f"{stage}.json").exists() for stage in stages)
        if interrupt:
            process.send_signal(signal.SIGTERM)
        output, _ = process.communicate(timeout=13)
        assert process.returncode == (143 if interrupt else 0), output
        assert "all worker groups stopped" in output
        for line in output.splitlines():
            if "private_reader_cache=" in line:
                assert not Path(line.split("private_reader_cache=")[1]).exists()
        for stage in stages:
            for pid in json.loads((tmp_path / f"{stage}.json").read_text()).values():
                # Successful launcher exit itself confirms quiescence, without a
                # second test-only grace period that hides incomplete shutdown.
                assert _process_has_exited(pid), (pid, output)
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            process.communicate(timeout=13)


def test_cluster_rejects_horizon_over_twelve_hours(tmp_path: Path) -> None:
    """The cap includes the shutdown grace period, before any worker is spawned."""
    env = dict(
        os.environ,
        MORPION_CLUSTER_HEADLESS="1",
        MORPION_WORK_DIR=str(tmp_path),
        MORPION_CLUSTER_MAX_SECONDS="43200",
        MORPION_CLUSTER_STOP_GRACE_SECONDS="60",
    )
    result = subprocess.run(
        ["bash", str(_SCRIPT)], env=env, capture_output=True, text=True, timeout=5
    )
    assert result.returncode != 0
    assert "at most 43200" in result.stderr
    assert not list(tmp_path.glob("*.json"))


def test_scientific_cache_cleanup_follows_group_quiescence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the supervisor's private cache is retired after ignoring-TERM children stop."""
    monkeypatch.setenv("MORPION_SCIENTIFIC", "1")
    (tmp_path / "run_state.json").write_text(
        '{"latest_tree_snapshot_path": "fixed-source"}'
    )
    test_headless_cluster_stops_all_four_worker_groups(tmp_path, interrupt=True)
