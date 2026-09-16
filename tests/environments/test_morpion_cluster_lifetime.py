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
    if "-c" in sys.argv: sys.exit(1)
    sys.exit(0)
stage=sys.argv[sys.argv.index("--pipeline-stage")+1]
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
        output, _ = process.communicate(timeout=8)
        assert process.returncode == (143 if interrupt else 0), output
        for stage in stages:
            for pid in json.loads((tmp_path / f"{stage}.json").read_text()).values():
                stat = Path(f"/proc/{pid}/stat")
                # SIGKILL delivery is asynchronous for grandchildren we cannot waitpid.
                exit_deadline = time.monotonic() + 1
                while stat.exists() and time.monotonic() < exit_deadline:
                    if stat.read_text().split(") ")[1].startswith("Z"):
                        break
                    time.sleep(0.01)
                assert not stat.exists() or stat.read_text().split(") ")[1].startswith(
                    "Z"
                ), (pid, output)
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            process.communicate(timeout=8)


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
