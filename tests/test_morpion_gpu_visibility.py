"""Hardware-independent tests for the shared Morpion GPU launcher bindings."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
VISIBILITY_SCRIPT = SCRIPT_DIR / "morpion_gpu_visibility.sh"


def _resolve(
    tmp_path: Path,
    *,
    gpu_count: str,
    training: str | None = None,
    reevaluation: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the Bash resolver against a fake backend reporting a GPU count."""
    python_stub = tmp_path / "fake_python"
    python_stub.write_text(
        '#!/bin/sh\nprintf "%s\\n" "${FAKE_GPU_COUNT:-0}"\n',
        encoding="utf-8",
    )
    python_stub.chmod(0o755)
    env = os.environ.copy()
    env["PYTHON_BIN"] = str(python_stub)
    env["FAKE_GPU_COUNT"] = gpu_count
    env.pop("MORPION_TRAINING_CUDA_VISIBLE_DEVICES", None)
    env.pop("MORPION_REEVALUATION_CUDA_VISIBLE_DEVICES", None)
    if training is not None:
        env["MORPION_TRAINING_CUDA_VISIBLE_DEVICES"] = training
    if reevaluation is not None:
        env["MORPION_REEVALUATION_CUDA_VISIBLE_DEVICES"] = reevaluation
    return subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; morpion_resolve_gpu_visibility || exit $?; '
            'printf "%s|%s\\n" "$MORPION_TRAINING_CUDA_VISIBLE_DEVICES" '
            '"$MORPION_REEVALUATION_CUDA_VISIBLE_DEVICES"',
            "--",
            str(VISIBILITY_SCRIPT),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


@pytest.mark.parametrize(
    ("count", "training", "reevaluation", "expected"),
    [
        ("0", None, None, "|"),
        ("1", None, None, "0|0"),
        ("2", None, None, "0|1"),
        ("4", None, None, "0|1"),
        ("invalid", None, None, "|"),
        ("2", "1", "cpu", "1|"),
        ("1", "cpu", "0", "|0"),
        ("1", "0,1", "cpu", "0,1|"),
        ("1", "", "", "|"),
    ],
)
def test_gpu_visibility(
    tmp_path: Path,
    count: str,
    training: str | None,
    reevaluation: str | None,
    expected: str,
) -> None:
    """Auto mode handles CPU/one/two GPUs; overrides take priority."""
    result = _resolve(
        tmp_path, gpu_count=count, training=training, reevaluation=reevaluation
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected


@pytest.mark.parametrize("invalid", ["0;echo bad", "GPU0", "0 1"])
def test_gpu_visibility_rejects_bad_assignments(tmp_path: Path, invalid: str) -> None:
    """Reject shell metacharacters and unsupported selectors before launching."""
    result = _resolve(tmp_path, gpu_count="2", reevaluation=invalid)
    assert result.returncode == 2
    assert "Invalid Morpion CUDA_VISIBLE_DEVICES" in result.stderr


@pytest.mark.parametrize(
    "name",
    [
        "morpion_gpu_visibility.sh",
        "launch_morpion_tmux_cluster.sh",
        "launch_morpion_gnome_cluster.sh",
    ],
)
def test_morpion_launchers_have_valid_bash_syntax(name: str) -> None:
    """The generated commands must remain valid Bash scripts."""
    subprocess.run(["bash", "-n", str(SCRIPT_DIR / name)], check=True)


@pytest.mark.parametrize(
    "name",
    ["launch_morpion_tmux_cluster.sh", "launch_morpion_gnome_cluster.sh"],
)
def test_morpion_launchers_use_configurable_visibility(name: str) -> None:
    """Both launchers must source the resolver and use both worker bindings."""
    source = (SCRIPT_DIR / name).read_text(encoding="utf-8")
    assert 'source "$SCRIPT_DIR/morpion_gpu_visibility.sh"' in source
    assert "CUDA_VISIBLE_DEVICES=$MORPION_TRAINING_CUDA_VISIBLE_DEVICES" in source
    assert "CUDA_VISIBLE_DEVICES=$MORPION_REEVALUATION_CUDA_VISIBLE_DEVICES" in source
