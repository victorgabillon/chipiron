"""Shell preflight must refuse unsafe test roots before any startup side effects."""

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast


@pytest.mark.parametrize("launcher", ["gnome", "tmux"])
@pytest.mark.parametrize(
    "destination", ["default", "protected", "alias", "private", "production"]
)
def test_shell_destination_preflight(
    tmp_path: Path, launcher: str, destination: str
) -> None:
    """Execute the real startup prefix, excluding all commands capable of writing.

    Even a broken guard cannot reach a historical file in this test. Existing
    real-process launcher lifetime tests exercise the allowed private path fully.
    """
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / f"launch_morpion_{launcher}_cluster.sh"
    )
    source = script.read_text()
    marker = "# Test destination guard ends before any startup side effects."
    assert marker in source
    prefix = source.split(marker)[0]
    assert "mkdir" not in prefix and "mktemp" not in prefix
    environment = dict(os.environ)
    protected = Path.home() / "oldata" / "victor" / "morpion_runs" / "guard-test"
    if destination == "default":
        environment.pop("MORPION_WORK_DIR", None)
    elif destination == "private":
        environment["MORPION_WORK_DIR"] = str(tmp_path)
    elif destination == "alias":
        alias = tmp_path / "alias"
        alias.symlink_to(protected, target_is_directory=True)
        environment["MORPION_WORK_DIR"] = str(alias)
    else:
        environment["MORPION_WORK_DIR"] = str(protected)
    if destination == "production":
        environment.pop("CHIPIRON_TEST_OUTPUT_ROOT", None)
    completed = subprocess.run(
        ["bash", "-c", prefix],
        env=environment,
        capture_output=True,
        text=True,
        timeout=5,
    )
    allowed = destination in {"private", "production"}
    assert (completed.returncode == 0) == allowed, completed.stdout + completed.stderr
    if not allowed:
        assert "Protected persistence destination" in completed.stderr
