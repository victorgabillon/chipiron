"""Runtime path helpers for Chipiron.

This module provides paths for user-writable output directories.
"""

import os
from pathlib import Path

from platformdirs import user_data_dir


def assert_test_output_path(path: str | Path) -> None:
    """Refuse known research destinations under the repository test configuration."""
    if not os.environ.get("CHIPIRON_TEST_OUTPUT_ROOT"):
        return
    resolved = Path(path).expanduser().resolve()
    protected = (
        Path.home() / "morpion_runs",
        Path.home() / ".cache" / "chipiron",
        Path(user_data_dir("chipiron")),
    )
    if any(resolved.is_relative_to(root.resolve()) for root in protected):
        message = f"Protected persistence destination selected during tests: {path}"
        raise ValueError(message)


def output_root_path_str() -> str:
    """Return the root output path for Chipiron runtime outputs.

    The path can be overridden by setting CHIPIRON_OUTPUT_DIR. If not set,
    it defaults to a user-specific data directory.
    """
    override = os.environ.get("CHIPIRON_OUTPUT_DIR")
    if override:
        return str(Path(override))
    return str(Path(user_data_dir("chipiron")))


def get_output_root(*, ensure_exists: bool = True) -> Path:
    """Root directory for chipiron runtime outputs (persistent, writable)."""
    override = os.environ.get("CHIPIRON_OUTPUT_DIR")
    p = Path(override) if override else Path(user_data_dir("chipiron"))
    assert_test_output_path(p)
    if ensure_exists:
        p.mkdir(parents=True, exist_ok=True)
    return p


def get_default_output_dir(*, ensure_exists: bool = True) -> Path:
    """Return the default output directory for scripts (persistent, writable)."""
    p = get_output_root(ensure_exists=ensure_exists) / "default_output_folder"
    if ensure_exists:
        p.mkdir(parents=True, exist_ok=True)
    return p
