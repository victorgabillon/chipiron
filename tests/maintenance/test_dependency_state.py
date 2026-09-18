"""Dependency records distinguish immutable installs from dirty local sources."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

import pytest


@pytest.fixture
def recorder() -> ModuleType:
    """Load the standalone recorder without changing the process import path."""
    path = Path(__file__).resolve().parents[2] / "scripts/record_dependency_state.py"
    spec = importlib.util.spec_from_file_location("dependency_recorder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_direct_url_record_does_not_disclose_credentials(recorder: ModuleType) -> None:
    """Installer URLs may contain credentials; keep only safe repository identity."""
    record = recorder.sanitized_direct_url(
        json.dumps({
            "url": "https://user:password@example.org/repo?token=secret#credential",
            "vcs_info": {"commit_id": "a" * 40},
        })
    )
    assert record == {
        "url": "https://example.org/repo",
        "vcs_info": {"commit_id": "a" * 40},
    }
    assert "secret" not in json.dumps(record)


def test_checkout_record_identifies_uncommitted_source(
    recorder: ModuleType, tmp_path: Path
) -> None:
    """A commit ID alone must not describe a locally modified source checkout."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    source = tmp_path / "module.py"
    source.write_text("VALUE = 1\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "module.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    clean = recorder.git_state(source)
    assert clean["commit"] and not clean["dirty"]
    source.write_text("VALUE = 2\n")
    dirty = recorder.git_state(source)
    assert dirty["commit"] == clean["commit"] and dirty["dirty"]
    assert (
        dirty["changed_files_sha256"]["module.py"]
        == hashlib.sha256(source.read_bytes()).hexdigest()
    )


def test_require_clean_rejects_dirty_or_missing_dependencies(
    recorder: ModuleType,
) -> None:
    """The opt-in gate must not call a dirty or missing dependency reproducible."""
    source: dict[str, object] = {"commit": "a" * 40, "dirty": False}
    package: dict[str, object] = {
        "version": "1.0",
        "import_origin": "/lib/example.py",
        "source_git": source,
    }
    record = {"chipiron_checkout": {"dirty": False}, "packages": {"example": package}}
    assert not recorder.has_unpinned_sources(record)
    source["dirty"] = True
    assert recorder.has_unpinned_sources(record)
    source["dirty"] = False
    package["version"] = None
    assert recorder.has_unpinned_sources(record)
