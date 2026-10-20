"""Publication errors and byte parity through both Reevaluation reader paths."""

from __future__ import annotations

import errno
import json
import os
import stat
from pathlib import Path

import pytest

import chipiron.environments.morpion.bootstrap.reevaluation_publication as publication
import chipiron.environments.morpion.bootstrap.reevaluation_worker as worker
from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
    MorpionReevaluationPatch,
    load_reevaluation_patch,
    reevaluation_patch_to_dict,
)
from tests.environments.test_morpion_streaming_reevaluation import (
    run_once,
    setup_worker,
)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("fault", ["file_fsync", "directory_fsync", "cleanup"])
def test_worker_never_advances_cursor_on_publication_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    streaming: bool,
    fault: str,
) -> None:
    """A post-link error keeps a complete pending patch and the old cursor on retry."""
    paths = setup_worker(tmp_path)
    before = paths.pipeline_reevaluation_cursor_path.read_bytes()
    original_fsync, original_unlink = os.fsync, Path.unlink

    def sync(descriptor: int) -> None:
        directory = stat.S_ISDIR(os.fstat(descriptor).st_mode)
        if (fault == "file_fsync" and not directory) or (
            fault == "directory_fsync" and directory
        ):
            raise OSError(errno.EIO, "injected publication fsync failure")
        original_fsync(descriptor)

    def unlink(temporary: Path, *, missing_ok: bool = False) -> None:
        if fault == "cleanup" and temporary.name.startswith(
            ".reevaluation_patch.json."
        ):
            raise OSError(errno.EIO, "injected publication cleanup failure")
        original_unlink(temporary, missing_ok=missing_ok)

    with monkeypatch.context() as injection:
        injection.setattr(publication.os, "fsync", sync)
        injection.setattr(Path, "unlink", unlink)
        with pytest.raises(publication.ReevaluationPatchPublicationError):
            run_once(paths, streaming=streaming, use_snapshot_value_fallback=True)
    assert paths.pipeline_reevaluation_cursor_path.read_bytes() == before
    published = fault != "file_fsync"
    assert paths.pipeline_reevaluation_patch_path.exists() == published
    if published:
        assert (
            load_reevaluation_patch(paths.pipeline_reevaluation_patch_path).patch_id
            == "fixed-patch"
        )
    retried = run_once(paths, streaming=streaming, use_snapshot_value_fallback=True)
    if published:
        assert not retried.patch_written
        assert retried.reason == "pending_patch_exists"
        assert paths.pipeline_reevaluation_cursor_path.read_bytes() == before
    else:
        assert retried.patch_written
        assert paths.pipeline_reevaluation_cursor_path.read_bytes() != before


@pytest.mark.parametrize("streaming", [False, True])
def test_worker_patch_and_cursor_bytes_match_previous_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, streaming: bool
) -> None:
    """Only the publication mechanism changes; every patch/cursor byte is identical."""
    old = setup_worker(tmp_path / "old")
    new = setup_worker(tmp_path / "new")

    def previous_writer(patch: MorpionReevaluationPatch, path: Path) -> bool:
        try:
            with path.open("x", encoding="utf-8") as handle:
                json.dump(
                    reevaluation_patch_to_dict(patch), handle, indent=2, sort_keys=True
                )
                handle.write("\n")
        except FileExistsError:
            return False
        return True

    with monkeypatch.context() as legacy:
        legacy.setattr(worker, "_save_reevaluation_patch_exclusive", previous_writer)
        expected = run_once(old, streaming=streaming, use_snapshot_value_fallback=True)
    actual = run_once(new, streaming=streaming, use_snapshot_value_fallback=True)
    assert actual == expected
    assert (
        new.pipeline_reevaluation_patch_path.read_bytes()
        == old.pipeline_reevaluation_patch_path.read_bytes()
    )
    assert (
        new.pipeline_reevaluation_cursor_path.read_bytes()
        == old.pipeline_reevaluation_cursor_path.read_bytes()
    )
