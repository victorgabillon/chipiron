"""Deterministic publication faults and competing processes in private directories."""

import errno
import io
import json
import os
import selectors
import stat
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

import chipiron.environments.morpion.bootstrap.reevaluation_publication as publication
from chipiron.environments.morpion.bootstrap.pipeline_artifacts import (
    MorpionReevaluationPatch,
    MorpionReevaluationPatchRow,
    load_reevaluation_patch,
    reevaluation_patch_to_dict,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import BinaryIO


@pytest.fixture
def patch() -> MorpionReevaluationPatch:
    """Include Unicode, ordering, optional values and nested scientific metadata."""
    return MorpionReevaluationPatch(
        patch_id="patch-é",
        created_at_utc="2026-10-20T12:00:00Z",
        evaluator_generation=2,
        evaluator_name="model-é",
        model_bundle_path="models/generation_000002/model-é",
        tree_generation=7,
        start_cursor="z",
        end_cursor="a",
        rows=(
            MorpionReevaluationPatchRow(node_id="z", direct_value=-0.0),
            MorpionReevaluationPatchRow(
                node_id="a",
                direct_value=1.25,
                backed_up_value=2.5,
                is_terminal=True,
                is_exact=True,
                metadata={"value": [1, "é", None]},
            ),
        ),
        metadata={"next_node_cursor": "b", "completed_full_pass": True},
    )


def legacy_bytes(patch: MorpionReevaluationPatch) -> bytes:
    """Use the previous producer's actual text-stream serialization as oracle."""
    stream = io.StringIO()
    json.dump(reevaluation_patch_to_dict(patch), stream, indent=2, sort_keys=True)
    stream.write("\n")
    return stream.getvalue().encode("utf-8")


def owned_temporaries(path: Path) -> list[Path]:
    """Select test observations only; production never sweeps other attempts."""
    return list(path.parent.glob(f".{path.name}.*.tmp"))


def test_exact_bytes_and_durability_order(
    tmp_path: Path, patch: MorpionReevaluationPatch, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The final link follows file fsync and precedes two directory barriers."""
    path = tmp_path / "patch.json"
    events: list[str] = []
    original_fsync, original_link = os.fsync, os.link

    def sync(descriptor: int) -> None:
        events.append(
            "directory" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "file"
        )
        original_fsync(descriptor)

    def link(source: Path, destination: Path) -> None:
        assert source.parent == destination.parent
        assert source.read_bytes() == legacy_bytes(patch)
        assert not destination.exists()
        events.append("link")
        original_link(source, destination)

    monkeypatch.setattr(publication.os, "fsync", sync)
    monkeypatch.setattr(publication.os, "link", link)
    assert publication.publish_reevaluation_patch_if_absent(patch, path)
    assert path.read_bytes() == legacy_bytes(patch)
    assert load_reevaluation_patch(path) == patch
    assert events == ["file", "link", "directory", "directory"]
    assert not owned_temporaries(path)


@pytest.mark.parametrize("entry", ["valid", "partial", "symlink", "directory"])
def test_existing_entry_is_never_overwritten(
    tmp_path: Path, patch: MorpionReevaluationPatch, entry: str
) -> None:
    """Every existing final entry wins, even a legacy malformed pending patch."""
    path = tmp_path / "patch.json"
    if entry == "directory":
        path.mkdir()
    elif entry == "symlink":
        path.symlink_to(tmp_path / "missing")
    else:
        path.write_bytes(legacy_bytes(patch) if entry == "valid" else b'{"partial":')
    before = path.lstat()
    assert not publication.publish_reevaluation_patch_if_absent(
        replace(patch, patch_id="loser"), path
    )
    assert path.lstat() == before
    if entry in {"valid", "partial"}:
        assert path.read_bytes() == (
            legacy_bytes(patch) if entry == "valid" else b'{"partial":'
        )
    assert not owned_temporaries(path)


@pytest.mark.parametrize("fault", ["raise", "short", "invalid", "changed"])
def test_temporary_write_failure_never_publishes(
    tmp_path: Path,
    patch: MorpionReevaluationPatch,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    """Short, corrupt, different-valid and interrupted temporary writes are rejected."""
    path = tmp_path / "patch.json"
    original = publication.tempfile.NamedTemporaryFile

    @contextmanager
    def temporary(**kwargs: Any) -> Iterator[BinaryIO]:
        with original(**kwargs) as handle:
            write = handle.write

            def faulty_write(contents: bytes) -> int:
                if fault == "raise":
                    write(contents[:10])
                    raise OSError(errno.ENOSPC, "injected interrupted write")
                altered = {
                    "short": contents[:10],
                    "invalid": b"{",
                    "changed": b"{}",
                }[fault]
                return write(altered)

            monkeypatch.setattr(handle, "write", faulty_write)
            yield handle

    monkeypatch.setattr(publication.tempfile, "NamedTemporaryFile", temporary)
    with pytest.raises((publication.ReevaluationPatchPublicationError, ValueError)):
        publication.publish_reevaluation_patch_if_absent(patch, path)
    assert not path.exists()
    assert not owned_temporaries(path)


@pytest.mark.parametrize(
    "fault", ["file_fsync", "link", "directory_fsync", "cleanup", "cleanup_fsync"]
)
def test_io_failure_inspects_and_preserves_final_state(
    tmp_path: Path,
    patch: MorpionReevaluationPatch,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    """Post-link errors preserve complete output; no error implies a rollback."""
    path = tmp_path / "patch.json"
    original_fsync, original_unlink = os.fsync, Path.unlink
    directory_syncs = 0

    def sync(descriptor: int) -> None:
        nonlocal directory_syncs
        directory = stat.S_ISDIR(os.fstat(descriptor).st_mode)
        directory_syncs += int(directory)
        if (
            (fault == "file_fsync" and not directory)
            or (fault == "directory_fsync" and directory)
            or (fault == "cleanup_fsync" and directory and directory_syncs == 2)
        ):
            raise OSError(errno.EIO, "injected fsync failure")
        original_fsync(descriptor)

    def no_link(_source: Path, _destination: Path) -> None:
        raise OSError(errno.EXDEV, "injected unsupported hard links")

    def unlink(temporary: Path, *, missing_ok: bool = False) -> None:
        if fault == "cleanup" and temporary.name.startswith(".patch.json."):
            raise OSError(errno.EIO, "injected cleanup failure")
        original_unlink(temporary, missing_ok=missing_ok)

    monkeypatch.setattr(publication.os, "fsync", sync)
    monkeypatch.setattr(Path, "unlink", unlink)
    if fault == "link":
        monkeypatch.setattr(publication.os, "link", no_link)
    with pytest.raises(publication.ReevaluationPatchPublicationError) as caught:
        publication.publish_reevaluation_patch_if_absent(patch, path)
    linked = fault in {"directory_fsync", "cleanup", "cleanup_fsync"}
    assert caught.value.link_succeeded == linked
    assert caught.value.final_state == ("published" if linked else "absent")
    assert path.exists() == linked
    if linked:
        assert path.read_bytes() == legacy_bytes(patch)
    assert len(owned_temporaries(path)) == int(fault == "cleanup")


def test_failure_never_deletes_an_existing_competitor(
    tmp_path: Path, patch: MorpionReevaluationPatch, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even a pre-link I/O failure leaves another producer's final bytes intact."""
    path = tmp_path / "patch.json"
    competitor = legacy_bytes(replace(patch, patch_id="winner"))
    path.write_bytes(competitor)

    def fail(_descriptor: int) -> None:
        raise OSError(errno.EIO, "injected file fsync failure")

    monkeypatch.setattr(publication.os, "fsync", fail)
    with pytest.raises(publication.ReevaluationPatchPublicationError) as caught:
        publication.publish_reevaluation_patch_if_absent(patch, path)
    assert caught.value.errno == errno.EIO
    assert caught.value.final_state == "other"
    assert not caught.value.link_succeeded
    assert path.read_bytes() == competitor
    assert not owned_temporaries(path)


def test_link_can_take_effect_before_reporting_failure(
    tmp_path: Path, patch: MorpionReevaluationPatch, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inspect the inode instead of assuming an unsuccessful return created nothing."""
    path = tmp_path / "patch.json"
    original_link = os.link

    def uncertain_link(source: Path, destination: Path) -> None:
        original_link(source, destination)
        raise OSError(errno.EIO, "injected ambiguous link result")

    monkeypatch.setattr(publication.os, "link", uncertain_link)
    with pytest.raises(publication.ReevaluationPatchPublicationError) as caught:
        publication.publish_reevaluation_patch_if_absent(patch, path)
    assert not caught.value.link_succeeded
    assert caught.value.final_state == "published"
    assert path.read_bytes() == legacy_bytes(patch)
    assert not owned_temporaries(path)


CHILD = r"""
import json, os, sys
from contextlib import contextmanager
from pathlib import Path
import chipiron.environments.morpion.bootstrap.reevaluation_publication as p
from chipiron.environments.morpion.bootstrap.pipeline_artifacts import reevaluation_patch_from_dict
path = Path(sys.argv[1])
patch = reevaluation_patch_from_dict(json.loads(sys.argv[2]))
mode = sys.argv[3]
def pause(stage):
    print(stage, flush=True)
    assert sys.stdin.readline().strip() == "go"
original_link = os.link
def link(source, destination):
    if mode == "race": pause("ready")
    if mode == "reader": pause("before_link")
    if mode == "before_link": os._exit(87)
    original_link(source, destination)
    if mode == "after_link": os._exit(87)
    if mode == "reader": pause("after_link")
p.os.link = link
if mode == "reader":
    factory = p.tempfile.NamedTemporaryFile
    @contextmanager
    def temporary(**kwargs):
        with factory(**kwargs) as f:
            original_write = f.write
            def write(data):
                middle = len(data) // 2
                first = original_write(data[:middle])
                f.flush()
                pause("mid_write")
                return first + original_write(data[middle:])
            f.write = write
            yield f
    p.tempfile.NamedTemporaryFile = temporary
if mode == "cleanup":
    original_unlink = Path.unlink
    def unlink(self, *args, **kwargs):
        if self.name.startswith("." + path.name + "."): os._exit(87)
        return original_unlink(self, *args, **kwargs)
    Path.unlink = unlink
print(json.dumps(p.publish_reevaluation_patch_if_absent(patch, path)), flush=True)
"""


def start_child(
    path: Path, patch: MorpionReevaluationPatch, mode: str
) -> subprocess.Popen[str]:
    """Spawn a fresh producer process with pipe barriers, never a real worker."""
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            CHILD,
            str(path),
            json.dumps(reevaluation_patch_to_dict(patch)),
            mode,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def release(process: subprocess.Popen[str]) -> None:
    """Release a child paused at a known I/O boundary."""
    assert process.stdin is not None
    process.stdin.write("go\n")
    process.stdin.flush()


def expect_stage(process: subprocess.Popen[str], stage: str) -> None:
    """Bound a pipe wait for a hung child without depending on race timing."""
    assert process.stdout is not None
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        assert selector.select(timeout=30), f"Producer never reached {stage}"
    assert process.stdout.readline().strip() == stage


def test_two_simultaneous_producers_have_one_winner(
    tmp_path: Path, patch: MorpionReevaluationPatch
) -> None:
    """Both producers finish fsync before racing the same final directory entry."""
    path = tmp_path / "patch.json"
    candidates = [patch, replace(patch, patch_id="competitor")]
    children = [start_child(path, candidate, "race") for candidate in candidates]
    try:
        for child in children:
            expect_stage(child, "ready")
        for child in children:
            release(child)
        results = [child.communicate(timeout=30) for child in children]
        assert [child.returncode for child in children] == [0, 0], results
        won = [json.loads(output) for output, _ in results]
        assert sorted(won) == [False, True]
        assert path.read_bytes() == legacy_bytes(candidates[won.index(True)])
        assert not owned_temporaries(path)
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=30)


def test_reader_sees_absence_then_complete_json(
    tmp_path: Path, patch: MorpionReevaluationPatch
) -> None:
    """A separate reader runs during a partial write and on both sides of link."""
    path = tmp_path / "patch.json"
    child = start_child(path, patch, "reader")
    try:
        for stage in ["mid_write", "before_link", "after_link"]:
            expect_stage(child, stage)
            for _ in range(10):
                if stage == "after_link":
                    assert path.read_bytes() == legacy_bytes(patch)
                    assert load_reevaluation_patch(path) == patch
                else:
                    with pytest.raises(FileNotFoundError):
                        path.read_bytes()
            release(child)
        output, error = child.communicate(timeout=30)
        assert child.returncode == 0, error
        assert json.loads(output) is True
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=30)


@pytest.mark.parametrize("stage", ["before_link", "after_link", "cleanup"])
def test_subprocess_crash_and_orphan_recovery(
    tmp_path: Path, patch: MorpionReevaluationPatch, stage: str
) -> None:
    """A restart ignores owned crash orphans and preserves any published winner."""
    path = tmp_path / "patch.json"
    child = start_child(path, patch, stage)
    output, error = child.communicate(timeout=30)
    assert child.returncode == 87, (output, error)
    [orphan] = owned_temporaries(path)
    assert orphan.read_bytes() == legacy_bytes(patch)
    restart = replace(patch, patch_id="restart")
    assert publication.publish_reevaluation_patch_if_absent(restart, path) == (
        stage == "before_link"
    )
    assert orphan.read_bytes() == legacy_bytes(patch)
    assert path.read_bytes() == legacy_bytes(
        restart if stage == "before_link" else patch
    )


def test_partial_orphan_is_ignored_and_parent_creation_works(
    tmp_path: Path, patch: MorpionReevaluationPatch
) -> None:
    """A partial file from a dead attempt is never reused or swept by a new one."""
    path = tmp_path / "new" / "pipeline" / "patch.json"
    assert publication.publish_reevaluation_patch_if_absent(patch, path)
    path.unlink()
    orphan = path.with_name(f".{path.name}.old-attempt.tmp")
    orphan.write_bytes(b'{"partial":')
    assert publication.publish_reevaluation_patch_if_absent(patch, path)
    assert orphan.read_bytes() == b'{"partial":'
    assert load_reevaluation_patch(path) == patch
