"""Complete create-only patch publication on local POSIX filesystems.

The hard link is the visibility point. A later failure never rolls it back:
callers must inspect the final path, and must not advance a cursor on an error.
This module does not make patch publication and cursor updates transactional.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from .pipeline_artifacts import (
    reevaluation_patch_from_dict,
    reevaluation_patch_to_dict,
)

if TYPE_CHECKING:
    from .pipeline_artifacts import MorpionReevaluationPatch

type PatchPublicationState = Literal["published", "absent", "other", "unknown"]


class ReevaluationPatchPublicationError(OSError):
    """Publication failed; the observed final path may already be complete.

    ``final_state`` is a point-in-time inode comparison, not an ownership lease.
    A consumer can remove the patch before or after that observation.
    ``link_succeeded`` records a successful syscall return; False alone does not
    prove that a failed syscall had no effect. The original errno is preserved.
    """

    def __init__(
        self,
        path: Path,
        error: OSError,
        *,
        link_succeeded: bool,
        final_state: PatchPublicationState,
    ) -> None:
        """Retain publication history separately from the observed namespace."""
        super().__init__(
            error.errno,
            f"Reevaluation patch publication failed: {error}; "
            f"link_succeeded={link_succeeded} final_state={final_state}; "
            "inspect the pending patch before retrying; cursor must not advance",
            str(path),
        )
        self.path = path
        self.link_succeeded = link_succeeded
        self.final_state = final_state


def _sync_directory(directory: Path) -> None:
    """Require directory durability; unsupported filesystems fail explicitly."""
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_parent_directory(directory: Path) -> None:
    """Persist newly created parent entries up to an existing directory."""
    missing: list[Path] = []
    ancestor = directory
    while not ancestor.exists():
        missing.append(ancestor)
        ancestor = ancestor.parent
    directory.mkdir(parents=True, exist_ok=True)
    for created in reversed(missing):
        _sync_directory(created.parent)


def _publication_state(
    path: Path, temporary_identity: os.stat_result | None
) -> PatchPublicationState:
    """Inspect actual state even when a failed link may have taken effect."""
    try:
        final_identity = path.lstat()
    except FileNotFoundError:
        return "absent"
    except OSError:
        return "unknown"
    if temporary_identity is None:
        return "other"
    if (final_identity.st_dev, final_identity.st_ino) == (
        temporary_identity.st_dev,
        temporary_identity.st_ino,
    ):
        return "published"
    return "other"


def publish_reevaluation_patch_if_absent(
    patch: MorpionReevaluationPatch, path: Path
) -> bool:
    """Publish complete, validated, fsynced patch bytes without replacement.

    Return False only for an existing final entry (including a dangling symlink).
    Return True after link publication, directory fsync, owned temporary cleanup,
    and a second directory fsync. I/O errors preserve any final entry and report
    its observed identity through ReevaluationPatchPublicationError.

    Requires hard links and directory fsync on the same local filesystem, stable
    parent directories, and an already durable existing ancestor. There is no
    replacement or direct-write fallback. Crash orphans are ignored, never reused.
    """
    contents = (
        json.dumps(reevaluation_patch_to_dict(patch), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary: Path | None = None
    temporary_identity: os.stat_result | None = None
    link_succeeded = False
    try:
        _ensure_parent_directory(path.parent)
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                temporary_identity = os.fstat(handle.fileno())
                handle.write(contents)
                handle.flush()
                handle.seek(0)
                written = handle.read()
                if written != contents:
                    message = f"Incomplete or changed temporary patch: {temporary}"
                    raise ValueError(message)
                reevaluation_patch_from_dict(json.loads(written))
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                return False
            link_succeeded = True
            _sync_directory(path.parent)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
                if link_succeeded:
                    _sync_directory(path.parent)
    except OSError as exc:
        raise ReevaluationPatchPublicationError(
            path,
            exc,
            link_succeeded=link_succeeded,
            final_state=_publication_state(path, temporary_identity),
        ) from exc
    return True
