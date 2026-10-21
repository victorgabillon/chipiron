"""Filesystem rules for opt-in immutable publication of scientific v1 artifacts.

The publication protocol is v1 too; it is unrelated to scientific sharded v2.
Only run_state.json decides which complete capsules have committed.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from .derived.provenance import inside, sync_checkpoint

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from .bootstrap_paths import MorpionBootstrapPaths
    from .pipeline_artifacts import MorpionPipelineGenerationManifest

PUBLICATION_KEY = "durable_generation_publication_v1"
PUBLICATION_NAMESPACE = "generation_publications_v1"
CAPSULE_SEAL = ".generation_publication_v1.json"
ATTEMPT_MARKER = ".publication_attempt_v1.json"


class GenerationPublicationError(ValueError):
    """Missing transaction evidence or a conflicting immutable artifact."""


def require(condition: bool, message: str) -> None:
    """Reject ambiguity before applying, acknowledging or deleting scientific data."""
    if not condition:
        raise GenerationPublicationError(message)


def read_object(path: Path) -> dict[str, Any]:
    """Read a required object without supplying transaction defaults."""
    value = json.loads(path.read_bytes())
    require(isinstance(value, dict), f"Expected transaction object: {path}")
    assert isinstance(value, dict)
    return value


def file_digest(path: Path) -> str:
    """Hash bytes with bounded RAM."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@contextmanager
def growth_receipt_ownership(paths: MorpionBootstrapPaths) -> Iterator[None]:
    """Keep a stable kernel lock across the whole Growth invocation, never a TTL."""
    import fcntl

    paths.pipeline_dir.mkdir(parents=True, exist_ok=True)
    with (paths.pipeline_dir / ".growth_patch_consumer.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def artifact_digest(path: Path) -> str:
    """Use the same exact-byte / sorted-file identity as PR #79."""
    require(not path.is_symlink(), f"Symlinked transaction artifact: {path}")
    if path.is_file():
        return file_digest(path)
    require(path.is_dir(), f"Missing transaction artifact: {path}")
    entries = sorted(path.rglob("*"))
    require(
        not any(p.is_symlink() for p in entries), f"Symlinked artifact tree: {path}"
    )
    files = [p for p in entries if p.is_file()]
    require(bool(files), f"Empty transaction artifact: {path}")
    inventory = [
        [p.relative_to(path).as_posix(), p.stat().st_size, file_digest(p)]
        for p in files
    ]
    return hashlib.sha256(
        json.dumps(inventory, separators=(",", ":")).encode()
    ).hexdigest()


def sync_directory(path: Path) -> None:
    """Require local POSIX directory durability; never silently skip a failure."""
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def sync_artifact(path: Path, work_dir: Path) -> None:
    """Persist all payload files and namespace entries through the existing run root."""
    require(
        path.is_relative_to(work_dir) and path != work_dir,
        "External publication artifact.",
    )
    sync_checkpoint(path)
    parent = path.parent
    while parent != work_dir:
        sync_directory(parent)
        parent = parent.parent
    sync_directory(work_dir)


def publish_directory_if_absent(staging: Path, final: Path, work_dir: Path) -> None:
    """Linux renameat2(RENAME_NOREPLACE), with no replace/check-then-rename fallback.

    An error after the rename can leave a complete final capsule. Its presence
    alone is never a committed generation; recovery consults actual run state.
    """
    library = ctypes.CDLL(None, use_errno=True)
    rename = getattr(library, "renameat2", None)
    require(
        rename is not None,
        "Durable publication requires Linux renameat2(RENAME_NOREPLACE).",
    )
    assert rename is not None
    rename.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    rename.restype = ctypes.c_int
    result = rename(-100, os.fsencode(staging), -100, os.fsencode(final), 1)
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(final))
    sync_directory(final.parent)
    sync_directory(staging.parent)
    sync_directory(final.parent.parent)
    sync_directory(work_dir)


def publication_records(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Read the append-only roots owned by the same atomic run-state commit."""
    data = metadata.get(PUBLICATION_KEY)
    require(
        isinstance(data, dict) and data.get("schema_version") == 1,
        "Missing or unsupported durable generation protocol.",
    )
    assert isinstance(data, dict)
    records = data.get("generations")
    require(
        isinstance(records, list) and bool(records),
        "Missing committed generation roots.",
    )
    assert isinstance(records, list)
    ids: set[str] = set()
    generations: set[int] = set()
    for record in records:
        require(isinstance(record, dict), "Malformed generation root.")
        identifier = record.get("id")
        generation = record.get("generation")
        require(
            isinstance(identifier, str)
            and len(identifier) == 32
            and all(c in "0123456789abcdef" for c in identifier),
            "Malformed capsule identity.",
        )
        require(
            isinstance(generation, int)
            and not isinstance(generation, bool)
            and generation > 0,
            "Malformed publication generation.",
        )
        require(
            identifier not in ids and generation not in generations,
            "Duplicate publication root.",
        )
        ids.add(identifier)
        generations.add(generation)
    require(
        data.get("head_id") == records[-1]["id"],
        "Publication head differs from committed roots.",
    )
    require(
        records[0].get("parent_id") is None,
        "Publication ancestry lost its baseline root.",
    )
    return records


def capsule_path(work_dir: Path, identifier: str) -> Path:
    """Resolve only a validated capsule identifier in the owned namespace."""
    require(
        len(identifier) == 32 and all(c in "0123456789abcdef" for c in identifier),
        "Malformed capsule identity.",
    )
    return work_dir / PUBLICATION_NAMESPACE / "objects" / identifier


def validate_capsule(work_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    """Verify every dependency before exposing scientific rows or acknowledging a patch."""
    root = capsule_path(work_dir, record["id"])
    seal = inside(work_dir, record["descriptor_path"])
    require(seal == root / CAPSULE_SEAL, "Descriptor escaped its committed capsule.")
    require(
        file_digest(seal) == record["descriptor_sha256"],
        "Committed capsule descriptor changed.",
    )
    descriptor = read_object(seal)
    require(
        (descriptor.get("protocol"), descriptor.get("id"), descriptor.get("generation"))
        == (PUBLICATION_KEY, record["id"], record["generation"]),
        "Capsule generation identity mismatch.",
    )
    entries = descriptor.get("files")
    require(
        isinstance(entries, list) and bool(entries),
        "Missing capsule dependency inventory.",
    )
    assert isinstance(entries, list)
    names: set[str] = set()
    for entry in entries:
        require(isinstance(entry, dict), "Malformed capsule dependency.")
        relative = entry["path"]
        require(
            isinstance(relative, str)
            and relative not in names
            and relative != CAPSULE_SEAL,
            "Conflicting capsule dependency path.",
        )
        names.add(relative)
        path = inside(root, relative)
        require(
            path.is_file()
            and path.stat().st_size == entry["size"]
            and file_digest(path) == entry["sha256"],
            f"Committed capsule dependency changed: {path}",
        )
    actual = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p != seal
    }
    require(actual == names, "Capsule inventory does not match its files.")
    require(
        not any(p.is_symlink() for p in root.rglob("*")),
        "Symlinked capsule dependency.",
    )
    if not descriptor.get("baseline"):
        for field in ["checkpoint_path", "tree_snapshot_path"]:
            value = record[field]
            if value is not None:
                path = inside(work_dir, value)
                require(
                    path.is_relative_to(root), "Committed pointer escaped its capsule."
                )
    return descriptor


@contextmanager
def capsule_lease(
    work_dir: Path,
    identifier: str,
    *,
    exclusive: bool = False,
    nonblocking: bool = False,
) -> Iterator[None]:
    """Stable inode pins are never unlinked, including by orphan collection."""
    import fcntl

    capsule_path(work_dir, identifier)  # Validate before using the ID as a filename.
    directory = work_dir / PUBLICATION_NAMESPACE / "pins"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{identifier}.lock").open("a+") as handle:
        flags = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        if nonblocking:
            flags |= fcntl.LOCK_NB
        fcntl.flock(handle, flags)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def locate_capsule(path: Path) -> tuple[Path, str] | None:
    """Recognize only the explicit publication namespace, leaving legacy v1 alone."""
    for ancestor in path.absolute().parents:
        if (
            ancestor.parent.name == "objects"
            and ancestor.parent.parent.name == PUBLICATION_NAMESPACE
        ):
            return ancestor.parent.parent.parent, ancestor.name
    return None


@contextmanager
def generation_read_pin(path: Path) -> Iterator[None]:
    """Read a raw capsule path only after adopting its run-state commit."""
    location = locate_capsule(path)
    if location is None:
        yield
        return
    work_dir, identifier = location
    with capsule_lease(work_dir, identifier):
        state = read_object(work_dir / "run_state.json")
        records = publication_records(state["metadata"])
        matching = [r for r in records if r["id"] == identifier]
        require(
            len(matching) == 1, "Uncommitted capsule is not a published generation."
        )
        validate_capsule(work_dir, matching[0])
        # A writer may have died between complete state replacement and dir fsync.
        sync_artifact(work_dir / "run_state.json", work_dir)
        yield


def assert_artifact_writable(path: Path) -> None:
    """Cooperative serializers cannot overwrite a sealed capsule or its hard links."""
    for ancestor in [path, *path.parents]:
        require(
            not (ancestor / CAPSULE_SEAL).exists(),
            "Cannot write into an immutable publication capsule.",
        )


def pinned_file_hashes(path: Path) -> dict[str, str]:
    """Expose already-validated capsule hashes to a reader holding its lifetime pin."""
    location = locate_capsule(path)
    if location is None:
        return {}
    work_dir, identifier = location
    root = capsule_path(work_dir, identifier)
    descriptor = read_object(root / CAPSULE_SEAL)
    return {str(root / entry["path"]): entry["sha256"] for entry in descriptor["files"]}


def validate_manifest_publication(
    work_dir: Path, manifest: MorpionPipelineGenerationManifest
) -> None:
    """Check discovery against the authoritative state, preserving legacy baselines."""
    if not (work_dir / PUBLICATION_NAMESPACE).exists():
        return
    state_path = work_dir / "run_state.json"
    if not state_path.is_file():
        return
    state = read_object(state_path)
    metadata = state.get("metadata", {})
    if PUBLICATION_KEY not in metadata:
        return
    records = publication_records(metadata)
    generation = manifest.generation
    baseline = records[0]["generation"]
    if generation < baseline:
        return  # Adopted legacy history is retained outside the managed namespace.
    matching = [r for r in records if r["generation"] == generation]
    require(len(matching) == 1, "Discovery manifest has no committed generation.")
    record = matching[0]
    require(
        (
            manifest.runtime_checkpoint_path,
            manifest.tree_snapshot_path,
        )
        == (record["checkpoint_path"], record["tree_snapshot_path"]),
        "Discovery manifest differs from its committed generation.",
    )
    validate_capsule(work_dir, record)
