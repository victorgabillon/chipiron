"""Filesystem rules shared by explicit scientific-v1 and scientific-v2 adapters.

The existing metadata key is retained as a fail-closed version fence.
Only run_state.json decides which complete capsules have committed.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import shutil
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from .cycle_timing import timestamp_utc_from_unix_s
from .derived.provenance import inside, sync_checkpoint
from .pipeline_artifacts import DISCOVERY_METADATA_KEY, MorpionPipelineActiveModel
from .sharded_training_v2_contract import (
    CLOSED_KIND,
    CLOSED_LAYOUT,
    CONTRACT,
    KIND,
    VERSION,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from .bootstrap_paths import MorpionBootstrapPaths
    from .pipeline_artifacts import MorpionPipelineGenerationManifest
    from .run_state import MorpionBootstrapRunState

PUBLICATION_KEY = "durable_generation_publication_v1"
PUBLICATION_NAMESPACE = "generation_publications_v1"
CAPSULE_SEAL = ".generation_publication_v1.json"
ATTEMPT_MARKER = ".publication_attempt_v1.json"
_PRIVATE_VALIDATION: ContextVar[Path | None] = ContextVar(
    "private_capsule_validation", default=None
)


def publication_namespace(version: int = 1) -> str:
    """Keep scientific-v2 transactions in an explicit, independent namespace."""
    require(
        isinstance(version, int)
        and not isinstance(version, bool)
        and version in (1, 2, 3),
        "Unsupported publication version.",
    )
    return f"generation_publications_v{version}"


def capsule_seal(version: int = 1) -> str:
    """Version the seal without changing any historical v1 payload."""
    publication_namespace(version)
    return f".generation_publication_v{version}.json"


@contextmanager
def private_generation_validation(staging: Path) -> Iterator[None]:
    """Permit only the owner's temporary validation index, never consumer discovery.

    The publisher holds the Growth lease. Its temporary index is removed before
    sealing and cannot become a cached consumer index for an uncommitted capsule.
    """
    require(staging.parent.name == "staging", "Private validation requires staging.")
    token = _PRIVATE_VALIDATION.set(staging.absolute())
    try:
        yield
    finally:
        _PRIVATE_VALIDATION.reset(token)


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
def growth_receipt_ownership(
    paths: MorpionBootstrapPaths, *, create: bool = True
) -> Iterator[None]:
    """Keep a stable kernel lock across the whole Growth invocation, never a TTL."""
    import fcntl

    if create:
        paths.pipeline_dir.mkdir(parents=True, exist_ok=True)
    path = paths.pipeline_dir / ".growth_patch_consumer.lock"
    if not create:
        inside(paths.work_dir, path.relative_to(paths.work_dir).as_posix())
    with path.open("a+" if create else "rb") as lock:
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
        isinstance(data, dict)
        and isinstance(data.get("schema_version"), int)
        and not isinstance(data.get("schema_version"), bool)
        and data["schema_version"] in (1, 2, 3),
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
    for index, record in enumerate(records):
        require(isinstance(record, dict), "Malformed generation root.")
        require(
            isinstance(record.get("publication_version", 1), int)
            and not isinstance(record.get("publication_version", 1), bool)
            and record.get("publication_version", 1) == data["schema_version"],
            "Mixed publication versions.",
        )
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
            index == 0
            or (
                generation == records[index - 1]["generation"] + 1
                and record.get("parent_id") == records[index - 1]["id"]
            ),
            "Broken publication ancestry.",
        )
    require(
        data.get("head_id") == records[-1]["id"],
        "Publication head differs from committed roots.",
    )
    require(
        records[0].get("parent_id") is None,
        "Publication ancestry lost its baseline root.",
    )
    return records


def capsule_path(work_dir: Path, identifier: str, version: int = 1) -> Path:
    """Resolve only a validated capsule identifier in the owned namespace."""
    require(
        len(identifier) == 32 and all(c in "0123456789abcdef" for c in identifier),
        "Malformed capsule identity.",
    )
    return work_dir / publication_namespace(version) / "objects" / identifier


def committed_evaluator(
    work_dir: Path, state: MorpionBootstrapRunState
) -> tuple[dict[str, Any] | None, Path | None]:
    """Resolve the evaluator from the verified scientific head, never a projection.

    The logical identity retains the original model path for exact patch receipts;
    only the sealed physical bundle is used for restoration. No model transition
    is inferred from a Training result or active_model.json.
    """
    head = publication_records(state.metadata)[-1]
    version = state.metadata[PUBLICATION_KEY]["schema_version"]
    require(version in (2, 3), "Committed evaluator requires scientific publication.")
    require(
        (head["generation"], head["checkpoint_path"], head["tree_snapshot_path"])
        == (
            state.generation,
            state.latest_runtime_checkpoint_path,
            state.latest_tree_snapshot_path,
        ),
        "Run state differs from its committed publication head.",
    )
    descriptor = validate_capsule(work_dir, head)
    require(
        descriptor["bootstrap_config_hash"]
        == state.metadata.get("bootstrap_config_hash"),
        "Committed evaluator configuration changed.",
    )
    require(
        ("evaluator_contract" in descriptor)
        == ("evaluator_contract" in state.metadata[PUBLICATION_KEY]),
        "Committed evaluator contract downgrade or mismatch.",
    )
    pins = descriptor["model_pins"]
    if "evaluator_contract" in state.metadata[PUBLICATION_KEY]:
        require(
            isinstance(state.metadata[PUBLICATION_KEY]["evaluator_contract"], int)
            and not isinstance(
                state.metadata[PUBLICATION_KEY]["evaluator_contract"], bool
            )
            and state.metadata[PUBLICATION_KEY]["evaluator_contract"] == 1
            and descriptor.get("evaluator_contract") == 1,
            "Unsupported committed evaluator contract.",
        )
        model = descriptor["active_evaluator"]
        pin = _active_evaluator_pin(descriptor)
        return model, inside(capsule_path(work_dir, head["id"], version), pin["path"])
    require(isinstance(pins, list) and len(pins) <= 1, "Ambiguous committed evaluator.")
    if not pins:
        return None, None
    pin = pins[0]
    model = {k: v for k, v in pin.items() if k != "origin"}
    model["path"] = pin["origin"]
    return model, inside(capsule_path(work_dir, head["id"], version), pin["path"])


def active_model_projection(
    paths: MorpionBootstrapPaths, state: MorpionBootstrapRunState
) -> MorpionPipelineActiveModel:
    """Derive advisory discovery from the sealed binding; do not infer activation."""
    model, _ = committed_evaluator(paths.work_dir, state)
    require(model is not None, "Missing explicitly committed evaluator.")
    assert model is not None
    assert state.last_save_unix_s is not None
    return MorpionPipelineActiveModel(
        generation=model["generation"],
        evaluator_name=model["name"],
        model_bundle_path=model["path"],
        source=model["source"],
        source_generation=model["source_generation"],
        local_trained_generation=model["local_trained_generation"],
        updated_at_utc=timestamp_utc_from_unix_s(state.last_save_unix_s),
        metadata={"committed_head": state.metadata[PUBLICATION_KEY]["head_id"]},
    )


def _active_evaluator_pin(descriptor: dict[str, Any]) -> dict[str, Any]:
    """One explicit scientific model binding, shared by sealing and restoration."""
    require(
        isinstance(descriptor.get("evaluator_contract"), int)
        and not isinstance(descriptor["evaluator_contract"], bool)
        and descriptor["evaluator_contract"] == 1,
        "Unsupported evaluator contract.",
    )
    model = descriptor.get("active_evaluator")
    require(isinstance(model, dict), "Missing active evaluator binding.")
    assert isinstance(model, dict)
    matches = [
        pin
        for pin in descriptor["model_pins"]
        if {**{k: v for k, v in pin.items() if k != "origin"}, "path": pin["origin"]}
        == model
    ]
    require(len(matches) == 1, "Active evaluator has no unique immutable binding.")
    require(
        descriptor["scientific_identity"]["producer"]["identity"]["evaluator"]
        == model["sha256"],
        "Scientific producer differs from active evaluator.",
    )
    return dict(matches[0])


def validate_capsule(work_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    """Verify every dependency before exposing scientific rows or acknowledging a patch."""
    version = record.get("publication_version", 1)
    root = capsule_path(work_dir, record["id"], version)
    seal = inside(work_dir, record["descriptor_path"])
    require(
        seal == root / capsule_seal(version),
        "Descriptor escaped its committed capsule.",
    )
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
    require(
        isinstance(descriptor.get("publication_version", 1), int)
        and not isinstance(descriptor.get("publication_version", 1), bool)
        and descriptor.get("publication_version", 1) == version,
        "Capsule version mismatch.",
    )
    if version in (2, 3):
        require(
            descriptor.get("scientific_identity") == record.get("scientific_identity")
            and isinstance(record.get("scientific_identity"), dict),
            "Scientific publication identity mismatch.",
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
            and relative != capsule_seal(version),
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
    if version in (2, 3):
        validate_committed_identity(work_dir, root, record, descriptor)
    if "evaluator_contract" in descriptor:
        require(version == 3, "Evaluator binding requires closed publication.")
        _active_evaluator_pin(descriptor)
    return descriptor


@contextmanager
def capsule_lease(
    work_dir: Path,
    identifier: str,
    *,
    exclusive: bool = False,
    nonblocking: bool = False,
    version: int = 1,
    create: bool = True,
) -> Iterator[None]:
    """Stable inode pins are never unlinked, including by orphan collection."""
    import fcntl

    capsule_path(
        work_dir, identifier, version
    )  # Validate before using the ID as a filename.
    directory = work_dir / publication_namespace(version) / "pins"
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{identifier}.lock"
    if not create:
        inside(work_dir, path.relative_to(work_dir).as_posix())
    with path.open("a+" if create else "rb") as handle:
        flags = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        if nonblocking:
            flags |= fcntl.LOCK_NB
        fcntl.flock(handle, flags)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def locate_capsule(path: Path) -> tuple[Path, str, int] | None:
    """Recognize only the explicit publication namespace, leaving legacy v1 alone."""
    for ancestor in path.absolute().parents:
        if ancestor.parent.name == "objects" and ancestor.parent.parent.name in {
            PUBLICATION_NAMESPACE,
            publication_namespace(2),
            publication_namespace(3),
        }:
            return (
                ancestor.parent.parent.parent,
                ancestor.name,
                int(ancestor.parent.parent.name[-1]),
            )
    return None


@contextmanager
def generation_read_pin(
    path: Path, *, manifest: MorpionPipelineGenerationManifest | None = None
) -> Iterator[None]:
    """Read a raw capsule path only after adopting its run-state commit."""
    for ancestor in path.absolute().parents:
        if ancestor.parent.parent.name in {
            publication_namespace(2),
            publication_namespace(3),
        } and ancestor.parent.name in {"staging", "trash"}:
            require(
                ancestor == _PRIVATE_VALIDATION.get(),
                "Uncommitted private scientific generation.",
            )
    location = locate_capsule(path)
    if location is None:
        yield
        return
    work_dir, identifier, version = location
    with capsule_lease(work_dir, identifier, version=version):
        state = read_object(work_dir / "run_state.json")
        records = publication_records(state["metadata"])
        require(
            state["metadata"][PUBLICATION_KEY]["schema_version"] == version,
            "Publication namespace/version mismatch.",
        )
        matching = [r for r in records if r["id"] == identifier]
        require(
            len(matching) == 1, "Uncommitted capsule is not a published generation."
        )
        head = records[-1]
        require(
            (
                state["generation"],
                state["latest_runtime_checkpoint_path"],
                state["latest_tree_snapshot_path"],
            )
            == (
                head["generation"],
                head["checkpoint_path"],
                head["tree_snapshot_path"],
            ),
            "Run state differs from its committed publication head.",
        )
        if manifest is not None:
            _validate_manifest_record(manifest, matching[0], records[0]["generation"])
        validate_capsule(work_dir, matching[0])
        # A writer may have died between complete state replacement and dir fsync.
        sync_artifact(work_dir / "run_state.json", work_dir)
        yield


def assert_artifact_writable(path: Path) -> None:
    """Cooperative serializers cannot overwrite a sealed capsule or its hard links."""
    require(
        not any(
            parent.name == "objects"
            and parent.parent.name
            in {"dataset_publications_v1", "training_publications_v1"}
            for parent in path.parents
        ),
        "Cannot write into an immutable Dataset or Training publication.",
    )
    for ancestor in [path, *path.parents]:
        require(
            not any((ancestor / capsule_seal(v)).exists() for v in (1, 2, 3)),
            "Cannot write into an immutable publication capsule.",
        )


def pinned_file_hashes(path: Path) -> dict[str, str]:
    """Expose already-validated capsule hashes to a reader holding its lifetime pin."""
    location = locate_capsule(path)
    if location is None:
        return {}
    work_dir, identifier, version = location
    root = capsule_path(work_dir, identifier, version)
    descriptor = read_object(root / capsule_seal(version))
    return {str(root / entry["path"]): entry["sha256"] for entry in descriptor["files"]}


def _validate_manifest_record(
    manifest: MorpionPipelineGenerationManifest, record: dict[str, Any], baseline: int
) -> None:
    """Compare advisory identity with the root observed under the selected pin."""
    generation = manifest.generation
    require(generation == record["generation"], "Discovery generation mismatch.")
    require(
        (
            manifest.runtime_checkpoint_path,
            manifest.tree_snapshot_path,
        )
        == (record["checkpoint_path"], record["tree_snapshot_path"]),
        "Discovery manifest differs from its committed generation.",
    )
    projection = manifest.metadata.get(DISCOVERY_METADATA_KEY)
    if projection is not None:
        assert isinstance(projection, dict)  # Validated by the manifest type.
        require(
            projection["publication"]
            == {
                k: v
                for k, v in record.items()
                if k not in {"accepted_dataset", "accepted_training"}
            }
            or (
                generation == baseline
                and projection["publication"] is None
                and projection["checkpoint"]["path"] == record["checkpoint_path"]
            ),
            "Discovery projection differs from its committed publication.",
        )


@contextmanager
def manifest_read_pin(
    work_dir: Path, manifest: MorpionPipelineGenerationManifest
) -> Iterator[None]:
    """Select a pin from advisory data, then re-observe authority inside that pin.

    Old external baselines have no capsule path; only those need an initial root
    lookup. It is a hint, never the authoritative observation used for validation.
    """
    projection = manifest.metadata.get(DISCOVERY_METADATA_KEY)
    assert projection is None or isinstance(projection, dict)
    record = None if projection is None else projection["publication"]
    path = None
    if record is not None:
        path = inside(work_dir, record["descriptor_path"])
        require(locate_capsule(path) is not None, "Invalid discovery capsule path.")
    else:
        for value in (manifest.tree_snapshot_path, manifest.runtime_checkpoint_path):
            if value is not None and locate_capsule(work_dir / value) is not None:
                path = work_dir / value
                break
    if path is None:
        state_path = work_dir / "run_state.json"
        if state_path.is_file():
            metadata = read_object(state_path).get("metadata", {})
            if PUBLICATION_KEY in metadata:
                records = publication_records(metadata)
                if manifest.generation >= records[0]["generation"]:
                    matching = [
                        r for r in records if r["generation"] == manifest.generation
                    ]
                    require(
                        len(matching) == 1,
                        "Discovery manifest has no committed generation.",
                    )
                    path = inside(work_dir, matching[0]["descriptor_path"])
        if path is None:
            yield  # Unmanaged history keeps its original interpretation.
            return
    with generation_read_pin(path, manifest=manifest):
        yield


def validate_manifest_publication(
    work_dir: Path, manifest: MorpionPipelineGenerationManifest
) -> None:
    """Verify one selected closure; consumers retain manifest_read_pin while using it."""
    with manifest_read_pin(work_dir, manifest):
        pass


def audit_generation_publications(work_dir: Path) -> None:
    """Explicitly verify every committed capsule, even without discovery manifests.

    Audit the roots in this run-state observation. Each root is re-adopted under
    its existing pin; generations committed later belong to a subsequent audit.
    """
    state = read_object(work_dir / "run_state.json")
    for record in publication_records(state["metadata"]):
        with generation_read_pin(inside(work_dir, record["descriptor_path"])):
            pass


def object_digest(value: object) -> str:
    """Bind an explicit JSON identity without discarding any provenance fields."""
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def copy_artifact(source: Path, destination: Path, *, link: bool = False) -> None:
    """Copy an external baseline, or link immutable committed capsule bytes."""
    require(not source.is_symlink(), f"Symlinked publication input: {source}")
    if source.is_dir():
        destination.mkdir(parents=True, exist_ok=False)
        for path in sorted(source.iterdir()):
            copy_artifact(path, destination / path.name, link=link)
    else:
        require(source.is_file(), f"Missing publication input: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if link:
            os.link(source, destination)
        else:
            shutil.copyfile(source, destination)


def validate_committed_identity(
    work_dir: Path, root: Path, record: dict[str, Any], descriptor: dict[str, Any]
) -> None:
    """Bind managed reader sources to the sealed checkpoint/scientific identity."""
    identity = record["scientific_identity"]
    path = inside(work_dir, record["tree_snapshot_path"])
    require(
        path.is_relative_to(root)
        and descriptor["sharded_snapshot_path"] == path.relative_to(root).as_posix(),
        "Scientific pointer escaped its sealed identity.",
    )
    require(
        (
            identity.get("source_format_kind"),
            identity.get("source_format_version"),
            identity.get("scientific_contract"),
        )
        == (
            CLOSED_KIND if record.get("publication_version") == 3 else KIND,
            VERSION,
            CONTRACT,
        ),
        "Unsupported committed scientific identity.",
    )
    if record.get("publication_version") == 3:
        # The seal binds a descriptor already checked by the scientific reader.
        # Compare its declared hashes to this freshly verified inventory.
        header = read_object(path)
        require(
            header["generation"] == record["generation"]
            and header["producer"] == identity["producer"],
            "Closed descriptor differs from committed identity.",
        )
        require(
            descriptor["bootstrap_config_hash"]
            == identity["producer"]["identity"]["config_sha256"]
            == hashlib.sha256(
                (root / "bootstrap_config.json").read_bytes().removesuffix(b"\n")
            ).hexdigest(),
            "Closed capsule config identity changed.",
        )
        for model in descriptor["model_pins"]:
            require(
                artifact_digest(inside(root, model["path"])) == model["sha256"],
                "Closed capsule model identity changed.",
            )
        inventory = {entry["path"]: entry for entry in descriptor["files"]}
        for key in ("record_shard", "update_shard"):
            shard = header[key]
            entry = inventory.get(
                (path.parent / shard["path"]).relative_to(root).as_posix()
            )
            require(
                entry is not None and entry["sha256"] == shard["sha256"],
                "Closed shard differs from its validated descriptor.",
            )
    require(
        identity.get("growth_generation")
        == identity.get("source_generation")
        == record["generation"]
        and identity.get("generation_mapping") == "growth-equals-scientific-from-1",
        "Scientific generation mapping mismatch.",
    )
    require(
        identity["producer_sha256"] == object_digest(identity["producer"])
        and identity["generation_descriptor_sha256"] == file_digest(path)
        and (
            identity.get("storage_layout_version") == CLOSED_LAYOUT
            and "catalog_sha256" not in identity
            if record.get("publication_version") == 3
            else identity["catalog_sha256"]
            == file_digest(path.parent.parent / "manifest.json")
        )
        and identity["checkpoint_sha256"]
        == artifact_digest(inside(work_dir, record["checkpoint_path"])),
        "Committed scientific/checkpoint identity changed.",
    )


def committed_scientific_source(path: Path) -> dict[str, Any] | None:
    """Bind a managed v2 patch to the exact committed source, not a generation label alone."""
    location = locate_capsule(path)
    if location is None or location[2] not in (2, 3):
        return None
    work_dir, identifier, _version = location
    with generation_read_pin(path):
        state = read_object(work_dir / "run_state.json")
        record = next(
            r for r in publication_records(state["metadata"]) if r["id"] == identifier
        )
        require(
            path.absolute()
            == inside(work_dir, record["tree_snapshot_path"]).absolute(),
            "Patch source must select its capsule's committed scientific head.",
        )
        return {
            "capsule_id": identifier,
            "scientific_identity": record["scientific_identity"],
        }
