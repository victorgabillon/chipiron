"""Committed evidence for reusing immutable canonical checks, never cache authority.

Only the publisher's private validation scope can consume this evidence. Every
shard is still parsed and freshly hashed by the existing bounded reader. Mutable
updates, creation keys and topology are always validated in a fresh SQLite index.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import ijson

from .generation_publication_io import (
    capsule_path,
    capsule_seal,
    generation_read_pin,
    locate_capsule,
    object_digest,
    read_object,
    require,
    validate_capsule,
)
from .sharded_training_v2_contract import CONTRACT, KIND, REFERENCE_CONTRACT, VERSION

if TYPE_CHECKING:
    from collections.abc import Iterator

EVIDENCE_KIND = "morpion-immutable-canonical-validation"
EVIDENCE_VERSION = 1


@dataclass(frozen=True)
class _ValidationScope:
    manifest: Path
    producer: dict[str, Any]
    records: dict[str, dict[str, Any]]


_SCOPE: ContextVar[_ValidationScope | None] = ContextVar(
    "scientific_canonical_validation", default=None
)


def validator_identity() -> dict[str, Any]:
    """Hash the pinned codec/validator implementation, independent of install path.

    Conservatively include complete relevant packages, not only their version
    labels. Runtime/dependency code must remain immutable during one process,
    as for all other scientific code; this is not a sandbox for hostile Python.
    No unbounded or persistent cache of code or scientific records is introduced.
    """
    digest = hashlib.sha256()
    for name in ("atomheart.games.morpion", "valanga", "anemone.checkpoints", "ijson"):
        module = importlib.import_module(name)
        require(module.__file__ is not None, "Missing canonical validator source.")
        assert module.__file__ is not None
        root = Path(module.__file__).parent
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in {".py", ".so"}:
                digest.update(f"{name}/{path.relative_to(root)}\0".encode())
                with path.open("rb") as handle:
                    digest.update(hashlib.file_digest(handle, "sha256").digest())
    for name in (
        "scientific_validation_evidence.py",
        "scientific_generation_publication.py",
        "generation_publication.py",
        "generation_publication_io.py",
        "sharded_training_reader.py",
        "sharded_training_v2.py",
        "sharded_training_v2_contract.py",
        "closed_scientific_snapshot.py",
    ):
        digest.update(name.encode() + b"\0")
        with Path(__file__).with_name(name).open("rb") as handle:
            digest.update(hashlib.file_digest(handle, "sha256").digest())
    return {
        "format_kind": KIND,
        "format_version": VERSION,
        "scientific_contract": CONTRACT,
        "state_ref_contract": REFERENCE_CONTRACT,
        "python": [
            sys.implementation.name,
            sys.version,
            sys.implementation.cache_tag,
            sys.flags.optimize,
            sys.byteorder,
            sys.maxsize,
        ],
        "ijson_backend": ijson.backend,
        "implementation_sha256": digest.hexdigest(),
    }


def _validate_evidence(evidence: object) -> None:
    """Use one schema validator for both existing committed-evidence consumers."""
    require(
        isinstance(evidence, dict)
        and set(evidence)
        == {
            "kind",
            "version",
            "validator",
            "scientific_identity",
            "record_shards",
            "reused_from",
        }
        and evidence["kind"] == EVIDENCE_KIND
        and isinstance(evidence["version"], int)
        and not isinstance(evidence["version"], bool)
        and evidence["version"] == EVIDENCE_VERSION,
        "Unsupported or malformed canonical validation evidence.",
    )


@contextmanager
def committed_canonical_validation(
    *,
    work_dir: Path,
    parent: dict[str, Any] | None,
    manifest: Path,
    header: dict[str, Any],
    validator: dict[str, Any],
) -> Iterator[dict[str, Any] | None]:
    """Authorize only the exact committed parent's immutable descriptor prefix.

    A missing old receipt or changed implementation forces complete validation.
    Malformed evidence, substitutions and inconsistent ancestry fail closed.
    The ordinary committed-generation pin verifies the authoritative run-state
    membership and capsule bytes; a disposable SQLite file grants no authority.
    """
    if parent is None:
        yield None
        return
    require(
        parent.get("publication_version") == 2
        and parent["generation"] + 1 == header["generation"],
        "Canonical validation requires consecutive scientific ancestry.",
    )
    source = work_dir / parent["tree_snapshot_path"]
    with generation_read_pin(source):
        descriptor = validate_capsule(work_dir, parent)
        evidence = descriptor.get("canonical_validation")
        if evidence is None:
            yield None
            return
        _validate_evidence(evidence)
        require(
            evidence["scientific_identity"] == descriptor["scientific_identity"]
            and evidence["scientific_identity"]["producer"] == header["producer"]
            and evidence["record_shards"] == header["record_shards"][:-1]
            and len(evidence["record_shards"]) == parent["generation"],
            "Canonical validation evidence differs from committed source ancestry.",
        )
        if evidence["validator"] != validator:
            yield None
            return
        scope = _ValidationScope(
            manifest=manifest.resolve(),
            producer=header["producer"],
            records={row["path"]: row for row in evidence["record_shards"]},
        )
        token = _SCOPE.set(scope)
        try:
            yield {
                "capsule_id": parent["id"],
                "descriptor_sha256": parent["descriptor_sha256"],
                "generation": parent["generation"],
                "evidence_sha256": object_digest(evidence),
            }
        finally:
            _SCOPE.reset(token)


def canonical_record_is_validated(
    manifest: Path, producer: dict[str, Any], descriptor: dict[str, Any]
) -> bool:
    """Permit one descriptor only inside the publisher's verified private scope.

    The reader has already freshly hashed the source against this descriptor,
    and rehashes it after indexing. All immutable row structure is still checked.
    """
    scope = _SCOPE.get()
    if scope is None:
        return False
    require(
        manifest == scope.manifest and producer == scope.producer,
        "Canonical validation scope does not match this scientific source.",
    )
    return scope.records.get(descriptor["path"]) == descriptor


def canonical_validation_evidence(
    *,
    identity: dict[str, Any],
    records: list[dict[str, Any]],
    validator: dict[str, Any],
    reused_from: dict[str, Any] | None,
) -> dict[str, Any]:
    """Seal an assertion only after complete bounded validation has succeeded."""
    return {
        "kind": EVIDENCE_KIND,
        "version": EVIDENCE_VERSION,
        "validator": validator,
        "scientific_identity": identity,
        "record_shards": records,
        "reused_from": reused_from,
    }


@contextmanager
def _validated_record_scope(
    manifest: Path, producer: dict[str, Any], records: list[dict[str, Any]]
) -> Iterator[None]:
    """Private transfer of completed checks over exact bytes, never persisted.

    Callers must validate canonicality and exact content hashes before entering;
    the indexed reader still hashes the specified files before/after indexing.
    """
    token = _SCOPE.set(
        _ValidationScope(
            manifest.resolve(), producer, {row["path"]: row for row in records}
        )
    )
    try:
        yield
    finally:
        _SCOPE.reset(token)


@contextmanager
def committed_source_validation(path: Path) -> Iterator[None]:
    """Reuse only a pinned current validator's committed canonical assertion.

    The caller already holds generation_read_pin and has verified that path is
    the committed scientific head dependency. Missing/stale evidence forces
    the existing full canonical validation. File hashes are never skipped.
    """
    location = locate_capsule(path)
    require(
        location is not None and location[2] in (2, 3),
        "Canonical source must be committed scientific v2.",
    )
    assert location is not None
    work_dir, identifier, version = location
    descriptor = read_object(
        capsule_path(work_dir, identifier, version) / capsule_seal(version)
    )
    evidence = descriptor.get("canonical_validation")
    if evidence is None:
        yield
        return
    _validate_evidence(evidence)
    require(
        evidence["scientific_identity"] == descriptor["scientific_identity"],
        "Canonical source identity mismatch.",
    )
    if evidence["validator"] != validator_identity():
        yield
        return
    with _validated_record_scope(
        path, evidence["scientific_identity"]["producer"], evidence["record_shards"]
    ):
        yield
