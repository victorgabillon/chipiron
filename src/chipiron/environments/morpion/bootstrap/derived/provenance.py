"""Durable identities and strict path boundaries for opt-in derived experiments."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

PROVENANCE_NAME = "derived_experiment_provenance.json"
SCHEMA = "morpion_derived_checkpoint_v1"


class DerivationError(ValueError):
    """An unsafe or scientifically ambiguous derivation must stop explicitly."""


def read_json(path: Path) -> dict[str, Any]:
    """Read a required JSON mapping; never supply experiment defaults."""
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        message = f"Expected a JSON object: {path}"
        raise DerivationError(message)
    return result


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    """Commit a small journal or pointer only after flushing its complete payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, delete=False, encoding="utf-8"
    ) as handle:
        temporary = Path(handle.name)
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def sha256(path: Path) -> str:
    """Hash files without materializing large artifacts in RAM."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def inside(root: Path, relative: str) -> Path:
    """Reject escaping, missing or symlinked experiment inputs."""
    candidate = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        message = f"Unsafe artifact path: {relative}"
        raise DerivationError(message)
    if not candidate.exists() or not candidate.resolve().is_relative_to(root.resolve()):
        message = f"Missing or external artifact: {candidate}"
        raise DerivationError(message)
    if candidate.is_symlink() or any(
        parent.is_symlink() for parent in candidate.parents if parent != root.parent
    ):
        message = f"Symlinked artifact: {candidate}"
        raise DerivationError(message)
    return candidate


def load_provenance(work_dir: Path) -> dict[str, Any]:
    """Validate opt-in identity and ensure the historical source is a different tree."""
    value = read_json(work_dir / PROVENANCE_NAME)
    if value.get("schema") != SCHEMA or value.get("target_work_dir") != str(
        work_dir.resolve()
    ):
        message = "Derived provenance does not identify this workspace."
        raise DerivationError(message)
    source = Path(value["source_work_dir"]).resolve()
    target = work_dir.resolve()
    if (
        target == source
        or target.is_relative_to(source)
        or source.is_relative_to(target)
    ):
        message = "Historical and derived workspaces must be disjoint."
        raise DerivationError(message)
    if value.get("scheduler_policy") != "sequential_all_generations_v1":
        message = "Unsupported derived scheduler policy."
        raise DerivationError(message)
    return value


def derived_external_local_bound(
    work_dir: Path, metadata: dict[str, object], source_generation: int
) -> int | None:
    """Permit promotion only for a seed explicitly bound to a derived provenance."""
    if not (work_dir / PROVENANCE_NAME).is_file():
        return None
    provenance = load_provenance(work_dir)
    if (
        metadata.get("derived_experiment_id") != provenance["experiment_id"]
        or source_generation != provenance["source_active_model_generation"]
    ):
        message = "External seed does not match the derived provenance."
        raise DerivationError(message)
    return int(provenance["source_tree_generation"])


def sync_checkpoint(directory: Path) -> None:
    """Flush a complete checkpoint before committing the journal's restore pointer."""
    for path in directory.rglob("*"):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    for path in [
        *sorted((p for p in directory.rglob("*") if p.is_dir()), reverse=True),
        directory,
        directory.parent,
    ]:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
