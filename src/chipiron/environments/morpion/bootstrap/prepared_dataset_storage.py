"""Bounded operational Dataset records and artifact ownership, never scientific authority."""

import fcntl
import os
import stat
import time
from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from .continuous_contract import Accepted, Context, Ref
from .derived.provenance import inside
from .generation_publication_io import file_digest, object_digest, read_object, require
from .worker_result_io import workspace_epoch

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from .bootstrap_paths import MorpionBootstrapPaths
    from .run_state import MorpionBootstrapRunState

KEY = "prepared_dataset_v1"
ROOT = "pipeline/prepared_inputs_v1"
FILES = frozenset({
    "flat.pt",
    "flat.manifest.json",
    "relational.pt",
    "relational.manifest.json",
    "split.json",
})
PHASES = frozenset({"PREPARING", "COMPLETE", "CONSUMING", "SUSPENDED", "RETIRED"})
FIELDS = frozenset({
    "dataset_claim_id",
    "schema",
    "accepted",
    "source",
    "context",
    "recipe",
    "rows",
    "generation",
    "phase",
    "token",
    "owner",
    "deadline",
    "utc_deadline",
    "reserved_bytes",
    "ready",
})


def entries(state: MorpionBootstrapRunState) -> dict[str, dict[str, Any]]:
    """Decode at most two live roots; retired audit descriptors stay on disk."""
    data = state.metadata.get(KEY, {})
    require(isinstance(data, dict) and len(data) <= 2, "Invalid prepared Dataset slots")
    for key, record in data.items():
        require(
            isinstance(record, dict) and frozenset(record) == FIELDS,
            "Invalid prepared Dataset record",
        )
        accepted = Accepted.from_metadata(record["accepted"])
        require(
            key == accepted.descriptor_sha256 and record["schema"] == 1,
            "Prepared Dataset key differs",
        )
        Context.from_metadata(record["context"])
        Ref.from_metadata(record["rows"])
        require(record["phase"] in PHASES, "Invalid prepared Dataset phase")
        require(
            isinstance(record["recipe"], dict) and isinstance(record["source"], dict),
            "Invalid preparation binding",
        )
        require(
            isinstance(record["reserved_bytes"], int)
            and not isinstance(record["reserved_bytes"], bool)
            and record["reserved_bytes"] >= 0,
            "Invalid preparation reservation",
        )
        require(
            isinstance(record["generation"], int)
            and not isinstance(record["generation"], bool)
            and record["generation"] > 0,
            "Invalid preparation generation",
        )
    return dict(data)


def context(
    paths: MorpionBootstrapPaths, state: MorpionBootstrapRunState
) -> dict[str, Any]:
    """Bind the real workspace/configuration and existing schedule, not future policy."""
    from .learning_schedule import schedule_from_metadata

    schedule = schedule_from_metadata(state.metadata)
    return Context(
        object_digest(workspace_epoch(state)),
        file_digest(paths.bootstrap_config_path),
        object_digest(schedule.to_metadata() if schedule is not None else None),
    ).to_metadata()


def updated(
    state: MorpionBootstrapRunState, data: dict[str, dict[str, Any]]
) -> MorpionBootstrapRunState:
    """Return the proposed operational update for the caller's one owned save."""
    result = replace(state, metadata={**state.metadata, KEY: data})
    entries(result)
    return result


def directory(paths: MorpionBootstrapPaths, key: str) -> Path:
    """Reject unsafe roots before creation as well as during reads."""
    Ref(f"{ROOT}/{key}/rows.jsonl", key, 0)
    root = paths.work_dir / ROOT / key
    require(
        not any(p.is_symlink() for p in (root, *root.parents)),
        "Symlinked prepared Dataset root",
    )
    return root


@contextmanager
def lease(
    paths: MorpionBootstrapPaths, key: str, *, deadline: float | None = None
) -> Iterator[Path]:
    """Stable exclusive inode; optional bounded waits happen outside Growth ownership."""
    root = directory(paths, key)
    root.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(root / ".lease", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        require(
            stat.S_ISREG(os.fstat(descriptor).st_mode), "Invalid prepared Dataset lease"
        )
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if deadline is None or time.monotonic() >= deadline:
                    raise
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        yield root
    finally:
        os.close(descriptor)


def reference(paths: MorpionBootstrapPaths, path: Path) -> dict[str, Any]:
    """Hash an ordinary contained file outside tree ownership."""
    relative = paths.relative_to_work_dir(path)
    checked = inside(paths.work_dir, relative)
    require(stat.S_ISREG(checked.lstat().st_mode), "Prepared artifact is not a file")
    return Ref(relative, file_digest(checked), checked.stat().st_size).to_metadata()


def verify_reference(paths: MorpionBootstrapPaths, value: object) -> Path:
    """Authenticate exact file bytes before any native consumer opens them."""
    ref = Ref.from_metadata(value)
    path = inside(paths.work_dir, ref.path)
    require(
        stat.S_ISREG(path.lstat().st_mode)
        and path.stat().st_size == ref.bytes
        and file_digest(path) == ref.sha256,
        "Prepared artifact changed",
    )
    return path


def ready_descriptor(
    paths: MorpionBootstrapPaths, record: dict[str, Any]
) -> dict[str, Any]:
    """Verify the complete five-file inventory and stable path binding."""
    key = record["accepted"]["descriptor_sha256"]
    root = directory(paths, key)
    path = verify_reference(paths, record["ready"])
    require(path == root / "ready.json", "Prepared ready path differs")
    value = read_object(path)
    require(
        set(value)
        == {
            "schema",
            "accepted",
            "source",
            "context",
            "recipe",
            "rows",
            "native",
            "files",
        }
        and value["schema"] == "operational_prepared_dataset_v1",
        "Invalid prepared descriptor",
    )
    for field in ("accepted", "source", "context", "recipe"):
        require(value[field] == record[field], "Prepared descriptor binding differs")
    rows = verify_reference(paths, value["rows"])
    require(
        rows == root / "rows.jsonl"
        and value["rows"]["sha256"] == record["rows"]["sha256"]
        and value["rows"]["bytes"] == record["rows"]["bytes"],
        "Prepared rows differ",
    )
    native = root / ".rows.jsonl.shared-preparation-v1" / "entry"
    require(
        verify_reference(paths, value["native"]) == native / "complete.json",
        "Prepared native path differs",
    )
    require(
        isinstance(value["files"], dict) and frozenset(value["files"]) == FILES,
        "Prepared inventory differs",
    )
    for name in FILES:
        require(
            verify_reference(paths, value["files"][name]) == native / name,
            "Prepared inventory path differs",
        )
    return value


def settle_owned(
    paths: MorpionBootstrapPaths, state: MorpionBootstrapRunState
) -> MorpionBootstrapRunState:
    """After managed native quiescence, suspend unfinished operations in the same save."""
    data = entries(state)
    for key, record in data.items():
        with lease(paths, key):  # Nonblocking, including under Growth ownership.
            if record["phase"] == "PREPARING":
                data[key] = {**record, "phase": "SUSPENDED"}
            elif record["phase"] == "CONSUMING":
                # Construction already completed. A new consumer must validate it
                # again, but does not need a renewed producer job allowance.
                data[key] = {**record, "phase": "COMPLETE"}
    return updated(state, data) if data else state


def require_settled(
    paths: MorpionBootstrapPaths, state: MorpionBootstrapRunState
) -> None:
    """Refuse live operational authority at the quiescent fork boundary."""
    for key, record in entries(state).items():
        require(
            record["phase"] in {"COMPLETE", "RETIRED"},
            "Unsettled prepared Dataset prevents fork",
        )
        with lease(paths, key):
            pass
