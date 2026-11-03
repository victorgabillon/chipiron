"""Attempt-owned serial preparation; one complete bundle and stable consumer lock.

This is operational storage, never Training authority. The caller retains the
accepted source/model pins. One exclusive preparation lock covers construction
and consumption, so this first serial slice also serializes concurrent readers.
No process, thread policy, accepted artifact or general cache service is created.
"""

import fcntl
import hashlib
import json
import logging
import os
import re
import shutil
import stat
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chipiron.environments.morpion.bootstrap.candidate_resources import (
    available_memory_mb,
)
from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    assert_artifact_writable,
    file_digest,
    publish_directory_if_absent,
    require,
    sync_directory,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.entity_relations import (
    MORPION_ENTITY_RELATION_SCHEMA,
    MORPION_ENTITY_RELATION_TYPE_COUNT,
    is_relational_entity_token_model_kind,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.entity_tokens import (
    MORPION_ENTITY_TOKEN_FEATURE_DIM,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.feature_schema import (
    MORPION_CANONICAL_FEATURE_NAMES,
)

from . import flat_tensor_cache as flat
from . import relational_entity_token_cache as relational
from .state_split import POLICY, StateSplit, _install_prepared_split
from .state_split_cache_identity import encoded, semantic_fingerprint

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .args import MorpionTrainingArgs

LOGGER = logging.getLogger(__name__)
FORMAT = "morpion_shared_preparation_v1"
ENVIRONMENT = "CHIPIRON_SHARED_PREPARATION"
LOCK_TIMEOUT_S = 600.0
_PRIVATE = re.compile(r"(?:attempt|operation)-[0-9a-f]{32}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_FILES = {
    "flat.pt",
    "flat.manifest.json",
    "relational.pt",
    "relational.manifest.json",
    "split.json",
}
_SPLIT_MAX_BYTES = 64 * 2**20


@dataclass(frozen=True)
class _Recipe:
    fraction: float
    seed: int
    max_rows: int | None
    entity_max_tokens: int = 1536

    def identity(self) -> dict[str, object]:
        """Bind exact split and representation semantics, excluding operational knobs."""
        return {
            "policy": POLICY,
            "fraction": self.fraction.hex(),
            "seed": self.seed,
            "max_rows": self.max_rows,
            "entity_max_tokens": self.entity_max_tokens,
            "flat_features": list(MORPION_CANONICAL_FEATURE_NAMES),
            "flat_format": flat.FLAT_TENSOR_CACHE_FORMAT,
            "relational_format": relational.MORPION_RELATIONAL_ENTITY_TOKEN_CACHE_FORMAT,
            "token_dim": MORPION_ENTITY_TOKEN_FEATURE_DIM,
            "relation_schema": MORPION_ENTITY_RELATION_SCHEMA,
            "relation_types": MORPION_ENTITY_RELATION_TYPE_COUNT,
            "feature_dtype": "torch.float32",
            "length_dtype": "torch.int64",
            "relation_dtype": "torch.int32",
            "global_geometry": "none",
            "edge_tokens": "drawn_only",
            "latent_window": "none",
        }


def _megabytes(name: str, default: int) -> int:
    raw = os.environ.get("CHIPIRON_SHARED_PREPARATION_" + name, str(default))
    require(raw.isdecimal() and int(raw) > 0, "Invalid shared-preparation byte budget")
    return int(raw) * 2**20


@dataclass(frozen=True)
class _Budget:
    packed_bytes: int = 2 * 2**30
    scratch_bytes: int = 6 * 2**30
    chunk_bytes: int = 64 * 2**20
    headroom_bytes: int = 512 * 2**20
    retained_bytes: int = 0
    resident_bytes: int = 0

    @classmethod
    def configured(cls) -> _Budget:
        """Resolve explicit byte limits without changing scientific row limits."""
        return cls(
            packed_bytes=_megabytes("MAX_PACKED_MB", 2048),
            scratch_bytes=_megabytes("MAX_SCRATCH_MB", 6144),
        )

    def check_memory(self, allocation: int) -> None:
        """Reserve host/container capacity before a bounded allocation."""
        require(
            available_memory_mb() * 2**20
            >= allocation + self.resident_bytes + self.headroom_bytes,
            "Insufficient memory headroom for shared preparation",
        )

    def check_stage(
        self, directory: Path, packed: int, staged: int, incoming: int
    ) -> None:
        """Bound staged bytes and live chunk allocations before writing."""
        require(packed <= self.packed_bytes, "Shared packing byte budget exceeded")
        require(
            staged + 2 * incoming + 2 * packed + self.retained_bytes
            <= self.scratch_bytes,
            "Shared staging/export disk budget exceeded",
        )
        require(
            shutil.disk_usage(directory).free >= 2 * incoming + self.headroom_bytes,
            "Insufficient free disk for shared staging",
        )
        self.check_memory(2 * self.chunk_bytes)

    def check_export(self, directory: Path, packed: int) -> None:
        """Include the full compatible packed payload and simultaneous staging."""
        staged = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
        require(
            packed <= self.packed_bytes
            and staged + 2 * packed + 2**25 + self.retained_bytes <= self.scratch_bytes,
            "Shared packed export exceeds declared budget",
        )
        require(
            shutil.disk_usage(directory).free >= 2 * packed + self.headroom_bytes,
            "Insufficient free disk for compatible export",
        )
        self.check_memory(packed + 2 * self.chunk_bytes)


@dataclass
class _Lease:
    pid: int
    active: bool = True


@dataclass(frozen=True)
class _PreparedInputs:
    rows: Path
    source_sha256: str
    recipe: _Recipe
    directory: Path
    split: StateSplit
    lease: _Lease

    @property
    def flat_paths(self) -> flat.FlatTensorCachePaths:
        """Return existing-format flat artifacts within the leased bundle."""
        return flat.FlatTensorCachePaths(
            self.directory / "flat.pt", self.directory / "flat.manifest.json"
        )

    @property
    def relational_paths(self) -> relational.MorpionRelationalEntityTokenCachePaths:
        """Return existing-format relational artifacts within the leased bundle."""
        return relational.MorpionRelationalEntityTokenCachePaths(
            self.directory / "relational.pt",
            self.directory / "relational.manifest.json",
        )

    def prepare_split(
        self, args: MorpionTrainingArgs, max_rows: int | None
    ) -> tuple[MorpionTrainingArgs, dict[str, object]]:
        """Verify the consumer recipe and seed the original split API."""
        require(
            self.lease.active and self.lease.pid == os.getpid(),
            "Prepared input lease is not active in this process",
        )
        require(
            Path(args.dataset_file).resolve() == self.rows,
            "Prepared input source path differs",
        )
        require(
            args.stable_validation_split
            and (args.validation_fraction, args.validation_seed, max_rows)
            == (self.recipe.fraction, self.recipe.seed, self.recipe.max_rows),
            "Prepared split recipe differs",
        )
        require(
            flat.is_flat_morpion_training_model_kind(args.model_kind)
            or (
                is_relational_entity_token_model_kind(args.model_kind)
                and args.entity_max_tokens == self.recipe.entity_max_tokens
                and args.legacy_graph_tokens is None
            ),
            "Prepared representation differs",
        )
        return _install_prepared_split(
            args, max_rows, source_sha256=self.source_sha256, value=self.split
        )


def _read(path: Path, limit: int) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        require(
            stat.S_ISREG(info.st_mode) and info.st_size <= limit,
            "Invalid or oversized preparation metadata",
        )
        result = stream.read(limit + 1)
    require(len(result) <= limit, "Preparation metadata grew beyond its bound")
    return result


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        require(key not in value, "Duplicate preparation JSON key")
        value[key] = item
    return value


def _json(path: Path, limit: int = 2**20) -> dict[str, Any]:
    value = json.loads(_read(path, limit), object_pairs_hook=_pairs)
    require(isinstance(value, dict), "Preparation metadata must be an object")
    assert isinstance(value, dict)
    return value


def _write(path: Path, value: object) -> None:
    with path.open("xb") as stream:
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def _ownership(root: Path, *, format_kind: str = FORMAT) -> Iterator[None]:
    assert_artifact_writable(root)
    require(not root.is_symlink(), "Preparation root cannot be a symlink")
    root.mkdir(exist_ok=True)
    descriptor = os.open(root / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        require(stat.S_ISREG(os.fstat(descriptor).st_mode), "Invalid preparation lock")
        deadline = time.monotonic() + LOCK_TIMEOUT_S
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    message = "Shared-preparation lock wait exhausted"
                    raise TimeoutError(message) from None
                time.sleep(0.05)
        owner = root / "OWNER"
        if not owner.exists():
            require(
                {p.name for p in root.iterdir()} <= {".lock", ".owner-pending"},
                "Refusing foreign preparation storage",
            )
            pending = root / ".owner-pending"
            if pending.exists():
                require(
                    not pending.is_symlink() and pending.is_file(),
                    "Invalid preparation owner staging",
                )
                pending.unlink()
            _write(pending, {"format": format_kind})
            os.rename(pending, owner)
        require(_json(owner) == {"format": format_kind}, "Invalid preparation owner")
        sync_directory(root)
        sync_directory(root.parent)
        for path in root.iterdir():
            require(not path.is_symlink(), "Symlink in preparation storage")
            if path.name in {".lock", "OWNER", "entry", "operation.json"}:
                continue
            require(
                _PRIVATE.fullmatch(path.name) is not None,
                "Unknown preparation artifact",
            )
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        yield
    finally:
        os.close(descriptor)


def _verify(directory: Path, budget: _Budget) -> dict[str, Any]:
    """Validate completed ownership/content before loading or replacing anything."""
    require(
        not directory.is_symlink() and directory.is_dir(),
        "Invalid completed preparation directory",
    )
    require(
        {p.name for p in directory.iterdir()} == _FILES | {"complete.json"},
        "Incomplete or unknown prepared output",
    )
    complete = _json(directory / "complete.json")
    require(
        set(complete) == {"format", "identity", "key", "files", "statistics"}
        and complete["format"] == FORMAT,
        "Unsupported preparation completion",
    )
    require(
        isinstance(complete["identity"], dict)
        and complete["key"]
        == hashlib.sha256(encoded(complete["identity"])).hexdigest(),
        "Prepared identity checksum mismatch",
    )
    inventory = complete["files"]
    require(
        isinstance(inventory, dict) and set(inventory) == _FILES,
        "Prepared payload inventory differs",
    )
    total = 0
    for name in sorted(_FILES):
        path = directory / name
        info = path.lstat()
        require(stat.S_ISREG(info.st_mode), "Prepared payload is not a regular file")
        require(
            isinstance(inventory[name], dict)
            and set(inventory[name]) == {"bytes", "sha256"},
            "Prepared payload descriptor differs",
        )
        require(
            info.st_size == inventory[name]["bytes"], "Prepared payload size mismatch"
        )
        require(
            info.st_size
            <= (
                budget.packed_bytes + 2**25
                if name.endswith(".pt")
                else _SPLIT_MAX_BYTES
            ),
            "Prepared payload exceeds consumption budget",
        )
        total += info.st_size
        require(
            file_digest(path) == inventory[name]["sha256"],
            "Prepared payload checksum mismatch",
        )
    require(total <= budget.scratch_bytes, "Prepared output exceeds scratch budget")
    return complete


def _split(directory: Path, recipe: _Recipe) -> StateSplit:
    value = _json(directory / "split.json", _SPLIT_MAX_BYTES)
    require(
        set(value) == {"indices", "rows", "population"}, "Prepared split fields differ"
    )
    count, indices, population = value["rows"], value["indices"], value["population"]
    require(
        isinstance(count, int)
        and not isinstance(count, bool)
        and count >= 0
        and (recipe.max_rows is None or count <= recipe.max_rows),
        "Invalid prepared split row count",
    )
    require(isinstance(indices, list), "Invalid prepared split membership")
    previous = -1
    for index in indices:
        require(
            isinstance(index, int)
            and not isinstance(index, bool)
            and previous < index < count,
            "Invalid prepared split index",
        )
        previous = index
    require(
        isinstance(population, str) and _DIGEST.fullmatch(population) is not None,
        "Invalid prepared population digest",
    )
    return StateSplit(frozenset(indices), count, population)


def _validate_native(prepared: _PreparedInputs, budget: _Budget) -> None:
    sizes = sum(
        p.stat().st_size
        for p in (
            prepared.flat_paths.tensor_path,
            prepared.relational_paths.tensor_path,
        )
    )
    budget.check_memory(sizes + 2 * budget.chunk_bytes)
    loaded_flat = flat.load_flat_tensor_cache(
        rows_path=prepared.rows,
        max_rows=prepared.recipe.max_rows,
        _paths=prepared.flat_paths,
    )
    require(
        loaded_flat.manifest.row_count == prepared.split.row_count,
        "Flat/split row counts differ",
    )
    del loaded_flat
    loaded_relational = relational.load_relational_entity_token_cache(
        rows_path=prepared.rows,
        max_rows=prepared.recipe.max_rows,
        entity_max_tokens=prepared.recipe.entity_max_tokens,
        _paths=prepared.relational_paths,
    )
    require(
        loaded_relational.manifest.row_count == prepared.split.row_count,
        "Relational/split row counts differ",
    )


def _record(root: Path, value: dict[str, Any]) -> None:
    pending = root / f"operation-{uuid.uuid4().hex}"
    _write(pending, value)
    os.replace(pending, root / "operation.json")
    sync_directory(root)
    LOGGER.info("shared_preparation %s", encoded(value).decode())


@contextmanager
def prepare_shared_inputs(
    rows_path: Path,
    *,
    fraction: float,
    seed: int,
    max_rows: int | None,
    row_chunk_size: int,
    budget: _Budget | None = None,
) -> Iterator[_PreparedInputs | None]:
    """Prepare one compatible flat+relational bundle, preserving native warm hits.

    Existing valid legacy tensor caches retain their original consumers. Their
    path/stat identities cannot certify a new exact-content bundle, so partial
    legacy availability takes that explicit fallback too. Own completed bundles
    must be intact; corrupt completions never silently rebuild.
    """
    from .shared_preparation_builder import build_shared_bundle
    from .shared_preparation_reuse import prior_inputs, requested

    requested()  # Reject malformed execution configuration even on a warm hit.
    require(rows_path.suffix == ".jsonl", "Shared preparation supports JSONL only")
    require(
        isinstance(row_chunk_size, int)
        and not isinstance(row_chunk_size, bool)
        and row_chunk_size > 0,
        "Invalid preparation chunk size",
    )
    require(
        isinstance(seed, int) and not isinstance(seed, bool) and 0 <= fraction < 1,
        "Invalid preparation split recipe",
    )
    require(
        max_rows is None
        or (
            isinstance(max_rows, int)
            and not isinstance(max_rows, bool)
            and max_rows >= 0
        ),
        "Invalid preparation row limit",
    )
    rows = rows_path.resolve()
    root = rows.with_name(f".{rows.name}.shared-preparation-v1")
    recipe = _Recipe(float(fraction), seed, max_rows)
    limits = _Budget.configured() if budget is None else budget
    started = time.perf_counter()
    with _ownership(root):
        wait_s = time.perf_counter() - started
        tick = time.perf_counter()
        source_sha256 = file_digest(rows)
        source_hash_s = time.perf_counter() - tick
        semantics = semantic_fingerprint()
        identity = {
            "format": FORMAT,
            "source": source_sha256,
            "semantics": semantics,
            "recipe": recipe.identity(),
        }
        key = hashlib.sha256(encoded(identity)).hexdigest()
        identity_s = time.perf_counter() - tick
        entry = root / "entry"
        tick = time.perf_counter()
        complete = _verify(entry, limits) if entry.exists() else None
        integrity_verify_s = time.perf_counter() - tick
        native_verify_s = 0.0
        resolved_budget = {
            "packed_bytes": limits.packed_bytes,
            "scratch_bytes": limits.scratch_bytes,
            "chunk_bytes": limits.chunk_bytes,
            "headroom_bytes": limits.headroom_bytes,
            "requested_chunk_rows": row_chunk_size,
            "resolved_chunk_rows": min(row_chunk_size, 128),
        }
        statistics: dict[str, Any] = {}
        outcome = "hit"
        if complete is None or complete["identity"] != identity:
            # Validate/load at most one original cache at a time; don't adopt a
            # legacy path/stat cache as an exact-content certified feature store.
            native_paths = (
                flat.default_flat_tensor_cache_paths(
                    rows, max_rows=max_rows
                ).tensor_path,
                relational.default_relational_entity_token_cache_paths(
                    rows, max_rows=max_rows, entity_max_tokens=recipe.entity_max_tokens
                ).tensor_path,
            )
            for native_path in native_paths:
                if native_path.exists():
                    size = native_path.stat().st_size
                    require(
                        size <= limits.packed_bytes + 2**25,
                        "Native cache exceeds shared consumption budget",
                    )
                    limits.check_memory(size + 2 * limits.chunk_bytes)
            native_flat = flat.flat_tensor_cache_is_valid(
                rows_path=rows, max_rows=max_rows
            )
            native_relational = relational.relational_entity_token_cache_is_valid(
                rows_path=rows,
                max_rows=max_rows,
                entity_max_tokens=recipe.entity_max_tokens,
            )
            if native_flat or native_relational:
                require(
                    file_digest(rows) == source_sha256, "Preparation source changed"
                )
                _record(
                    root,
                    {
                        "key": key,
                        "outcome": "native-warm-fallback",
                        "budget": resolved_budget,
                        "flat_hit": native_flat,
                        "relational_hit": native_relational,
                        "elapsed_s": time.perf_counter() - started,
                    },
                )
                yield None
                require(
                    file_digest(rows) == source_sha256,
                    "Preparation source changed during native consumption",
                )
                return
            if entry.exists():
                shutil.rmtree(entry)
                sync_directory(root)
            attempt = root / f"attempt-{uuid.uuid4().hex}"
            attempt.mkdir()
            with prior_inputs(rows, recipe, limits, semantics) as reuse:
                active_limits = limits if reuse is None else reuse.budget
                split, statistics = build_shared_bundle(
                    rows, attempt, recipe, active_limits, row_chunk_size, reuse=reuse
                )
                require(
                    file_digest(rows) == source_sha256,
                    "Preparation source changed during build",
                )
                inventory = {
                    name: {
                        "bytes": (attempt / name).stat().st_size,
                        "sha256": file_digest(attempt / name),
                    }
                    for name in sorted(_FILES)
                }
                _write(
                    attempt / "complete.json",
                    {
                        "format": FORMAT,
                        "identity": identity,
                        "key": key,
                        "files": inventory,
                        "statistics": statistics,
                    },
                )
                tick = time.perf_counter()
                _verify(attempt, active_limits)
                integrity_verify_s += time.perf_counter() - tick
                temporary = _PreparedInputs(
                    rows, source_sha256, recipe, attempt, split, _Lease(os.getpid())
                )
                tick = time.perf_counter()
                _validate_native(temporary, active_limits)
                native_verify_s = time.perf_counter() - tick
                temporary.lease.active = False
                if reuse is not None:
                    reuse.publish(attempt)
                    statistics.update(
                        reuse_hits=reuse.hits,
                        reuse_reencoded_rows=reuse.misses,
                        reuse_lookup_s=reuse.lookup_s,
                        reuse_index_binding_s=reuse.index_s,
                        reuse_publish_verify_s=reuse.publish_s,
                        reuse_prior_verify_s=reuse.prior_verify_s,
                        reuse_previous_bytes=reuse.budget.retained_bytes,
                    )
                sync_directory(attempt)
                publish_directory_if_absent(attempt, entry, root)
            outcome = "miss"
        split = _split(entry, recipe)
        lease = _Lease(os.getpid())
        prepared = _PreparedInputs(rows, source_sha256, recipe, entry, split, lease)
        if outcome == "hit":
            tick = time.perf_counter()
            _validate_native(prepared, limits)
            native_verify_s = time.perf_counter() - tick
        require(
            file_digest(rows) == source_sha256,
            "Preparation source changed before consumption",
        )
        _record(
            root,
            {
                "key": key,
                "outcome": outcome,
                "lock_and_namespace_s": wait_s,
                "identity_s": identity_s,
                "source_hash_s": source_hash_s,
                "integrity_verify_s": integrity_verify_s,
                "native_verify_s": native_verify_s,
                "budget": resolved_budget,
                "statistics": statistics,
                "elapsed_s": time.perf_counter() - started,
            },
        )
        try:
            yield prepared
            require(
                file_digest(rows) == source_sha256,
                "Preparation source changed during consumption",
            )
        finally:
            lease.active = False
