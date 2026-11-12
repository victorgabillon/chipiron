"""Opt-in, one-entry derived input reuse; current rows/targets/splits stay current.

The stable exclusive lock owns the prior mmap lifetime and replacement. This is
not source admission or a scientific proof. Every current row is still validated
and decoded by the existing builder. No model, target or split data is stored.
"""

import hashlib
import logging
import os
import pickle
import shutil
import sqlite3
import stat
import time
import uuid
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    assert_artifact_writable,
    file_digest,
    publish_directory_if_absent,
    require,
    sync_directory,
)

from . import flat_tensor_cache as flat
from . import relational_entity_token_cache as relational
from .shared_preparation import _json, _ownership, _record, _write
from .state_split_cache_identity import encoded

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .shared_preparation import _Budget, _Recipe

ENVIRONMENT = "CHIPIRON_SHARED_PREPARATION_REUSE"
DIRECTORY = ENVIRONMENT + "_DIR"
FORMAT = "morpion_prior_prepared_inputs_v1"
LOGGER = logging.getLogger(__name__)
_FILES = {"inputs.pt", "index.sqlite"}
_NAMES = {"flat", "tokens", "relations", "token_lengths", "relation_lengths"}


def requested() -> bool:
    """Resolve the execution-only opt-in without a scientific setting change."""
    value = os.environ.get(ENVIRONMENT, "0")
    require(value in {"0", "1"}, "Input reuse must be 0 or 1")
    return value == "1"


def _integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _limit() -> int:
    raw = os.environ.get(ENVIRONMENT + "_MAX_MB", "2048")
    require(raw.isdecimal() and int(raw) > 0, "Invalid input-reuse byte budget")
    return int(raw) * 2**20


def _identity(recipe: _Recipe, semantics: str) -> dict[str, Any]:
    inputs = recipe.identity()
    # These control CURRENT population/membership, never deterministic inputs.
    for name in ("policy", "fraction", "seed", "max_rows"):
        del inputs[name]
    return {"format": FORMAT, "semantics": semantics, "inputs": inputs}


def _binding(payload: bytes, tensors: tuple[torch.Tensor, ...]) -> str:
    digest = hashlib.sha256(payload)
    for tensor in tensors:
        digest.update(encoded([str(tensor.dtype), list(tensor.shape)]))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _manifest(entry: Path, limit: int) -> dict[str, Any]:
    require(entry.is_dir() and not entry.is_symlink(), "Invalid input entry")
    require(
        {p.name for p in entry.iterdir()} == _FILES | {"complete.json"},
        "Incomplete input entry",
    )
    value = _json(entry / "complete.json")
    require(
        set(value) == {"format", "identity", "rows", "files"}
        and value["format"] == FORMAT
        and _integer(value["rows"])
        and value["rows"] >= 0
        and isinstance(value["identity"], dict)
        and isinstance(value["files"], dict)
        and set(value["files"]) == _FILES,
        "Invalid input completion",
    )
    total = (entry / "complete.json").stat().st_size
    for name in _FILES:
        path = entry / name
        info = path.lstat()
        descriptor = value["files"][name]
        require(
            stat.S_ISREG(info.st_mode)
            and isinstance(descriptor, dict)
            and set(descriptor) == {"bytes", "sha256"}
            and _integer(descriptor["bytes"])
            and info.st_size == descriptor["bytes"],
            "Invalid input payload descriptor",
        )
        total += info.st_size
        require(total <= limit, "Input entry exceeds byte budget")
        require(file_digest(path) == descriptor["sha256"], "Input checksum differs")
    return value


class _Prior:
    def __init__(
        self, entry: Path, manifest: dict[str, Any], max_tokens: int, row_bytes: int
    ) -> None:
        self.entry, self.manifest = entry, manifest
        self.tensors: dict[str, torch.Tensor] = {}
        self.connection: sqlite3.Connection | None = None
        try:
            tensors = torch.load(
                entry / "inputs.pt", map_location="cpu", weights_only=True, mmap=True
            )
            require(
                isinstance(tensors, dict) and set(tensors) == _NAMES,
                "Input tensors differ",
            )
            for name, tensor in tensors.items():
                require(
                    isinstance(tensor, torch.Tensor)
                    and tensor.device.type == "cpu"
                    and tensor.is_contiguous()
                    and not tensor.requires_grad
                    and tensor.dtype
                    == (
                        torch.int64
                        if name.endswith("lengths")
                        else torch.int32
                        if name == "relations"
                        else torch.float32
                    ),
                    "Invalid input tensor type",
                )
            self.tensors = tensors
            f, t, r, tl, rl = (
                tensors[k]
                for k in (
                    "flat",
                    "tokens",
                    "relations",
                    "token_lengths",
                    "relation_lengths",
                )
            )
            n = manifest["rows"]
            require(
                f.shape == (n, 41)
                and t.ndim == 2
                and t.shape[1] == 25
                and r.ndim == 2
                and r.shape[1] == 3
                and tl.shape == rl.shape == (n,),
                "Invalid input shapes",
            )
            self.connection = sqlite3.connect(
                f"{(entry / 'index.sqlite').as_uri()}?mode=ro", uri=True
            )
            self.connection.execute("PRAGMA cache_size=-2048")
            self.connection.execute("PRAGMA mmap_size=0")
            require(
                set(self.connection.execute("SELECT type,name FROM sqlite_schema"))
                == {("table", "inputs"), ("index", "payload_lookup")},
                "Input index schema differs",
            )
            require(
                self.connection.execute("PRAGMA integrity_check").fetchone() == ("ok",),
                "Input index integrity differs",
            )
            require(
                self.connection.execute(
                    "SELECT COUNT(*) FROM inputs WHERE typeof(payload)!='blob' "
                    "OR length(payload)>?",
                    (row_bytes,),
                ).fetchone()
                == (0,),
                "Cached state payload exceeds chunk memory budget",
            )
            count = token_offset = relation_offset = 0
            for row in self.connection.execute("SELECT * FROM inputs ORDER BY ordinal"):
                ordinal, key, payload, to, tn, ro, rn, binding = row
                require(
                    ordinal == count
                    and all(_integer(v) for v in (ordinal, to, tn, ro, rn))
                    and isinstance(payload, bytes)
                    and key == hashlib.sha256(payload).hexdigest()
                    and to == token_offset
                    and ro == relation_offset
                    and 0 < tn <= max_tokens
                    and rn >= 0
                    and count < n
                    and tn == int(tl[count])
                    and rn == int(rl[count])
                    and to + tn <= len(t)
                    and ro + rn <= len(r),
                    "Input key/offset binding differs",
                )
                values = (f[count], t[to : to + tn], r[ro : ro + rn])
                require(
                    sum(v.numel() * v.element_size() for v in values) + 16 <= row_bytes,
                    "Cached input row exceeds chunk memory budget",
                )
                require(
                    all(bool(torch.isfinite(v).all()) for v in values[:2]),
                    "Nonfinite input features",
                )
                require(
                    not rn
                    or (
                        bool(torch.all(values[2][:, :2] >= 0))
                        and bool(torch.all(values[2][:, :2] < tn))
                        and bool(torch.all(values[2][:, 2] >= 0))
                        and bool(torch.all(values[2][:, 2] < 16))
                    ),
                    "Input relation bounds differ",
                )
                require(
                    _binding(payload, values) == binding, "Input tensor binding differs"
                )
                count += 1
                token_offset += tn
                relation_offset += rn
            require(
                count == n and token_offset == len(t) and relation_offset == len(r),
                "Incomplete input index",
            )
        except BaseException:
            self.close()
            raise

    def lookup(self, payload: bytes) -> tuple[torch.Tensor, ...] | None:
        """Gather a fully verified exact-payload row without changing prior data."""
        require(self.connection is not None, "Input cache lease closed")
        assert self.connection is not None
        row = self.connection.execute(
            "SELECT ordinal,token_offset,token_count,relation_offset,relation_count "
            "FROM inputs WHERE key=? AND payload=? ORDER BY ordinal LIMIT 1",
            (hashlib.sha256(payload).hexdigest(), payload),
        ).fetchone()
        if row is None:
            return None
        i, to, tn, ro, rn = row
        return (
            self.tensors["flat"][i],
            self.tensors["tokens"][to : to + tn],
            self.tensors["relations"][ro : ro + rn],
        )

    def close(self) -> None:
        """End the private mapped/index consumer before replacement."""
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        self.tensors.clear()


class _Reuse:
    def __init__(
        self,
        root: Path,
        identity: dict[str, Any],
        budget: _Budget,
        prior: _Prior | None,
        limit: int,
        outcome: str,
    ) -> None:
        self.root, self.identity, self.budget = root, identity, budget
        self.prior, self.limit, self.outcome = prior, limit, outcome
        self.stage = root / f"attempt-{uuid.uuid4().hex}"
        self.stage.mkdir()
        self.connection = sqlite3.connect(self.stage / "index.sqlite")
        self.connection.execute("PRAGMA journal_mode=OFF")
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.execute("PRAGMA cache_size=-2048")
        self.connection.executescript(
            "CREATE TABLE inputs (ordinal INTEGER PRIMARY KEY, "
            "key TEXT NOT NULL,payload BLOB NOT NULL,token_offset INTEGER NOT NULL,"
            "token_count INTEGER NOT NULL,relation_offset INTEGER NOT NULL,"
            "relation_count INTEGER NOT NULL,binding TEXT NOT NULL);"
            "CREATE INDEX payload_lookup ON inputs(key);"
        )
        self.rows = self.tokens = self.relations = self.hits = self.misses = 0
        self.lookup_s = self.index_s = self.publish_s = 0.0
        self.prior_verify_s = 0.0
        self.index_wire_bytes = 0

    @property
    def build_budget(self) -> _Budget:
        """Include the current side index while native staging/export coexist."""
        self.connection.commit()
        return replace(
            self.budget,
            retained_bytes=self.budget.retained_bytes
            + (self.stage / "index.sqlite").stat().st_size,
        )

    def lookup(self, payload: object) -> tuple[torch.Tensor, ...] | None:
        """Count exact hits without changing current row placement or RNG."""
        started = time.perf_counter()
        value = None if self.prior is None else self.prior.lookup(encoded(payload))
        self.lookup_s += time.perf_counter() - started
        self.hits += value is not None
        self.misses += value is None
        return value

    def append(self, payload: object, values: tuple[torch.Tensor, ...]) -> None:
        """Bind each current input row, including duplicates, to its packed offsets."""
        started = time.perf_counter()
        wire = encoded(payload)
        require(
            len(wire) <= self.budget.chunk_bytes,
            "State payload exceeds reuse chunk memory budget",
        )
        # Account for the side index before buffering another payload. SQLite
        # also has a fixed 2 MiB page-cache bound; commit checks exact disk size.
        self.index_wire_bytes += len(wire) + 512
        require(
            self.index_wire_bytes + 2**14 <= self.limit,
            "Input index exceeds reuse budget",
        )
        self.connection.execute(
            "INSERT INTO inputs VALUES (?,?,?,?,?,?,?,?)",
            (
                self.rows,
                hashlib.sha256(wire).hexdigest(),
                wire,
                self.tokens,
                len(values[1]),
                self.relations,
                len(values[2]),
                _binding(wire, values),
            ),
        )
        self.rows += 1
        self.tokens += len(values[1])
        self.relations += len(values[2])
        if self.rows % 128 == 0:
            self.connection.commit()
            require(
                (self.stage / "index.sqlite").stat().st_size <= self.limit,
                "Input index exceeds reuse budget",
            )
        self.index_s += time.perf_counter() - started

    def publish(self, native: Path) -> None:
        """Verify and publish one input-only entry after current native validation."""
        started = time.perf_counter()
        self.connection.commit()
        self.connection.close()
        f = torch.load(
            native / "flat.pt", map_location="cpu", weights_only=True, mmap=True
        )
        r = torch.load(
            native / "relational.pt", map_location="cpu", weights_only=True, mmap=True
        )
        inputs = {
            "flat": f[flat.FLAT_TENSOR_CACHE_INPUT_KEY],
            "tokens": r[relational.RELATIONAL_CACHE_PACKED_INPUT_KEY],
            "relations": r[relational.RELATIONAL_CACHE_PACKED_RELATIONS_KEY],
            "token_lengths": r[relational.RELATIONAL_CACHE_TOKEN_LENGTHS_KEY],
            "relation_lengths": r[relational.RELATIONAL_CACHE_RELATION_LENGTHS_KEY],
        }
        incoming = sum(t.numel() * t.element_size() for t in inputs.values()) + 2**15
        index_bytes = (self.stage / "index.sqlite").stat().st_size
        require(
            incoming + index_bytes <= self.limit, "Input reuse storage budget exceeded"
        )
        native_bytes = sum(p.stat().st_size for p in native.rglob("*") if p.is_file())
        require(
            native_bytes + self.budget.retained_bytes + incoming + index_bytes
            <= self.budget.scratch_bytes,
            "Input reuse coexistence exceeds scratch budget",
        )
        require(
            shutil.disk_usage(self.root).free >= incoming + self.budget.headroom_bytes,
            "Insufficient input cache disk headroom",
        )
        self.budget.check_memory(incoming + 2 * self.budget.chunk_bytes)
        with (self.stage / "inputs.pt").open("xb") as stream:
            torch.save(inputs, stream)
            stream.flush()
            os.fsync(stream.fileno())
        del inputs, f, r
        with (self.stage / "index.sqlite").open("rb") as stream:
            os.fsync(stream.fileno())
        _write(
            self.stage / "complete.json",
            {
                "format": FORMAT,
                "identity": self.identity,
                "rows": self.rows,
                "files": {
                    name: {
                        "bytes": (self.stage / name).stat().st_size,
                        "sha256": file_digest(self.stage / name),
                    }
                    for name in _FILES
                },
            },
        )
        manifest = _manifest(self.stage, self.limit)
        # Full binding validation also certifies our produced packed input export.
        with_prior = _Prior(
            self.stage,
            manifest,
            self.identity["inputs"]["entity_max_tokens"],
            self.budget.chunk_bytes,
        )
        with_prior.close()
        if self.prior is not None:
            require(
                _manifest(self.prior.entry, self.limit) == self.prior.manifest,
                "Prior inputs changed during gathering",
            )
            self.prior.close()
            self.prior = None
        entry = self.root / "entry"
        if entry.exists():
            shutil.rmtree(entry)
            sync_directory(self.root)
        sync_directory(self.stage)
        publish_directory_if_absent(self.stage, entry, self.root)
        self.publish_s = time.perf_counter() - started
        _record(
            self.root,
            {
                "format": FORMAT,
                "outcome": self.outcome,
                "rows": self.rows,
                "hits": self.hits,
                "reencoded_rows": self.misses,
                "lookup_s": self.lookup_s,
                "index_binding_s": self.index_s,
                "publish_verify_s": self.publish_s,
                "prior_verify_s": self.prior_verify_s,
                "cache_bytes": sum(p.stat().st_size for p in entry.iterdir()),
                "retained_entries": 1,
                "max_cache_bytes": self.limit,
                "prior_bytes": self.budget.retained_bytes,
            },
        )

    def close(self) -> None:
        """Release only owned resources; failed staging remains nonauthoritative."""
        self.connection.close()
        if self.prior is not None:
            self.prior.close()


@contextmanager
def prior_inputs(
    rows: Path,
    recipe: _Recipe,
    budget: _Budget,
    semantics: str,
    *,
    deadline: float | None = None,
) -> Iterator[_Reuse | None]:
    """No caller authority changes; one global derived entry, one synchronous writer."""
    if not requested():
        yield None
        return
    if torch.get_default_dtype() != torch.float32:
        LOGGER.info("input_reuse fallback=unsupported-default-dtype")
        yield None
        return
    raw = os.environ.get(DIRECTORY)
    require(
        bool(raw) and Path(str(raw)).is_absolute(), "Input reuse requires absolute DIR"
    )
    root = Path(str(raw))
    require(
        not any(p.is_symlink() for p in (root, *root.parents)),
        "Symlinked reuse storage",
    )
    require(
        root != rows.parent and rows.parent not in root.parents,
        "Input reuse must outlive the attempt input directory",
    )
    assert_artifact_writable(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    limit = _limit()
    identity = _identity(recipe, semantics)
    with _ownership(root, format_kind=FORMAT, deadline=deadline):
        started = time.perf_counter()
        prior = None
        outcome = "absent"
        entry = root / "entry"
        if entry.exists():
            require(
                entry.is_dir() and not entry.is_symlink(), "Unsafe prior input entry"
            )
            require(
                all(not path.is_symlink() for path in entry.iterdir()),
                "Symlink in prior input entry",
            )
            failures = (
                ValueError,
                OSError,
                sqlite3.DatabaseError,
                RuntimeError,
                pickle.UnpicklingError,
                EOFError,
            )
            manifest = None
            try:
                manifest = _manifest(entry, limit)
            except failures as exc:
                outcome = "corrupt-rebuild"
                LOGGER.info(
                    "input_reuse fallback=%s error=%s", outcome, type(exc).__name__
                )
            if manifest is not None:
                if manifest["identity"] != identity:
                    outcome = "incompatible"
                else:
                    bounded = replace(
                        budget,
                        retained_bytes=budget.retained_bytes
                        + sum(p.stat().st_size for p in entry.iterdir()),
                        resident_bytes=manifest["files"]["inputs.pt"]["bytes"],
                    )
                    bounded.check_memory(2 * budget.chunk_bytes)
                    try:
                        prior = _Prior(
                            entry,
                            manifest,
                            recipe.entity_max_tokens,
                            budget.chunk_bytes,
                        )
                        outcome = "hit"
                    except failures as exc:
                        outcome = "corrupt-rebuild"
                        LOGGER.info(
                            "input_reuse fallback=%s error=%s",
                            outcome,
                            type(exc).__name__,
                        )
            if prior is None:
                shutil.rmtree(entry)
                sync_directory(root)
        bounded = (
            budget
            if prior is None
            else replace(
                budget,
                retained_bytes=budget.retained_bytes
                + sum(p.stat().st_size for p in prior.entry.iterdir()),
                resident_bytes=prior.manifest["files"]["inputs.pt"]["bytes"],
            )
        )
        bounded.check_memory(2 * budget.chunk_bytes)
        session = _Reuse(root, identity, bounded, prior, limit, outcome)
        session.prior_verify_s = time.perf_counter() - started
        try:
            yield session
        finally:
            session.close()
