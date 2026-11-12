"""Immutable v1 wire records for the *unwired* continuous Dataset handoff.

These records validate structure, not authority. The future owned publisher must
verify bytes, source publications, claims, budgets, parents, leases and managed
sessions before committing a proposed state in its single durable save. Promotion
bodies, lifecycle recovery and runtime policy dispatch are intentionally absent.
"""

import json
import re
from dataclasses import asdict, dataclass, fields
from pathlib import PurePosixPath
from typing import Any, ClassVar, Self

from .generation_publication_io import object_digest


class ContractError(ValueError):
    """Malformed wire evidence or a conflicting proposed transition."""


def require(condition: bool, message: str) -> None:
    """Reject ambiguity without modifying any input."""
    if not condition:
        raise ContractError(message)


def integer(value: object, minimum: int = 0) -> None:
    """Reject bool, nonintegral and nonfinite numeric evidence."""
    require(
        isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
        "Invalid contract integer.",
    )


def _text(value: object) -> None:
    require(isinstance(value, str) and bool(value.strip()), "Empty contract identity.")


def _digest(value: object) -> None:
    require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        "Invalid SHA256 digest.",
    )


def _path(value: str) -> None:
    _text(value)
    require(
        not PurePosixPath(value).is_absolute()
        and all(part not in {"", ".", ".."} for part in value.split("/"))
        and not any(ord(c) < 32 or c in "\\:" for c in value),
        "Expected a contained lexical POSIX path (symlinks require caller IO checks).",
    )


@dataclass(frozen=True, slots=True)
class _Wire:
    """Small common serialization helpers, without a runtime schema registry."""

    def to_metadata(self) -> dict[str, Any]:
        """Return a detached JSON object."""
        return asdict(self)

    @classmethod
    def from_metadata(cls, value: object) -> Self:
        """Decode a flat record with exact keys; constructors validate values."""
        return cls(**_fields(value, cls))


def _fields(value: object, kind: type[_Wire]) -> dict[str, Any]:
    require(isinstance(value, dict), "Expected a contract object.")
    assert isinstance(value, dict)
    require(
        set(value) == {f.name for f in fields(kind)}, "Unknown/missing record keys."
    )
    return dict(value)


def _optional[T: _Wire](kind: type[T], value: object) -> T | None:
    return None if value is None else kind.from_metadata(value)


@dataclass(frozen=True, slots=True)
class Policy(_Wire):
    """The reviewed v1 choices; constructing this does not enable a schedule."""

    kind: str = "bounded_continuous_v1"
    max_teacher_epoch_lag: int = 1
    max_source_age_cycles: int = 8
    max_unresolved_growth_cycles: int = 8
    max_promotion_growth_cycles: int = 2
    activation: str = "owned_cold_full"
    selection: str = "newest_complete_once"
    slots: tuple[int, ...] = (1, 1, 1, 1, 1, 1)

    def __post_init__(self) -> None:
        """Validate direct construction as strictly as decoded wire input."""
        require(self.kind == "bounded_continuous_v1", "Unsupported continuous policy.")
        for value, expected in [
            (self.max_teacher_epoch_lag, 1),
            (self.max_source_age_cycles, 8),
            (self.max_unresolved_growth_cycles, 8),
            (self.max_promotion_growth_cycles, 2),
        ]:
            integer(value, 1)
            require(value == expected, "Unreviewed v1 policy limit.")
        require(self.activation == "owned_cold_full", "Unsupported activation policy.")
        require(
            self.selection == "newest_complete_once", "Unsupported selection policy."
        )
        require(
            isinstance(self.slots, tuple) and len(self.slots) == 6, "Invalid slots."
        )
        for slot in self.slots:
            integer(slot, 1)
            require(slot == 1, "Only one item per slot is supported.")

    def to_metadata(self) -> dict[str, Any]:
        """Keep the wire's JSON array separate from the immutable tuple."""
        return {**asdict(self), "slots": list(self.slots)}

    @classmethod
    def from_metadata(cls, value: object) -> Self:
        """Require explicit policy fields; never infer missing wire defaults."""
        data = _fields(value, cls)
        require(isinstance(data["slots"], list), "Slots must be a JSON array.")
        data["slots"] = tuple(data["slots"])
        return cls(**data)

    def digest(self) -> str:
        """Use the existing publisher's canonical identity."""
        return object_digest(self.to_metadata())


@dataclass(frozen=True, slots=True)
class Context(_Wire):
    """Expected workspace, configuration and scheduling identity."""

    workspace_epoch_sha256: str
    config_sha256: str
    policy_sha256: str

    def __post_init__(self) -> None:
        """Validate direct construction as strictly as decoded wire input."""
        for value in (
            self.workspace_epoch_sha256,
            self.config_sha256,
            self.policy_sha256,
        ):
            _digest(value)


@dataclass(frozen=True, slots=True)
class Ref(_Wire):
    """A lexical reference, never a claim of filesystem authenticity."""

    path: str
    sha256: str
    bytes: int

    def __post_init__(self) -> None:
        """Validate direct construction as strictly as decoded wire input."""
        _path(self.path)
        _digest(self.sha256)
        integer(self.bytes)


@dataclass(frozen=True, slots=True)
class Accepted(_Wire):
    """The existing accepted Dataset/Training reference shape."""

    attempt_id: str
    descriptor_sha256: str

    def __post_init__(self) -> None:
        """Validate direct construction as strictly as decoded wire input."""
        _text(self.attempt_id)
        _digest(self.descriptor_sha256)


@dataclass(frozen=True, slots=True)
class Source(_Wire):
    """Committed source identity and caller-authenticated cycle/epoch annotation."""

    publication_id: str
    generation: int
    descriptor_sha256: str
    cycle: int
    incumbent_epoch: int

    def __post_init__(self) -> None:
        """Validate direct construction as strictly as decoded wire input."""
        _text(self.publication_id)
        _digest(self.descriptor_sha256)
        integer(self.generation, 1)
        integer(self.cycle)
        integer(self.incumbent_epoch)


@dataclass(frozen=True, slots=True)
class Debt(_Wire):
    """An oldest-work anchor that coalescing cannot slide."""

    id: str
    anchor_cycle: int

    def __post_init__(self) -> None:
        """Validate direct construction as strictly as decoded wire input."""
        _text(self.id)
        integer(self.anchor_cycle)


@dataclass(frozen=True, slots=True)
class State(_Wire):
    """All six slots, including opaque future promotion references."""

    schema_version: int
    revision: int
    policy_sha256: str
    incumbent_epoch: int
    pending_source: Source | None
    dataset_job: Ref | None
    ready: Ref | None
    cohort: Ref | None
    reserved_promotion: Ref | None
    pending_promotion: Ref | None
    learning_debt: Debt | None
    promotion_debt: Debt | None
    last_consumed_generation: int
    last_accepted_round: int
    admission_open: bool

    def __post_init__(self) -> None:
        """Validate direct construction as strictly as decoded wire input."""
        integer(self.schema_version, 1)
        require(self.schema_version == 1, "Unsupported state schema.")
        _digest(self.policy_sha256)
        for value in (
            self.revision,
            self.incumbent_epoch,
            self.last_consumed_generation,
            self.last_accepted_round,
        ):
            integer(value)
        require(isinstance(self.admission_open, bool), "Invalid admission flag.")
        require(
            self.pending_source is None or isinstance(self.pending_source, Source),
            "Invalid pending source.",
        )
        for ref in (
            self.dataset_job,
            self.ready,
            self.cohort,
            self.reserved_promotion,
            self.pending_promotion,
        ):
            require(ref is None or isinstance(ref, Ref), "Invalid slot reference.")
        for debt in (self.learning_debt, self.promotion_debt):
            require(debt is None or isinstance(debt, Debt), "Invalid debt.")
        require(
            not any((self.pending_source, self.dataset_job, self.ready, self.cohort))
            or self.learning_debt is not None,
            "Learning work has no debt anchor.",
        )
        require(
            not any((self.reserved_promotion, self.pending_promotion))
            or self.promotion_debt is not None,
            "Promotion work has no debt anchor.",
        )
        if self.pending_source is not None:
            require(
                self.pending_source.generation > self.last_consumed_generation,
                "Pending source was already consumed.",
            )
            require(
                self.pending_source.incumbent_epoch <= self.incumbent_epoch,
                "Source epoch is in the future.",
            )

    @classmethod
    def from_metadata(cls, value: object) -> Self:
        """Decode slots without inspecting deferred promotion payloads."""
        data = _fields(value, cls)
        data["pending_source"] = _optional(Source, data["pending_source"])
        for key in (
            "dataset_job",
            "ready",
            "cohort",
            "reserved_promotion",
            "pending_promotion",
        ):
            data[key] = _optional(Ref, data[key])
        for key in ("learning_debt", "promotion_debt"):
            data[key] = _optional(Debt, data[key])
        return cls(**data)


@dataclass(frozen=True, slots=True)
class Job(_Wire):
    """Only Dataset EXTRACT/PREPARE and Training admission are supported here."""

    id: str
    stage: str
    source: Source
    claim: Ref
    managed_launch: Ref
    budget: Ref
    status: str
    accepted_dataset: Accepted | None
    ready: Ref | None
    parents: Ref | None
    cohort_id: str | None
    round_number: int | None
    reserved_id: str | None

    def __post_init__(self) -> None:
        """Validate direct construction as strictly as decoded wire input."""
        _text(self.id)
        require(
            isinstance(self.stage, str)
            and self.stage in {"EXTRACT", "PREPARE", "TRAIN"},
            "Unsupported job stage.",
        )
        require(
            isinstance(self.status, str) and self.status in {"ADMITTED", "SUSPENDED"},
            "Invalid job status.",
        )
        require(isinstance(self.source, Source), "Invalid job source.")
        for ref in (self.claim, self.managed_launch, self.budget):
            require(isinstance(ref, Ref), "Invalid job reference.")
        require(self.reserved_id is None, "Comparison jobs are deferred.")
        require(
            self.accepted_dataset is None
            or isinstance(self.accepted_dataset, Accepted),
            "Invalid accepted Dataset.",
        )
        if self.stage == "EXTRACT":
            require(
                self.accepted_dataset is None, "Extraction already has accepted data."
            )
        else:
            require(self.accepted_dataset is not None, "Missing accepted Dataset.")
        if self.stage == "TRAIN":
            require(
                isinstance(self.ready, Ref) and isinstance(self.parents, Ref),
                "Missing ready/parent binding.",
            )
            _text(self.cohort_id)
            integer(self.round_number, 1)
        else:
            require(
                self.ready is None
                and self.parents is None
                and self.cohort_id is None
                and self.round_number is None,
                "Dataset preparation cannot bind learner parents.",
            )

    @classmethod
    def from_metadata(cls, value: object) -> Self:
        """Decode the reviewed Job wire shape; COMPARE is explicitly unsupported."""
        data = _fields(value, cls)
        data["source"] = Source.from_metadata(data["source"])
        for key in ("claim", "managed_launch", "budget"):
            data[key] = Ref.from_metadata(data[key])
        for key in ("ready", "parents"):
            data[key] = _optional(Ref, data[key])
        data["accepted_dataset"] = _optional(Accepted, data["accepted_dataset"])
        return cls(**data)


_NATIVE_FILES = (
    "flat.manifest.json",
    "flat.pt",
    "relational.manifest.json",
    "relational.pt",
    "split.json",
)


@dataclass(frozen=True, slots=True)
class Ready(_Wire):
    """A complete prepared inventory, with immutable nested storage."""

    accepted_dataset: Accepted
    source: Source
    accepted_rows: Ref
    rows: Ref
    recipe_sha256: str
    native_complete: Ref
    split: Ref
    files: tuple[tuple[str, Ref], ...]
    producer_job_id: str

    def __post_init__(self) -> None:
        """Validate direct construction as strictly as decoded wire input."""
        require(
            isinstance(self.accepted_dataset, Accepted)
            and isinstance(self.source, Source),
            "Invalid ready source/Dataset.",
        )
        for ref in (self.accepted_rows, self.rows, self.native_complete, self.split):
            require(isinstance(ref, Ref), "Invalid ready file reference.")
        _digest(self.recipe_sha256)
        _text(self.producer_job_id)
        require(
            isinstance(self.files, tuple)
            and all(isinstance(pair, tuple) and len(pair) == 2 for pair in self.files),
            "Expected immutable file pairs.",
        )
        require(
            tuple(name for name, _ in self.files) == _NATIVE_FILES,
            "Incomplete/duplicate native inventory.",
        )
        root = f"pipeline/prepared_inputs_v1/{self.accepted_dataset.descriptor_sha256}"
        entry = f"{root}/.rows.jsonl.shared-preparation-v1/entry"
        require(
            self.rows.path == f"{root}/rows.jsonl",
            "Rows are not at the final stable path.",
        )
        require(
            self.accepted_rows.path != self.rows.path,
            "Operational rows must be separate.",
        )
        require(
            (self.rows.sha256, self.rows.bytes)
            == (self.accepted_rows.sha256, self.accepted_rows.bytes),
            "Prepared rows differ from accepted rows.",
        )
        require(
            self.native_complete.path == f"{entry}/complete.json",
            "Invalid native completion path.",
        )
        for name, ref in self.files:
            require(
                isinstance(ref, Ref) and ref.path == f"{entry}/{name}",
                "Native file escaped its entry.",
            )
        require(
            dict(self.files)["split.json"] == self.split,
            "Split differs from inventory.",
        )

    def to_metadata(self) -> dict[str, Any]:
        """Return detached native file mapping, not the internal tuple."""
        return {
            **asdict(self),
            "files": {name: ref.to_metadata() for name, ref in self.files},
        }

    @classmethod
    def from_metadata(cls, value: object) -> Self:
        """Validate every structural binding without opening any file."""
        data = _fields(value, cls)
        data["accepted_dataset"] = Accepted.from_metadata(data["accepted_dataset"])
        data["source"] = Source.from_metadata(data["source"])
        for key in ("accepted_rows", "rows", "native_complete", "split"):
            data[key] = Ref.from_metadata(data[key])
        inventory = data["files"]
        require(
            isinstance(inventory, dict) and set(inventory) == set(_NATIVE_FILES),
            "Invalid native inventory keys.",
        )
        data["files"] = tuple(
            (name, Ref.from_metadata(inventory[name])) for name in _NATIVE_FILES
        )
        return cls(**data)


@dataclass(frozen=True, slots=True)
class JobRecord:
    """Immutable job envelope proposed for future publication."""

    context: Context
    payload: Job
    schema: ClassVar[str] = "continuous_job_v1"

    def __post_init__(self) -> None:
        """Validate direct construction as strictly as decoded wire input."""
        require(
            isinstance(self.context, Context) and isinstance(self.payload, Job),
            "Invalid job envelope.",
        )

    def to_metadata(self) -> dict[str, Any]:
        """Serialize the exact reviewed envelope."""
        return {
            "schema": self.schema,
            "context": self.context.to_metadata(),
            "payload": self.payload.to_metadata(),
        }

    @classmethod
    def from_metadata(cls, value: object) -> Self:
        """Decode only a Dataset/Training job envelope."""
        data = _envelope(value, cls.schema)
        return cls(
            Context.from_metadata(data["context"]), Job.from_metadata(data["payload"])
        )


@dataclass(frozen=True, slots=True)
class ReadyRecord:
    """Immutable prepared-ready envelope; no prediction or lease is serialized."""

    context: Context
    payload: Ready
    schema: ClassVar[str] = "prepared_dataset_ready_v1"

    def __post_init__(self) -> None:
        """Validate direct construction as strictly as decoded wire input."""
        require(
            isinstance(self.context, Context) and isinstance(self.payload, Ready),
            "Invalid ready envelope.",
        )

    def to_metadata(self) -> dict[str, Any]:
        """Serialize the exact reviewed envelope."""
        return {
            "schema": self.schema,
            "context": self.context.to_metadata(),
            "payload": self.payload.to_metadata(),
        }

    @classmethod
    def from_metadata(cls, value: object) -> Self:
        """Decode a complete prepared-ready inventory."""
        data = _envelope(value, cls.schema)
        return cls(
            Context.from_metadata(data["context"]), Ready.from_metadata(data["payload"])
        )


def _envelope(value: object, schema: str) -> dict[str, Any]:
    require(isinstance(value, dict), "Expected an envelope.")
    assert isinstance(value, dict)
    require(
        set(value) == {"schema", "context", "payload"} and value["schema"] == schema,
        "Unsupported envelope.",
    )
    return dict(value)


def encode_record(record: JobRecord | ReadyRecord) -> bytes:
    """Canonical proposed bytes; the caller must fsync/publish before state commit."""
    return json.dumps(
        record.to_metadata(), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def record_ref(record: JobRecord | ReadyRecord) -> Ref:
    """Describe proposed bytes, without asserting an artifact exists."""
    digest = object_digest(record.to_metadata())
    return Ref(
        f"pipeline/continuous_v1/objects/{digest}.json",
        digest,
        len(encode_record(record)),
    )
