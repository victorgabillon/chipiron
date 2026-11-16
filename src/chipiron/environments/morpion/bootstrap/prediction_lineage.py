"""B4 direct-write evidence contract; not a writer or scientific admission path.

The existing capsule owns these bytes. C3 must authenticate origins from actual
Growth writes, retained model pins and receipt/disposition evidence before using
this structural validator. Passing a requested evaluator is never that evidence.
C5 must bind the sidecar into scientific admission; all live readers still refuse
the new scientific contract until then. Mixed evaluator origins are intentional;
neither activation nor Growth requires a homogeneous tree. This map records only
current direct origins, not per-node history. Derived tree/effective values belong
to the sealed search snapshot, not to any one of these direct origins.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING, Any, Literal, cast

from .generation_publication_io import object_digest
from .sharded_training_v2_contract import LINEAGE_CONTRACT

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .checkpoint_patch_receipts import CheckpointPatchReceipt, PendingPatch

OriginKind = Literal["genesis", "full_refresh", "growth_birth", "accepted_patch"]
_ORIGIN_KINDS = frozenset({"genesis", "full_refresh", "growth_birth", "accepted_patch"})


class PredictionLineageError(ValueError):
    """Missing, contradictory or unqualified direct-write evidence."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PredictionLineageError(message)


def _digest(value: object) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value),
        "Lineage requires an explicit SHA256 identity",
    )
    return cast("str", value)


def _integer(value: object, *, minimum: int = 0) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
        "Invalid lineage integer",
    )
    return cast("int", value)


def _fields(value: object, names: set[str]) -> dict[str, Any]:
    _require(isinstance(value, dict) and set(value) == names, "Invalid lineage fields")
    return cast("dict[str, Any]", value)


@dataclass(frozen=True, slots=True)
class PredictionOrigin:
    """One interned write event, referring to existing evidence, never a model catalog.

    Producer/config/numeric recipe hashes refer to the exact qualified records.
    Evidence identifies a proven genesis/full refresh, owned birth transaction or
    accepted patch plus its application disposition, not Training acceptance.
    """

    kind: OriginKind
    evaluator_sha256: str
    producer_sha256: str
    config_sha256: str
    numeric_recipe_sha256: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        """Reject missing identities even for in-memory construction."""
        _require(
            isinstance(self.kind, str) and self.kind in _ORIGIN_KINDS,
            "Unsupported prediction origin kind",
        )
        for value in (
            self.evaluator_sha256,
            self.producer_sha256,
            self.config_sha256,
            self.numeric_recipe_sha256,
            self.evidence_sha256,
        ):
            _digest(value)


@dataclass(frozen=True, slots=True)
class PatchDisposition:
    """Bounded application outcome to seal with the existing patch/receipt bytes.

    Applied includes equal-score predictions. Protected and certified-reuse IDs
    are visits, not neural writes; missing IDs cannot acquire a new origin.
    C3 must compare these IDs to the actual Growth/native outcome.
    """

    applied: tuple[str, ...]
    protected: tuple[str, ...]
    missing: tuple[str, ...]
    certified_reuse: tuple[str, ...]

    def validate(self, requested: tuple[str, ...]) -> None:
        """Require an exact, disjoint partition without guessed count reconciliation."""
        groups = (
            requested,
            self.applied,
            self.protected,
            self.missing,
            self.certified_reuse,
        )
        for group in groups:
            _require(
                all(isinstance(node, str) and bool(node) for node in group)
                and len(group) == len(set(group)),
                "Duplicate or invalid disposition node ID",
            )
        actual = self.applied + self.protected + self.missing + self.certified_reuse
        _require(
            len(actual) == len(set(actual)) and set(actual) == set(requested),
            "Patch disposition is not an exact disjoint partition",
        )

    def validate_receipt(
        self,
        pending: PendingPatch,
        receipt: CheckpointPatchReceipt,
        origin: PredictionOrigin,
    ) -> None:
        """Bind dispositions to existing exact patch/receipt identities and counts.

        Neither receipt counts nor this check reconstruct the actual applied IDs.
        C3 must observe those IDs at Growth's write boundary and seal this evidence
        with the patch bytes. Historical receipts are never upgraded in place.
        """
        patch = pending.patch
        self.validate(tuple(row.node_id for row in patch.rows))
        _require(
            receipt.patch_id == patch.patch_id
            and receipt.content_sha256 == _digest(pending.content_sha256)
            and receipt.tree_generation == patch.tree_generation
            and receipt.evaluator_generation == patch.evaluator_generation
            and receipt.evaluator_name == patch.evaluator_name
            and receipt.model_bundle_path == patch.model_bundle_path
            and receipt.rows_requested == len(patch.rows)
            and receipt.rows_applied == len(self.applied) + len(self.certified_reuse),
            "Patch disposition differs from accepted receipt",
        )
        _require(
            receipt.blend_alpha == 1.0, "Blended writes have no B4 origin contract"
        )
        _require(
            origin.kind == "accepted_patch"
            and origin.evaluator_sha256 == receipt.model_sha256
            and origin.evidence_sha256 == self.evidence_sha256(receipt),
            "Prediction origin differs from accepted patch evidence",
        )
        applied = set(self.applied)
        reused = set(self.certified_reuse)
        for row in patch.rows:
            if row.node_id in applied:
                _require(
                    row.metadata.get("source") == "active_model_reevaluation"
                    and row.is_exact is False
                    and row.is_terminal is False
                    and row.backed_up_value is None,
                    "Applied origin requires a built-in direct model prediction",
                )
            elif row.node_id in reused:
                _require(
                    row.metadata.get("source") == "terminal_existing_value"
                    and row.is_exact is True
                    and row.is_terminal is True
                    and row.backed_up_value is None,
                    "Certified reuse requires an existing terminal value",
                )

    def evidence_sha256(self, receipt: CheckpointPatchReceipt) -> str:
        """Bind existing receipt identity and disposition with the capsule JSON digest.

        C3 must seal these exact records and the referenced patch bytes. Computing
        this digest does not establish that this receipt/disposition was committed.
        """
        return object_digest({
            "contract": LINEAGE_CONTRACT,
            "receipt": asdict(receipt),
            "disposition": asdict(self),
        })


@dataclass(frozen=True, slots=True)
class PredictionLineage:
    """Self-contained compact map for one checkpoint, with no scalar duplication.

    Origins are indexed by small integers. An optional proven default covers
    creation generations through its bound; later births use explicit generation
    entries. Sparse overrides describe later applied writes by stable public ID.
    None of these defaults refers to the currently active/requested evaluator.
    """

    checkpoint_sha256: str
    origins: tuple[PredictionOrigin, ...]
    default: tuple[int, int] | None
    births: tuple[tuple[int, int], ...]
    overrides: tuple[tuple[str, int], ...]

    def to_bytes(self) -> bytes:
        """Encode only the new sidecar; never change historical source bytes."""
        return json.dumps(
            {"contract": LINEAGE_CONTRACT, **asdict(self)},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()

    def validate(
        self,
        *,
        checkpoint_sha256: str,
        creation_generations: Mapping[str, int],
        estimate_node_ids: frozenset[str],
        qualified_origins: frozenset[PredictionOrigin],
    ) -> frozenset[PredictionOrigin]:
        """Check coverage against authenticated facts supplied by the existing owner.

        This is structural validation, not proof that writes happened. C3 must
        derive qualified_origins and the exact inventory/ESTIMATE set from owned
        write evidence and the sealed snapshot, not from this sidecar itself.
        Certificates and null direct slots are deliberately outside the map.
        """
        _require(
            _digest(self.checkpoint_sha256) == _digest(checkpoint_sha256),
            "Lineage checkpoint binding differs",
        )
        _require(len(self.origins) == len(set(self.origins)), "Duplicate origin record")
        _require(
            set(self.origins) <= qualified_origins,
            "Unqualified prediction origin: owned write evidence required",
        )
        for node, generation in creation_generations.items():
            _require(isinstance(node, str) and bool(node), "Invalid public node ID")
            _integer(generation, minimum=1)
        _require(
            estimate_node_ids <= creation_generations.keys(),
            "Estimate absent from checkpoint inventory",
        )

        def origin_at(index: int) -> PredictionOrigin:
            _require(_integer(index) < len(self.origins), "Unknown prediction origin")
            return self.origins[index]

        bound = 0
        default_origin = None
        if self.default is not None:
            bound = _integer(self.default[0], minimum=1)
            default_origin = origin_at(self.default[1])
            _require(
                default_origin.kind in {"genesis", "full_refresh"},
                "Default requires proven genesis or full refresh",
            )
            _require(
                bound <= max(creation_generations.values(), default=0),
                "Default exceeds checkpoint inventory",
            )
        births = dict(self.births)
        overrides = dict(self.overrides)
        known_generations = set(creation_generations.values())
        _require(len(births) == len(self.births), "Duplicate birth generation")
        _require(len(overrides) == len(self.overrides), "Duplicate override node ID")
        for generation, index in self.births:
            _require(
                _integer(generation, minimum=1) > bound
                and generation in known_generations,
                "Birth overlaps default or is absent from inventory",
            )
            _require(origin_at(index).kind == "growth_birth", "Invalid birth origin")
        for node, index in self.overrides:
            _require(node in estimate_node_ids, "Override is not a direct ESTIMATE")
            origin_at(index)
        used: set[PredictionOrigin] = set()
        for node in estimate_node_ids:
            generation = creation_generations[node]
            if node in overrides:
                origin = origin_at(overrides[node])
            elif generation in births:
                origin = origin_at(births[generation])
            else:
                _require(
                    default_origin is not None and generation <= bound,
                    "Missing direct ESTIMATE origin",
                )
                assert default_origin is not None
                origin = default_origin
            used.add(origin)
        return frozenset(used)


def validate_homogeneous_direct_claim(
    origins: frozenset[PredictionOrigin], expected: PredictionOrigin
) -> None:
    """Check an optional claim; homogeneity is not required for search or admission.

    Even one model can have different numeric/producer recipes. Evidence events
    may differ, but a homogeneous direct source must share the complete recipe.
    This says nothing about derived tree/effective values or traversal coverage.
    Mixed origins pass PredictionLineage.validate; only a false homogeneous
    assertion fails here. No runtime activation or Dataset path calls this helper.
    """
    _require(
        bool(origins), "No direct origins establish a homogeneous prediction claim"
    )
    for origin in origins:
        _require(
            (
                origin.evaluator_sha256,
                origin.producer_sha256,
                origin.config_sha256,
                origin.numeric_recipe_sha256,
            )
            == (
                expected.evaluator_sha256,
                expected.producer_sha256,
                expected.config_sha256,
                expected.numeric_recipe_sha256,
            ),
            "Homogeneous claim contradicts actual direct origins",
        )


def read_prediction_lineage(
    data: bytes, *, sidecar_sha256: str, capsule_file_sha256: str
) -> PredictionLineage:
    """Read exact bytes bound by both scientific identity and capsule inventory.

    No file discovery or new authority is introduced. The caller supplies the
    two hashes from the existing authenticated descriptor; digest integrity alone
    does not qualify origins or admit a scientific source.
    """
    _require(
        hashlib.sha256(data).hexdigest()
        == _digest(sidecar_sha256)
        == _digest(capsule_file_sha256),
        "Lineage sidecar binding differs",
    )
    payload = _fields(
        json.loads(data, object_pairs_hook=_unique_object),
        {"contract", "checkpoint_sha256", "origins", "default", "births", "overrides"},
    )
    _require(payload["contract"] == LINEAGE_CONTRACT, "Unsupported lineage contract")
    _require(isinstance(payload["origins"], list), "Invalid origin table")
    origins = []
    for entry in payload["origins"]:
        origin_fields = _fields(
            entry, {field.name for field in fields(PredictionOrigin)}
        )
        origins.append(PredictionOrigin(**origin_fields))
    default = payload["default"]
    if default is not None:
        _require(isinstance(default, list) and len(default) == 2, "Invalid default")
        default = (_integer(default[0], minimum=1), _integer(default[1]))
    births = tuple(
        (_integer(key, minimum=1), _integer(index))
        for key, index in _pairs(payload["births"])
    )
    overrides = []
    for key, index in _pairs(payload["overrides"]):
        _require(isinstance(key, str) and bool(key), "Invalid override ID")
        overrides.append((cast("str", key), _integer(index)))
    return PredictionLineage(
        _digest(payload["checkpoint_sha256"]),
        tuple(origins),
        default,
        births,
        tuple(overrides),
    )


def _pairs(value: object) -> list[list[Any]]:
    _require(
        isinstance(value, list)
        and all(isinstance(pair, list) and len(pair) == 2 for pair in value),
        "Invalid lineage map entries",
    )
    return cast("list[list[Any]]", value)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = dict(pairs)
    _require(len(result) == len(pairs), "Duplicate lineage JSON key")
    return result
