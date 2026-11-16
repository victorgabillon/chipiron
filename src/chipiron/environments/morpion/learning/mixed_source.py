"""Dataset interpretation of C3-authenticated current direct prediction origins.

C3 owns write authentication. This module neither qualifies proposed origins nor
changes a scalar: derived targets belong to the sealed search snapshot.
"""
# ruff: noqa: TRY003

import copy
import importlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from typing import TYPE_CHECKING, Any

from chipiron.environments.morpion.bootstrap.sharded_training_v2_contract import (
    CLOSED_KIND,
    LINEAGE_CONTRACT,
    MIXED_SOURCE_CONTRACT,
    require_historical_source_policy,
)

if TYPE_CHECKING:
    from pathlib import Path

    from chipiron.environments.morpion.bootstrap.prediction_lineage import (
        PredictionOrigin,
    )


def require(condition: bool, message: str) -> None:
    """Refuse ambiguous mixed metadata without importing the runtime authority."""
    if not condition:
        raise ValueError(message)


SOURCE_POLICY = "mixed_search_targets_v1"


def _sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def validate_mixed_reference(reference: object) -> None:
    """Require C3's exact capsule-owned reference, never a qualified-origin list."""
    require(
        isinstance(reference, dict)
        and set(reference)
        == {"contract", "path", "sha256", "evidence_path", "evidence_sha256"}
        and reference["contract"] == LINEAGE_CONTRACT
        and reference["path"] == "prediction_lineage.json"
        and reference["evidence_path"] == "prediction_evidence.json"
        and _sha(reference["sha256"])
        and _sha(reference["evidence_sha256"]),
        "B4 source requires its sealed current-origin and write-evidence reference.",
    )


def validate_source_policy(metadata: Mapping[str, Any]) -> bool:
    """Validate serialized identity, not write authenticity; preserve legacy fences."""
    source = metadata.get("scientific_source_provenance", metadata)
    if (
        not isinstance(source, Mapping)
        or source.get("scientific_contract") != MIXED_SOURCE_CONTRACT
    ):
        require_historical_source_policy(metadata)
        return False
    if source is not metadata:
        require_historical_source_policy({
            k: v for k, v in metadata.items() if k != "scientific_source_provenance"
        })
    require(
        source.get("source_policy") == SOURCE_POLICY
        and source.get("source_format_kind") == CLOSED_KIND
        and type(source.get("source_format_version")) is int
        and source["source_format_version"] == 2
        and type(source.get("storage_layout_version")) is int
        and source["storage_layout_version"] == 1
        and type(source.get("source_generation")) is int
        and source["source_generation"] >= 1
        and isinstance(source.get("producer"), dict)
        and isinstance(source.get("capsule_id"), str)
        and len(source["capsule_id"]) == 32
        and all(c in "0123456789abcdef" for c in source["capsule_id"])
        and all(
            _sha(source.get(key))
            for key in (
                "capsule_descriptor_sha256",
                "checkpoint_sha256",
                "generation_descriptor_sha256",
            )
        ),
        "B4 source provenance is not admitted: incomplete mixed scientific identity.",
    )
    validate_mixed_reference(source.get("prediction_lineage"))
    origins = source.get("prediction_origins")
    counts = source.get("direct_origin_counts")
    require(
        isinstance(origins, list)
        and isinstance(counts, list)
        and len(origins) == len(counts),
        "B4 source has no interned direct-origin distribution.",
    )
    assert isinstance(origins, list) and isinstance(counts, list)
    from chipiron.environments.morpion.bootstrap.prediction_lineage import (
        PredictionOrigin,
    )

    require(
        all(
            isinstance(origin, dict)
            and set(origin) == {f.name for f in fields(PredictionOrigin)}
            for origin in origins
        ),
        "B4 source has malformed prediction origins.",
    )
    parsed = tuple(PredictionOrigin(**origin) for origin in origins)
    require(
        len(set(parsed)) == len(parsed)
        and all(type(n) is int and n >= 0 for n in counts),
        "B4 source has contradictory direct origins/counts.",
    )
    require(
        "homogeneous_evaluator_sha256" not in source,
        "B4 mixed search targets do not assert a homogeneous teacher.",
    )
    return True


@dataclass(frozen=True)
class MixedDatasetSource:
    """Interpret an already authenticated C3 result during one pinned extraction."""

    provenance: Mapping[str, Any]
    current: "Mapping[str, PredictionOrigin]"
    origins: "tuple[PredictionOrigin, ...]"
    creation_generations: Mapping[str, int]
    _indices: "dict[PredictionOrigin, int]" = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Index the interned table once for bounded per-row origin lookup."""
        object.__setattr__(
            self,
            "_indices",
            {origin: index for index, origin in enumerate(self.origins)},
        )

    def direct_origin(self, node_id: str, certainty: object) -> int | None:
        """Require one actual write for estimates; certificates/null are not neural."""
        origin = self.current.get(node_id)
        require(
            (origin is not None) == (certainty == "ESTIMATE"),
            "Direct certainty contradicts authenticated write coverage.",
        )
        return None if origin is None else self._indices[origin]


def read_mixed_dataset_source(
    path: Path, header: Mapping[str, Any]
) -> MixedDatasetSource:
    """Use the C3 committed reader under the caller's existing generation read pin.

    A missing C3 implementation fails closed. A digest, supplied origin table or
    uncommitted export cannot replace that reader. No evaluator activation check.
    """
    from chipiron.environments.morpion.bootstrap.generation_publication_io import (
        file_digest,
        locate_capsule,
        publication_records,
        read_object,
    )
    from chipiron.environments.morpion.bootstrap.prediction_lineage import (
        PredictionOrigin,
    )

    validate_mixed_reference(header.get("prediction_lineage"))
    require(
        header.get("source_policy") == SOURCE_POLICY,
        "Unsupported mixed Dataset source policy.",
    )
    location = locate_capsule(path)
    require(
        location is not None and location[2] == 3,
        "Mixed Dataset requires a committed closed capsule.",
    )
    assert location is not None
    work_dir, identifier, _version = location
    state = read_object(work_dir / "run_state.json")
    matches = [
        r for r in publication_records(state["metadata"]) if r["id"] == identifier
    ]
    require(len(matches) == 1, "Mixed Dataset source is not committed.")
    record = matches[0]
    require(
        (work_dir / record["tree_snapshot_path"]).resolve() == path.resolve(),
        "Mixed Dataset selected another scientific manifest.",
    )
    authenticated = _read_c3_origins(work_dir, record)
    reference = header["prediction_lineage"]
    identity = record["scientific_identity"]
    require(
        record.get("prediction_lineage") == reference
        and identity.get("prediction_lineage") == reference
        and identity.get("scientific_contract") == MIXED_SOURCE_CONTRACT
        and identity.get("source_policy") == SOURCE_POLICY
        and identity["generation_descriptor_sha256"] == file_digest(path)
        and identity["checkpoint_sha256"] == authenticated.lineage.checkpoint_sha256
        and identity["source_generation"]
        == header["generation"]
        == record["generation"]
        and identity["producer"] == header["producer"],
        "Mixed scientific source contradicts committed Growth evidence.",
    )
    lineage = authenticated.lineage
    current = authenticated.current_origins
    indices = {origin: index for index, origin in enumerate(lineage.origins)}
    require(
        isinstance(current, Mapping)
        and all(
            isinstance(origin, PredictionOrigin) and origin in indices
            for origin in current.values()
        ),
        "C3 returned inconsistent current origins.",
    )
    counts = [0] * len(lineage.origins)
    for origin in current.values():
        counts[indices[origin]] += 1
    provenance = {
        **copy.deepcopy(dict(identity)),
        "capsule_id": identifier,
        "capsule_descriptor_sha256": record["descriptor_sha256"],
        "prediction_origins": [asdict(origin) for origin in lineage.origins],
        "direct_origin_counts": counts,
    }
    validate_source_policy(provenance)
    return MixedDatasetSource(
        provenance, current, lineage.origins, authenticated.creation_generations
    )


def selected_origin(
    *, slot: str, effective_source: str, certainty: object, direct_origin: int | None
) -> dict[str, Any]:
    """Describe the raw selected slot, without assigning a model to a tree backup."""
    if certainty in {"FORCED", "TERMINAL"}:
        return {"kind": "certified_value"}
    if slot == "direct" or (slot == "effective" and effective_source == "direct_self"):
        require(
            direct_origin is not None, "Direct estimate lacks an authenticated origin."
        )
        return {"kind": "direct_prediction", "origin_index": direct_origin}
    return {"kind": "derived_search_value"}


def validate_tracked_historical_source(path: Path, header: Mapping[str, Any]) -> None:
    """Refuse a homogeneous header over tracked mixed writes; preserve old artifacts.

    Historical sources without C3 tracking remain historical. When actual write
    evidence exists, a canonical header cannot erase its contradictory origins.
    This assertion concerns direct predictions only, never a derived target.
    """
    from chipiron.environments.morpion.bootstrap.generation_publication_io import (
        locate_capsule,
        object_digest,
        publication_records,
        read_object,
    )
    from chipiron.environments.morpion.bootstrap.prediction_lineage import (
        validate_homogeneous_direct_claim,
    )

    location = locate_capsule(path)
    if location is None:
        return
    work_dir, identifier, _version = location
    records = publication_records(read_object(work_dir / "run_state.json")["metadata"])
    matches = [record for record in records if record["id"] == identifier]
    require(len(matches) == 1, "Scientific source is not committed.")
    record = matches[0]
    if "prediction_lineage" not in record:
        return
    origins = _read_c3_origins(work_dir, record)
    used = frozenset(origins.current_origins.values())
    if used:
        first = next(iter(used))
        validate_homogeneous_direct_claim(used, first)
        producer = header["producer"]
        require(
            first.evaluator_sha256 == producer["identity"]["evaluator"]
            and first.producer_sha256 == object_digest(producer)
            and first.config_sha256 == producer["identity"]["config_sha256"],
            "Historical homogeneous source contradicts actual direct write origins.",
        )


def _read_c3_origins(work_dir: Path, record: dict[str, Any]) -> Any:
    """One fail-closed dependency boundary for both source discriminators."""
    try:
        c3 = importlib.import_module(
            "chipiron.environments.morpion.bootstrap.prediction_origin_publication"
        )
    except ModuleNotFoundError as error:
        raise ValueError(
            "B4 mixed admission requires C3 actual Growth write evidence."
        ) from error
    authenticated = c3.read_committed_prediction_origins(work_dir, record)
    require(
        authenticated is not None, "Missing authenticated Growth prediction origins."
    )
    return authenticated
