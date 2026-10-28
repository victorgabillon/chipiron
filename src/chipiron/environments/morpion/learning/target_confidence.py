"""Explicit selected-value confidence for scientific Dataset v2.

PV membership is a heuristic grouping, never a certificate for another state.
The scientific export's effective-node flag is retained independently.
"""
# ruff: noqa: TRY003

import copy
import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from anemone.training_export import TrainingNodeSnapshot

DatasetConfidencePolicy = Literal["legacy_v1", "selected_value_v2"]
CONFIDENCE_CONTRACT = "morpion-selected-value-confidence-1"
SCIENTIFIC_CONTRACT = "anemone-values-canonical-morpion-1"
SAFE_FAMILY_POLICIES = frozenset({
    "none",
    "pv_mean_prediction",
    "pv_min_prediction",
    "pv_blend_mean_prediction",
    "pv_blend_min_prediction",
})


class DatasetConfidenceError(ValueError):
    """Reject ambiguous contracts, unsupported policies and contradictory proofs."""

    @classmethod
    def missing_source(cls) -> DatasetConfidenceError:
        """Describe a source that cannot associate scalars with certificates."""
        return cls(
            "Dataset v2 requires explicit per-value contracts and source provenance"
        )

    @classmethod
    def invalid_row_flags(cls) -> DatasetConfidenceError:
        """Describe nonboolean serialized flags."""
        return cls("Dataset v2 row flags must be explicit booleans")

    @classmethod
    def invalid_envelope(cls) -> DatasetConfidenceError:
        """Describe incompatible row and envelope contracts."""
        return cls("Dataset JSONL version disagrees with confidence contract")


def is_certified(certainty: object) -> bool:
    """Preserve Anemone's FORCED and TERMINAL exactness semantics."""
    return certainty in ("FORCED", "TERMINAL")


def _finite_number(value: object) -> bool:
    """Accept finite JSON numbers, excluding booleans."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def validate_confidence_policy(policy: str, family_policy: str) -> None:
    """Gate all cross-state proof policies before extraction or publication."""
    if policy not in ("legacy_v1", "selected_value_v2"):
        raise DatasetConfidenceError(
            f"Unsupported Dataset confidence policy: {policy!r}"
        )
    if policy == "selected_value_v2" and family_policy not in SAFE_FAMILY_POLICIES:
        raise DatasetConfidenceError(
            "Dataset v2 gates pv_exact_then_* pending an explicit cross-state proof contract"
        )


@dataclass(frozen=True)
class DatasetConfidenceContext:
    """Load one node's independent value contract; retain no population cache."""

    load_contract: Callable[[str], Mapping[str, Any]]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Require an explicit scientific v2 identity, including its producer."""
        generation = self.provenance.get("source_generation")
        if (
            not isinstance(generation, int)
            or isinstance(generation, bool)
            or generation < 1
        ):
            raise DatasetConfidenceError(
                "Dataset v2 requires an explicit source generation"
            )
        if (
            not (
                self.provenance.get("source_format_kind")
                == "morpion_scientific_training_export"
                or (
                    self.provenance.get("source_format_kind")
                    == "morpion_closed_scientific_snapshot"
                    and isinstance(self.provenance.get("storage_layout_version"), int)
                    and not isinstance(
                        self.provenance.get("storage_layout_version"), bool
                    )
                    and self.provenance["storage_layout_version"] == 1
                )
            )
            or self.provenance.get("source_format_version") != 2
            or self.provenance.get("scientific_contract") != SCIENTIFIC_CONTRACT
            or not isinstance(self.provenance.get("producer"), dict)
        ):
            raise DatasetConfidenceError(
                "Dataset v2 requires scientific v2 source provenance"
            )

    def metadata(self) -> dict[str, Any]:
        """Return version and full source identity for artifact/cache consumers."""
        return {
            "dataset_version": 2,
            "dataset_confidence_policy": "selected_value_v2",
            "dataset_confidence_contract": CONFIDENCE_CONTRACT,
            "scientific_source_provenance": copy.deepcopy(dict(self.provenance)),
            "family_proof_contract": None,
        }

    def require_generation(self, generation: int) -> None:
        """Keep the Dataset generation associated with its scientific source."""
        if generation != self.provenance["source_generation"]:
            raise DatasetConfidenceError(
                "Dataset generation disagrees with scientific source"
            )

    def selected(
        self, node: TrainingNodeSnapshot, value: float, source: str
    ) -> dict[str, Any]:
        """Associate the raw scalar with its actual source, never node flags."""
        contract = self.load_contract(node.node_id)
        values, certainties = _validated_value_contract(contract)
        slot = "tree" if source == "backed_up_value" else "direct"
        payload = values.get(slot)
        if payload is not None and payload["score"] != value:
            raise DatasetConfidenceError(
                "Selected scalar disagrees with its scientific source"
            )
        certainty = certainties.get(slot) if payload is not None else None
        return {
            "contract": CONFIDENCE_CONTRACT,
            "node_is_exact": node.is_exact,
            "node_is_terminal": node.is_terminal,
            "node_effective_certainty": certainties.get("effective"),
            "raw_target_value": value,
            "raw_target_source": f"{slot}_value",
            "raw_target_certainty": certainty,
            "training_target_value": value,
            "training_target_source": "raw_target",
            "training_target_certainty": certainty,
            "source_value_contract": copy.deepcopy(dict(contract)),
        }


def _validated_value_contract(
    contract: object,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Check same-state certificates independently of family membership."""
    if not isinstance(contract, dict):
        raise DatasetConfidenceError("Missing per-value scientific contract")
    values = contract.get("values")
    certainties = contract.get("certainties")
    if not isinstance(values, dict) or not isinstance(certainties, dict):
        raise DatasetConfidenceError("Missing per-value scientific contract")
    certified_scores = set()
    for name in ("direct", "tree", "effective", "target"):
        payload = values.get(name)
        certainty = certainties.get(name)
        if certainty not in (None, "ESTIMATE", "FORCED", "TERMINAL"):
            raise DatasetConfidenceError("Unknown scientific value certainty")
        if payload is not None:
            if (
                not isinstance(payload, dict)
                or payload.get("certainty") != certainty
                or not isinstance(payload.get("score"), (int, float))
                or isinstance(payload.get("score"), bool)
                or not math.isfinite(payload["score"])
            ):
                raise DatasetConfidenceError("Inconsistent scientific value payload")
            if is_certified(certainty):
                certified_scores.add(payload["score"])
    # Certificates concern this node and the pinned single-agent objective.
    # Different nodes in a family are deliberately never compared here.
    if len(certified_scores) > 1:
        raise DatasetConfidenceError(
            "Contradictory certificates for the same state/objective"
        )
    return values, certainties


def adjusted_confidence(
    confidence: Mapping[str, Any], *, value: float, rule: str
) -> dict[str, Any]:
    """Keep same-state anchors and demote every heuristic derivation to ESTIMATE."""
    result = copy.deepcopy(dict(confidence))
    raw_rule = rule in ("raw_backup", "hard_exact_anchor")
    if is_certified(result["raw_target_certainty"]) and (
        not raw_rule or value != result["raw_target_value"]
    ):
        raise DatasetConfidenceError(
            "A family prediction cannot overwrite a certified same-state target"
        )
    result["training_target_value"] = value
    result["training_target_source"] = "raw_target" if raw_rule else rule
    result["training_target_certainty"] = (
        result["raw_target_certainty"] if raw_rule else "ESTIMATE"
    )
    return result


def validate_confident_row(
    confidence: object, *, value: float, is_exact: bool, is_terminal: bool
) -> None:
    """Reject malformed serialized v2 confidence without interpreting v1 flags."""
    if (
        not isinstance(confidence, dict)
        or confidence.get("contract") != CONFIDENCE_CONTRACT
    ):
        raise DatasetConfidenceError(
            "Dataset v2 row requires its explicit confidence contract"
        )
    certainty = confidence.get("training_target_certainty")
    required = {
        "raw_target_certainty",
        "raw_target_source",
        "training_target_source",
        "source_value_contract",
        "raw_target_value",
        "node_is_exact",
        "node_effective_certainty",
    }
    if not required <= confidence.keys():
        raise DatasetConfidenceError("Missing Dataset v2 row confidence fields")
    if (
        certainty not in (None, "ESTIMATE", "FORCED", "TERMINAL")
        or confidence.get("training_target_value") != value
        or is_exact != is_certified(certainty)
        or confidence.get("node_is_terminal") is not is_terminal
    ):
        raise DatasetConfidenceError("Inconsistent Dataset v2 row confidence")
    values, certainties = _validated_value_contract(confidence["source_value_contract"])
    raw_source = confidence["raw_target_source"]
    if raw_source not in ("direct_value", "tree_value"):
        raise DatasetConfidenceError("Unsupported selected target source")
    slot = "direct" if raw_source == "direct_value" else "tree"
    payload = values.get(slot)
    raw_certainty = certainties.get(slot) if payload is not None else None
    raw_value = confidence.get("raw_target_value")
    if not _finite_number(raw_value) or not _finite_number(value):
        raise DatasetConfidenceError("Invalid Dataset v2 target scalar")
    if (
        confidence["raw_target_certainty"] != raw_certainty
        or (payload is not None and payload["score"] != raw_value)
        or not isinstance(confidence.get("node_is_exact"), bool)
        or confidence.get("node_effective_certainty") != certainties.get("effective")
    ):
        raise DatasetConfidenceError("Inconsistent selected-source confidence")
    rule = confidence["training_target_source"]
    if rule == "raw_target":
        if value != raw_value or certainty != raw_certainty:
            raise DatasetConfidenceError(
                "Raw target certainty/value changed without provenance"
            )
    elif (
        rule not in SAFE_FAMILY_POLICIES - {"none"}
        or certainty != "ESTIMATE"
        or is_certified(raw_certainty)
    ):
        raise DatasetConfidenceError(
            "Unjustified family proof or certified-target overwrite"
        )


def dataset_format_version(metadata: Mapping[str, Any]) -> int:
    """Identify new confidence artifacts explicitly, with frozen v1 as default."""
    if metadata.get("dataset_version", 1) == 1:
        return 1
    if (
        isinstance(metadata.get("dataset_version"), int)
        and metadata["dataset_version"] == 2
        and metadata.get("dataset_confidence_contract") == CONFIDENCE_CONTRACT
        and metadata.get("dataset_confidence_policy") == "selected_value_v2"
    ):
        return 2
    raise DatasetConfidenceError("Unsupported Dataset version/confidence contract")


def finalize_confidence_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Bind the confidence, source and complete extraction settings into identity."""
    if dataset_format_version(metadata) == 2:
        metadata.pop("dataset_extraction_identity", None)
        metadata["dataset_extraction_identity"] = hashlib.sha256(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    return metadata
