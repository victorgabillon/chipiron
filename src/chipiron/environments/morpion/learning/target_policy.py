"""Chipiron learning policy over values already selected/exported by Anemone."""

import copy
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

from chipiron.environments.morpion.bootstrap.sharded_training_v2_contract import (
    require_historical_source_policy,
)

from .target_confidence import (
    CONFIDENCE_CONTRACT,
    SAFE_FAMILY_POLICIES,
    TARGET_CONTRACT,
    DatasetConfidenceContext,
)

if TYPE_CHECKING:
    from anemone.training_export import TrainingNodeSnapshot

TargetSource = Literal["effective", "tree", "direct"]


def validate_target_policy(
    source: TargetSource | None, use_backed_up_value: bool | None
) -> None:
    """Keep the legacy preference and exact-slot selection mutually exclusive."""
    if source is None:
        if not isinstance(use_backed_up_value, bool):
            raise ValueError("Legacy target selection requires use_backed_up_value.")  # noqa: TRY003
    elif source not in ("effective", "tree", "direct"):
        raise ValueError(f"Unsupported Dataset target source: {source!r}")  # noqa: TRY003
    elif use_backed_up_value is not None:
        raise ValueError(  # noqa: TRY003
            "Explicit target_source requires use_backed_up_value=null; "
            "the historical boolean is not an alias for a target policy."
        )


def select_exported_target(
    node: TrainingNodeSnapshot, source: TargetSource
) -> tuple[float, str] | None:
    """Select one exported slot without fallback or recomputing effective value."""
    if source == "effective":
        if node.effective_value_source not in ("direct_self", "tree_child", "none"):
            raise ValueError(  # noqa: TRY003
                "Canonical effective-value provenance is unavailable; use a scientific export."
            )
        value = node.effective_value_scalar
        if (value is None) != (node.effective_value_source == "none"):
            raise ValueError("Inconsistent exported effective value/source.")  # noqa: TRY003
    elif source == "tree":
        value = node.tree_value_scalar
    elif source == "direct":
        value = node.direct_value_scalar
    else:
        raise ValueError(f"Unsupported Dataset target source: {source!r}")  # noqa: TRY003
    return None if value is None else (float(value), f"{source}_value")


def target_policy_metadata(source: TargetSource | None) -> dict[str, str]:
    """Add policy identity only for new explicit policies; freeze historical JSON."""
    return (
        {}
        if source is None
        else {
            "dataset_target_source": source,
            "dataset_target_contract": TARGET_CONTRACT,
        }
    )


def training_target_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Carry explicit Dataset semantics into the existing Training/bundle metadata."""
    require_historical_source_policy(metadata)
    source = metadata.get("dataset_target_source")
    if source is None:
        return {}
    if (
        source not in ("effective", "tree", "direct")
        or metadata.get("dataset_target_contract") != TARGET_CONTRACT
        or metadata.get("dataset_confidence_policy") != "selected_value_v2"
        or not metadata.get("dataset_extraction_identity")
    ):
        raise ValueError("Incomplete explicit Dataset target provenance.")  # noqa: TRY003
    result = copy.deepcopy({
        "dataset_target_policy": {
            "source": source,
            "contract": TARGET_CONTRACT,
            "family_target_policy": metadata["dataset_family_target_policy"],
            "family_prediction_blend": metadata["dataset_family_prediction_blend"],
        },
        "dataset_extraction_identity": metadata["dataset_extraction_identity"],
        "scientific_source_provenance": metadata["scientific_source_provenance"],
        "dataset_confidence_contract": metadata["dataset_confidence_contract"],
    })
    validate_model_target_provenance(
        model_target_kind(result["dataset_target_policy"]), result
    )
    return result


def model_target_kind(policy: object) -> str:
    """Keep historical bundle labels; name new raw or smoothed targets truthfully."""
    if policy is None:
        return "backup_value"
    if (
        not isinstance(policy, Mapping)
        or policy.get("contract") != TARGET_CONTRACT
        or set(policy)
        != {"source", "contract", "family_target_policy", "family_prediction_blend"}
        or policy.get("source")
        not in (
            "effective",
            "tree",
            "direct",
        )
    ):
        raise ValueError("Unsupported model target policy.")  # noqa: TRY003
    family = policy["family_target_policy"]
    blend = policy["family_prediction_blend"]
    if not isinstance(family, str) or family not in SAFE_FAMILY_POLICIES:
        raise ValueError("Unsupported model family target policy.")  # noqa: TRY003
    if (
        isinstance(blend, bool)
        or not isinstance(blend, (float, int))
        or not math.isfinite(blend)
        or not 0 <= blend <= 1
    ):
        raise ValueError("Invalid model family prediction blend.")  # noqa: TRY003
    return (
        f"{policy['source']}_value"
        if policy["family_target_policy"] == "none"
        else "family_adjusted_value"
    )


def validate_model_target_kind(kind: str, metadata: Mapping[str, Any]) -> None:
    """Accept declared historical labels or a complete, consistent explicit policy."""
    policy = metadata.get("dataset_target_policy")
    if (
        "dataset_target_policy" in metadata and policy is None
    ) or kind != model_target_kind(policy):
        raise ValueError("Model target kind disagrees with its declared target policy.")  # noqa: TRY003


def validate_model_target_provenance(
    kind: str,
    metadata: Mapping[str, Any],
    *,
    declared: bool = True,
    allow_implicit_legacy: bool = False,
) -> None:
    """Validate a trained model's own semantics, independently of comparison targets."""
    require_historical_source_policy(metadata)
    validate_model_target_kind(kind, metadata)
    policy = metadata.get("dataset_target_policy")
    if not declared and not (allow_implicit_legacy and policy is None):
        raise ValueError(  # noqa: TRY003
            "Scientific target comparison/continuation requires a declared model target kind."
        )
    if policy is None:
        # backup_value is the declared historical contract, not an effective alias.
        return
    identity = metadata.get("dataset_extraction_identity")
    provenance = metadata.get("scientific_source_provenance")
    if (
        not isinstance(identity, str)
        or len(identity) != 64
        or any(c not in "0123456789abcdef" for c in identity)
        or not isinstance(provenance, dict)
        or metadata.get("dataset_confidence_contract") != CONFIDENCE_CONTRACT
    ):
        raise ValueError("Incomplete explicit model scientific target provenance.")  # noqa: TRY003
    DatasetConfidenceContext(lambda _node_id: {}, provenance)
    continuation = metadata.get("continuation")
    if continuation and (
        not isinstance(continuation, dict)
        or continuation.get("dataset_target_policy") != policy
    ):
        raise ValueError(  # noqa: TRY003
            "Model continuation target policy disagrees with its provenance."
        )
