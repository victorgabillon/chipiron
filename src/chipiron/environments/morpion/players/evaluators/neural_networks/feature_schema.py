"""Canonical Morpion handcrafted feature schema and subset helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

MORPION_FEATURE_SCHEMA: Final[str] = "morpion_handcrafted_v1"
DEFAULT_MORPION_FEATURE_SUBSET_NAME: Final[str] = "handcrafted_41"

MORPION_CANONICAL_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "moves",
    "num_points",
    "num_used_unit_segments",
    "bbox_width",
    "bbox_height",
    "bbox_area",
    "point_density_in_bbox",
    "legal_action_count",
    "legal_actions_dir_0",
    "legal_actions_dir_1",
    "legal_actions_dir_2",
    "legal_actions_dir_3",
    "num_distinct_playable_cells",
    "mean_legal_actions_per_playable_cell",
    "max_legal_actions_per_playable_cell",
    "playable_cells_with_1_action",
    "playable_cells_with_2_actions",
    "playable_cells_with_ge_3_actions",
    "dir_usage_value_0_count",
    "dir_usage_value_1_count",
    "dir_usage_value_2_count",
    "dir_usage_value_3_count",
    "points_with_any_dir_usage_3",
    "points_with_ge_2_nonzero_dir_usages",
    "segments_4_present_1_missing_geometric",
    "segments_4_present_1_missing_overlap_ok",
    "segments_4_present_1_missing_parallel_ok",
    "segments_4_present_1_missing_legal",
    "segments_3_present_2_missing_geometric",
    "segments_3_present_2_missing_overlap_ok",
    "segments_3_present_2_missing_parallel_ok",
    "segments_3_present_2_missing_alive",
    "segments_4p1m_dir_0_legal",
    "segments_4p1m_dir_1_legal",
    "segments_4p1m_dir_2_legal",
    "segments_4p1m_dir_3_legal",
    "frontier_cell_count",
    "frontier_cells_in_any_ge3_candidate_segment",
    "frontier_cells_in_any_legal_4p1m_segment",
    "occupied_connected_components",
    "largest_occupied_component_size",
)

_FEATURE_SUBSET_REGISTRY: Final[dict[str, tuple[str, ...]]] = {
    DEFAULT_MORPION_FEATURE_SUBSET_NAME: MORPION_CANONICAL_FEATURE_NAMES,
    "full": MORPION_CANONICAL_FEATURE_NAMES,
    "full_41": MORPION_CANONICAL_FEATURE_NAMES,
}


class InvalidMorpionFeatureSubsetError(ValueError):
    """Raised when one Morpion feature subset is invalid."""


class InconsistentMorpionFeatureSubsetDefinitionError(
    InvalidMorpionFeatureSubsetError
):
    """Raised when a known subset name is paired with conflicting feature names."""

    def __init__(
        self,
        subset_name: str,
        expected_feature_names: tuple[str, ...],
        provided_feature_names: tuple[str, ...],
    ) -> None:
        """Initialize the error for one conflicting known subset definition."""
        super().__init__(
            "Known Morpion feature subset names must match their registered "
            f"feature list exactly; subset {subset_name!r} expects "
            f"{expected_feature_names!r} but got {provided_feature_names!r}."
        )


class UnknownMorpionFeatureSubsetError(ValueError):
    """Raised when one named Morpion feature subset is unknown."""

    def __init__(self, subset_name: str) -> None:
        """Initialize the error with the unknown subset name."""
        super().__init__(f"Unknown Morpion feature subset: {subset_name!r}.")


@dataclass(frozen=True, slots=True)
class MorpionFeatureSchema:
    """Canonical ordered Morpion handcrafted feature schema."""

    name: str
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate the schema eagerly."""
        _validate_ordered_feature_names(self.feature_names, context="schema")

    @property
    def dimension(self) -> int:
        """Return the full schema width."""
        return len(self.feature_names)


@dataclass(frozen=True, slots=True)
class MorpionFeatureSubset:
    """Named ordered Morpion handcrafted feature subset."""

    name: str
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate the subset eagerly."""
        _validate_ordered_feature_names(self.feature_names, context="subset")

    @property
    def dimension(self) -> int:
        """Return the subset width."""
        return len(self.feature_names)


def full_morpion_feature_schema() -> MorpionFeatureSchema:
    """Return the canonical Morpion handcrafted feature schema."""
    return MorpionFeatureSchema(
        name=MORPION_FEATURE_SCHEMA,
        feature_names=MORPION_CANONICAL_FEATURE_NAMES,
    )


def full_morpion_feature_subset() -> MorpionFeatureSubset:
    """Return the canonical full handcrafted Morpion feature subset."""
    return MorpionFeatureSubset(
        name=DEFAULT_MORPION_FEATURE_SUBSET_NAME,
        feature_names=MORPION_CANONICAL_FEATURE_NAMES,
    )


def morpion_feature_subset_from_name(name: str) -> MorpionFeatureSubset:
    """Resolve one named Morpion feature subset from the registry."""
    try:
        feature_names = _FEATURE_SUBSET_REGISTRY[name]
    except KeyError as exc:
        raise UnknownMorpionFeatureSubsetError(name) from exc
    canonical_name = (
        DEFAULT_MORPION_FEATURE_SUBSET_NAME
        if feature_names == MORPION_CANONICAL_FEATURE_NAMES
        else name
    )
    return MorpionFeatureSubset(name=canonical_name, feature_names=feature_names)


def morpion_feature_subset_from_feature_names(
    name: str,
    feature_names: Sequence[str],
) -> MorpionFeatureSubset:
    """Build one validated Morpion feature subset from explicit names."""
    subset = MorpionFeatureSubset(
        name=name,
        feature_names=tuple(str(item) for item in feature_names),
    )
    validate_morpion_feature_subset(subset)
    return subset


def resolve_morpion_feature_subset(
    *,
    feature_subset_name: str | None = None,
    feature_names: Sequence[str] | None = None,
) -> MorpionFeatureSubset:
    """Resolve one Morpion feature subset from name and-or explicit names."""
    if feature_names is not None:
        provided_feature_names = tuple(str(item) for item in feature_names)
        subset_name = (
            DEFAULT_MORPION_FEATURE_SUBSET_NAME
            if feature_subset_name is None
            else feature_subset_name
        )
        registered_feature_names = _FEATURE_SUBSET_REGISTRY.get(subset_name)
        if (
            registered_feature_names is not None
            and tuple(registered_feature_names) != provided_feature_names
        ):
            raise InconsistentMorpionFeatureSubsetDefinitionError(
                subset_name,
                tuple(registered_feature_names),
                provided_feature_names,
            )
        return morpion_feature_subset_from_feature_names(
            subset_name,
            provided_feature_names,
        )
    if feature_subset_name is None:
        return full_morpion_feature_subset()
    return morpion_feature_subset_from_name(feature_subset_name)


def validate_morpion_feature_subset(
    subset: MorpionFeatureSubset,
    *,
    schema: MorpionFeatureSchema | None = None,
) -> None:
    """Validate that ``subset`` is an ordered subset of the canonical schema."""
    resolved_schema = full_morpion_feature_schema() if schema is None else schema
    index_by_name = {
        feature_name: index
        for index, feature_name in enumerate(resolved_schema.feature_names)
    }
    missing_names = [
        feature_name
        for feature_name in subset.feature_names
        if feature_name not in index_by_name
    ]
    if missing_names:
        raise InvalidMorpionFeatureSubsetError(
            "Morpion feature subset contains unknown feature names: "
            + ", ".join(missing_names)
        )
    indices = tuple(index_by_name[feature_name] for feature_name in subset.feature_names)
    if tuple(sorted(indices)) != indices:
        raise InvalidMorpionFeatureSubsetError(
            "Morpion feature subset order must follow the canonical handcrafted feature order."
        )


def subset_indices(
    subset: MorpionFeatureSubset,
    *,
    schema: MorpionFeatureSchema | None = None,
) -> tuple[int, ...]:
    """Return the canonical full-schema indices for one subset."""
    validate_morpion_feature_subset(subset, schema=schema)
    resolved_schema = full_morpion_feature_schema() if schema is None else schema
    index_by_name = {
        feature_name: index
        for index, feature_name in enumerate(resolved_schema.feature_names)
    }
    return tuple(index_by_name[feature_name] for feature_name in subset.feature_names)


def _validate_ordered_feature_names(
    feature_names: Sequence[str],
    *,
    context: str,
) -> None:
    """Validate one ordered sequence of Morpion feature names."""
    if not feature_names:
        raise InvalidMorpionFeatureSubsetError(
            f"Morpion {context} must contain at least one feature name."
        )
    seen: set[str] = set()
    duplicates: list[str] = []
    for feature_name in feature_names:
        if not isinstance(feature_name, str):
            raise InvalidMorpionFeatureSubsetError(
                f"Morpion {context} feature names must all be strings."
            )
        if feature_name in seen:
            duplicates.append(feature_name)
        seen.add(feature_name)
    if duplicates:
        raise InvalidMorpionFeatureSubsetError(
            "Morpion feature names must be unique; duplicate names: "
            + ", ".join(sorted(set(duplicates)))
        )


__all__ = [
    "DEFAULT_MORPION_FEATURE_SUBSET_NAME",
    "InconsistentMorpionFeatureSubsetDefinitionError",
    "InvalidMorpionFeatureSubsetError",
    "MORPION_CANONICAL_FEATURE_NAMES",
    "MORPION_FEATURE_SCHEMA",
    "MorpionFeatureSchema",
    "MorpionFeatureSubset",
    "UnknownMorpionFeatureSubsetError",
    "full_morpion_feature_schema",
    "full_morpion_feature_subset",
    "morpion_feature_subset_from_feature_names",
    "morpion_feature_subset_from_name",
    "resolve_morpion_feature_subset",
    "subset_indices",
    "validate_morpion_feature_subset",
]