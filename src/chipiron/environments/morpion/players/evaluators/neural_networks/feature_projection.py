"""Canonical flat feature indices shared by inference and Training caches."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import os

from .feature_schema import MORPION_CANONICAL_FEATURE_NAMES


class InvalidMorpionFlatTensorCacheError(ValueError):
    """Raised when one flat tensor cache cannot be used safely."""

    @classmethod
    def missing_or_stale(
        cls,
        rows_path: str | os.PathLike[str],
    ) -> InvalidMorpionFlatTensorCacheError:
        """Return the missing or stale cache error."""
        return cls(
            f"Flat tensor cache is missing or stale for rows artifact: {rows_path!r}."
        )

    @classmethod
    def unknown_feature(cls, feature_name: str) -> InvalidMorpionFlatTensorCacheError:
        """Return the unknown feature-name error."""
        return cls(f"Unknown Morpion flat feature name: {feature_name!r}.")

    @classmethod
    def invalid_feature_order(cls) -> InvalidMorpionFlatTensorCacheError:
        """Return the invalid canonical feature-order error."""
        return cls("Flat feature subsets must follow canonical feature order.")

    @classmethod
    def empty_feature_subset(cls) -> InvalidMorpionFlatTensorCacheError:
        """Return the empty feature-subset error."""
        return cls("Flat feature subsets must contain at least one feature.")


def feature_indices_for_subset(
    requested_feature_names: tuple[str, ...],
) -> tuple[int, ...]:
    """Return canonical feature indices for one ordered flat-feature subset."""
    canonical_index = {
        feature_name: index
        for index, feature_name in enumerate(MORPION_CANONICAL_FEATURE_NAMES)
    }
    indices: list[int] = []
    previous_index = -1
    for feature_name in requested_feature_names:
        try:
            index = canonical_index[feature_name]
        except KeyError as exc:
            raise InvalidMorpionFlatTensorCacheError.unknown_feature(
                feature_name
            ) from exc
        if index <= previous_index:
            raise InvalidMorpionFlatTensorCacheError.invalid_feature_order()
        indices.append(index)
        previous_index = index
    if not indices:
        raise InvalidMorpionFlatTensorCacheError.empty_feature_subset()
    return tuple(indices)
