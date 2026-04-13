"""Tests for explicit Morpion handcrafted feature subsets."""

from __future__ import annotations

import pytest

from chipiron.environments.morpion.players.evaluators.neural_networks import (
    DEFAULT_MORPION_FEATURE_SUBSET_NAME,
    InconsistentMorpionFeatureSubsetDefinitionError,
    InvalidMorpionFeatureSubsetError,
    MORPION_CANONICAL_FEATURE_NAMES,
    UnknownMorpionFeatureSubsetError,
    full_morpion_feature_subset,
    morpion_feature_subset_from_feature_names,
    morpion_feature_subset_from_name,
    resolve_morpion_feature_subset,
    subset_indices,
)


def test_full_subset_resolution_is_stable() -> None:
    """The canonical full subset should resolve with the expected name and order."""
    full_subset = full_morpion_feature_subset()

    assert full_subset.name == DEFAULT_MORPION_FEATURE_SUBSET_NAME
    assert full_subset.feature_names == MORPION_CANONICAL_FEATURE_NAMES
    assert subset_indices(full_subset) == tuple(range(len(MORPION_CANONICAL_FEATURE_NAMES)))


def test_named_full_subset_aliases_normalize_to_canonical_full_subset() -> None:
    """Full-width aliases should normalize to the same canonical subset."""
    assert morpion_feature_subset_from_name("full") == full_morpion_feature_subset()
    assert morpion_feature_subset_from_name("full_41") == full_morpion_feature_subset()


def test_unknown_subset_name_fails_clearly() -> None:
    """Unknown subset names should raise a dedicated error."""
    with pytest.raises(UnknownMorpionFeatureSubsetError):
        morpion_feature_subset_from_name("handcrafted_20")


def test_explicit_subset_indices_preserve_canonical_order() -> None:
    """Explicit subsets should keep the canonical feature ordering exactly."""
    subset = morpion_feature_subset_from_feature_names(
        "handcrafted_3_custom",
        (
            MORPION_CANONICAL_FEATURE_NAMES[0],
            MORPION_CANONICAL_FEATURE_NAMES[3],
            MORPION_CANONICAL_FEATURE_NAMES[7],
        ),
    )

    assert subset.feature_names == (
        MORPION_CANONICAL_FEATURE_NAMES[0],
        MORPION_CANONICAL_FEATURE_NAMES[3],
        MORPION_CANONICAL_FEATURE_NAMES[7],
    )
    assert subset_indices(subset) == (0, 3, 7)


def test_duplicate_feature_names_are_rejected() -> None:
    """Explicit subsets should fail when they repeat one feature name."""
    with pytest.raises(InvalidMorpionFeatureSubsetError):
        morpion_feature_subset_from_feature_names(
            "handcrafted_duplicate",
            (
                MORPION_CANONICAL_FEATURE_NAMES[0],
                MORPION_CANONICAL_FEATURE_NAMES[0],
            ),
        )


def test_unknown_or_out_of_order_feature_names_are_rejected() -> None:
    """Subset validation should reject unknown names and canonical-order violations."""
    with pytest.raises(InvalidMorpionFeatureSubsetError):
        morpion_feature_subset_from_feature_names(
            "handcrafted_unknown",
            (MORPION_CANONICAL_FEATURE_NAMES[0], "missing_feature"),
        )

    with pytest.raises(InvalidMorpionFeatureSubsetError):
        morpion_feature_subset_from_feature_names(
            "handcrafted_out_of_order",
            (
                MORPION_CANONICAL_FEATURE_NAMES[3],
                MORPION_CANONICAL_FEATURE_NAMES[1],
            ),
        )


def test_known_subset_name_with_conflicting_explicit_feature_names_fails() -> None:
    """Known subset names should not be allowed to mean different feature lists."""
    with pytest.raises(InconsistentMorpionFeatureSubsetDefinitionError):
        resolve_morpion_feature_subset(
            feature_subset_name="full",
            feature_names=MORPION_CANONICAL_FEATURE_NAMES[:10],
        )

    with pytest.raises(InconsistentMorpionFeatureSubsetDefinitionError):
        resolve_morpion_feature_subset(
            feature_subset_name=DEFAULT_MORPION_FEATURE_SUBSET_NAME,
            feature_names=MORPION_CANONICAL_FEATURE_NAMES[:10],
        )