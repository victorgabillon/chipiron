"""Evaluator configuration types for the Morpion bootstrap workflow."""

from __future__ import annotations

from dataclasses import dataclass, field

from chipiron.environments.morpion.players.evaluators.neural_networks.feature_schema import (
    DEFAULT_MORPION_FEATURE_SUBSET_NAME,
    MorpionFeatureSubset,
    resolve_morpion_feature_subset,
)

from .bootstrap_errors import (
    EmptyMorpionEvaluatorsConfigError,
    InconsistentMorpionEvaluatorSpecNameError,
)


def _empty_evaluator_specs() -> dict[str, MorpionEvaluatorSpec]:
    """Return a typed empty evaluator-spec mapping."""
    return {}


@dataclass(frozen=True, slots=True)
class MorpionEvaluatorSpec:
    """Training spec for one named Morpion evaluator."""

    name: str
    model_type: str
    hidden_sizes: tuple[int, ...] | None
    num_epochs: int
    batch_size: int
    learning_rate: float
    feature_subset_name: str = DEFAULT_MORPION_FEATURE_SUBSET_NAME
    feature_names: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Normalize feature subset metadata into a canonical explicit form."""
        subset = resolve_morpion_feature_subset(
            feature_subset_name=self.feature_subset_name,
            feature_names=None if not self.feature_names else self.feature_names,
        )
        object.__setattr__(self, "feature_subset_name", subset.name)
        object.__setattr__(self, "feature_names", subset.feature_names)

    @property
    def feature_subset(self) -> MorpionFeatureSubset:
        """Return the resolved Morpion feature subset for this evaluator."""
        return MorpionFeatureSubset(
            name=self.feature_subset_name,
            feature_names=self.feature_names,
        )


@dataclass(frozen=True, slots=True)
class MorpionEvaluatorsConfig:
    """Deterministic collection of evaluator specs for one bootstrap run."""

    evaluators: dict[str, MorpionEvaluatorSpec] = field(
        default_factory=_empty_evaluator_specs
    )

    def __post_init__(self) -> None:
        """Copy and validate the evaluator mapping eagerly."""
        copied: dict[str, MorpionEvaluatorSpec] = dict(self.evaluators)
        if not copied:
            raise EmptyMorpionEvaluatorsConfigError
        for key, spec in copied.items():
            if key != spec.name:
                raise InconsistentMorpionEvaluatorSpecNameError(key, spec.name)
        object.__setattr__(self, "evaluators", copied)


__all__ = ["MorpionEvaluatorSpec", "MorpionEvaluatorsConfig"]
