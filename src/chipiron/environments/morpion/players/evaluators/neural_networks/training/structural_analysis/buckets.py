"""Deterministic structural buckets and regression metrics."""

from __future__ import annotations

import bisect
import math
import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from chipiron.learning.supervised import regression_quality_stats

from .features import structural_numeric_value

if TYPE_CHECKING:
    from .rows import EvaluatorRoles


@dataclass(frozen=True, slots=True)
class BucketFamilySpec:
    """One semantic bucket family and deterministic quantile count."""

    name: str
    value_path: str
    quantile_count: int


BUCKET_FAMILY_SPECS = (
    BucketFamilySpec("target_deciles", "target", 10),
    BucketFamilySpec("state_move_count", "state.moves", 4),
    BucketFamilySpec("legal_action_count", "state.legal_action_count", 4),
    BucketFamilySpec("num_points", "state.num_points", 4),
    BucketFamilySpec("used_segment_count", "state.used_unit_segment_count", 4),
    BucketFamilySpec("move_token_count", "tokens.moves", 4),
    BucketFamilySpec("relation_count", "relations.total", 4),
    BucketFamilySpec("moves_per_new_dot_max", "geometry.moves_per_new_dot_max", 4),
    BucketFamilySpec(
        "shared_new_dot_pair_count", "geometry.move_pairs_sharing_new_dot", 4
    ),
    BucketFamilySpec(
        "shared_prospective_segment_pair_count",
        "geometry.move_pairs_sharing_prospective_segment",
        4,
    ),
)


def build_error_buckets(
    *,
    rows: tuple[dict[str, object], ...],
    evaluator_names: tuple[str, ...],
    roles: EvaluatorRoles,
) -> dict[str, object]:
    """Build deterministic quantile buckets for every required feature family."""
    families: dict[str, object] = {}
    for spec in BUCKET_FAMILY_SPECS:
        values = tuple(structural_numeric_value(row, spec.value_path) for row in rows)
        boundaries = deterministic_quantile_boundaries(values, spec.quantile_count)
        positions_by_bucket: list[list[int]] = [[] for _ in range(len(boundaries) + 1)]
        for position, value in enumerate(values):
            positions_by_bucket[bisect.bisect_left(boundaries, value)].append(position)
        families[spec.name] = {
            "value_path": spec.value_path,
            "boundary_generation_policy": (
                f"nearest_rank_{spec.quantile_count}_quantiles_deduplicated"
            ),
            "boundaries": list(boundaries),
            "buckets": [
                _bucket_payload(
                    rows=rows,
                    positions=positions,
                    evaluator_names=evaluator_names,
                    roles=roles,
                    bucket_index=bucket_index,
                    boundaries=boundaries,
                )
                for bucket_index, positions in enumerate(positions_by_bucket)
            ],
        }
    return {
        "schema": "morpion_structural_error_buckets_v1",
        "boundary_policy": (
            "Deterministic nearest-rank quantiles over selected validation rows; "
            "duplicate boundaries are removed and boundary values belong to the "
            "lower bucket."
        ),
        "families": families,
    }


def deterministic_quantile_boundaries(
    values: tuple[float, ...],
    quantile_count: int,
) -> tuple[float, ...]:
    """Return deduplicated nearest-rank internal quantile boundaries."""
    if not values or quantile_count < 2:
        return ()
    ordered = sorted(values)
    boundaries = {
        ordered[math.ceil(step * len(ordered) / quantile_count) - 1]
        for step in range(1, quantile_count)
    }
    return tuple(sorted(boundaries))


def regression_metric_payload(
    predictions: tuple[float, ...],
    targets: tuple[float, ...],
) -> dict[str, object]:
    """Return required bucket/global regression statistics."""
    if not targets:
        return {
            "count": 0,
            "mse": None,
            "rmse": None,
            "mae": None,
            "mean_residual": None,
            "prediction_mean": None,
            "target_mean": None,
            "pearson_correlation": None,
        }
    stats = regression_quality_stats(
        predictions=torch.tensor(predictions, dtype=torch.float64),
        targets=torch.tensor(targets, dtype=torch.float64),
    )
    mse = cast("float", stats.mse)
    return {
        "count": stats.count,
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": stats.mae,
        "mean_residual": stats.residual_mean,
        "prediction_mean": stats.prediction_mean,
        "target_mean": stats.target_mean,
        "pearson_correlation": stats.pearson_correlation,
    }


def _bucket_payload(
    *,
    rows: tuple[dict[str, object], ...],
    positions: list[int],
    evaluator_names: tuple[str, ...],
    roles: EvaluatorRoles,
    bucket_index: int,
    boundaries: tuple[float, ...],
) -> dict[str, object]:
    """Return metrics and paired wins for one bucket subset."""
    selected = tuple(rows[position] for position in positions)
    targets = tuple(float(cast("int | float", row["target"])) for row in selected)
    evaluator_metrics = {
        name: regression_metric_payload(
            tuple(_prediction(row, "evaluators", name) for row in selected),
            targets,
        )
        for name in evaluator_names
    }
    ensemble_names = (
        "ordinary_plus_relational",
        "mlp_plus_relational",
        "all_evaluators",
    )
    ensemble_metrics = {
        name: regression_metric_payload(
            tuple(_prediction(row, "ensembles", name) for row in selected),
            targets,
        )
        for name in ensemble_names
    }
    improvements = tuple(
        _squared_error(row, roles.ordinary) - _squared_error(row, roles.relational)
        for row in selected
    )
    wins = tuple(_ordinary_relational_winner(row, roles) for row in selected)
    count = len(selected)
    return {
        "label": _bucket_label(bucket_index, boundaries),
        "lower_exclusive": None if bucket_index == 0 else boundaries[bucket_index - 1],
        "upper_inclusive": (
            None if bucket_index == len(boundaries) else boundaries[bucket_index]
        ),
        "count": count,
        "evaluators": evaluator_metrics,
        "ensembles": ensemble_metrics,
        "ordinary_vs_relational": {
            "ordinary_win_fraction": (
                None if count == 0 else wins.count(roles.ordinary) / count
            ),
            "relational_win_fraction": (
                None if count == 0 else wins.count(roles.relational) / count
            ),
            "tie_fraction": None if count == 0 else wins.count("tie") / count,
            "mean_relational_mse_improvement": (
                None if count == 0 else math.fsum(improvements) / count
            ),
            "median_relational_mse_improvement": (
                None if count == 0 else statistics.median(improvements)
            ),
        },
    }


def _prediction(row: dict[str, object], section: str, name: str) -> float:
    """Read one evaluator or ensemble prediction."""
    section_payload = cast("dict[str, dict[str, float]]", row[section])
    return section_payload[name]["prediction"]


def _squared_error(row: dict[str, object], evaluator_name: str) -> float:
    """Read one evaluator squared error."""
    evaluators = cast("dict[str, dict[str, float]]", row["evaluators"])
    return evaluators[evaluator_name]["squared_error"]


def _ordinary_relational_winner(row: dict[str, object], roles: EvaluatorRoles) -> str:
    """Read the stable ordinary-versus-relational paired winner."""
    pairs = cast("dict[str, dict[str, object]]", row["pairs"])
    return cast("str", pairs["ordinary_vs_relational"]["winner"])


def _bucket_label(index: int, boundaries: tuple[float, ...]) -> str:
    """Return one explicit human-readable interval label."""
    if not boundaries:
        return "all values"
    if index == 0:
        return f"value <= {boundaries[0]:g}"
    if index == len(boundaries):
        return f"value > {boundaries[-1]:g}"
    return f"{boundaries[index - 1]:g} < value <= {boundaries[index]:g}"


__all__ = [
    "BUCKET_FAMILY_SPECS",
    "BucketFamilySpec",
    "build_error_buckets",
    "deterministic_quantile_boundaries",
    "regression_metric_payload",
]
