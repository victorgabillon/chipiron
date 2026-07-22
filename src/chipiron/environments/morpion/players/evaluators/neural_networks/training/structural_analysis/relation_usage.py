"""Relation frequency associations and learned relation-bias reporting."""

from __future__ import annotations

import bisect
import math
from typing import TYPE_CHECKING, cast

import torch
from coral.neural_networks.models.relation_biased_entity_token_transformer_value_net import (
    RelationBiasedEntityTokenTransformerValueNet,
)

from chipiron.environments.morpion.players.evaluators.neural_networks.bundle import (
    load_morpion_model_bundle,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.entity_relations import (
    MORPION_ENTITY_RELATION_TYPE_COUNT,
    MorpionEntityRelationType,
    is_relational_entity_token_model_kind,
)

from .args import InvalidRelationBiasTableError
from .buckets import deterministic_quantile_boundaries

if TYPE_CHECKING:
    from pathlib import Path

    from .rows import EvaluatorRoles


def build_relation_usage(
    *,
    rows: tuple[dict[str, object], ...],
    roles: EvaluatorRoles,
) -> dict[str, object]:
    """Report active relation frequencies and confounded presence associations."""
    relation_types: dict[str, object] = {}
    row_count = len(rows)
    for relation_type in MorpionEntityRelationType:
        if relation_type is MorpionEntityRelationType.NO_RELATION:
            continue
        name = relation_type.name
        counts = tuple(_relation_count(row, name) for row in rows)
        present_positions = [index for index, count in enumerate(counts) if count > 0]
        absent_positions = [index for index, count in enumerate(counts) if count == 0]
        present_counts = tuple(counts[index] for index in present_positions)
        quartile_boundaries = deterministic_quantile_boundaries(present_counts, 4)
        absent_association = _association_metrics(rows, absent_positions, roles)
        present_association = _association_metrics(rows, present_positions, roles)
        relation_types[name] = {
            "numeric_relation_id": int(relation_type),
            "name": name,
            "total_occurrence_count": sum(counts),
            "row_presence_count": len(present_positions),
            "row_presence_fraction": len(present_positions) / row_count,
            "mean_count_per_row": math.fsum(counts) / row_count,
            "mean_count_conditional_on_presence": (
                None
                if not present_counts
                else math.fsum(present_counts) / len(present_counts)
            ),
            **_count_percentiles(counts),
            "maximum_count": max(counts, default=0),
            "presence_association": {
                "absent": absent_association,
                "present": present_association,
                "present_minus_absent_relational_improvement": (
                    _presence_improvement_contrast(
                        absent_association,
                        present_association,
                    )
                ),
                "present_count_quartile_boundaries": list(quartile_boundaries),
                "present_count_quartiles": [
                    {
                        "label": _quartile_label(index, quartile_boundaries),
                        "count_range_lower_exclusive": (
                            None if index == 0 else quartile_boundaries[index - 1]
                        ),
                        "count_range_upper_inclusive": (
                            None
                            if index == len(quartile_boundaries)
                            else quartile_boundaries[index]
                        ),
                        **_association_metrics(
                            rows,
                            [
                                position
                                for position in present_positions
                                if _quartile_index(
                                    counts[position], quartile_boundaries
                                )
                                == index
                            ],
                            roles,
                        ),
                    }
                    for index in range(len(quartile_boundaries) + 1)
                ],
            },
        }
    total_relations = tuple(
        int(
            cast(
                "int | float",
                cast("dict[str, object]", row["relations"])["total"],
            )
        )
        for row in rows
    )
    active_names = tuple(
        relation_type.name
        for relation_type in MorpionEntityRelationType
        if relation_type is not MorpionEntityRelationType.NO_RELATION
    )
    move_to_move_names = tuple(
        name for name in active_names if name.startswith("MOVES_")
    )
    return {
        "schema": "morpion_relation_usage_analysis_v1",
        "interpretation": {
            "analysis_kind": "observational_relation_frequency_association",
            "causal_ablation": False,
            "confounding_warning": (
                "Relation presence is confounded with state complexity; these "
                "associations do not establish causal relation contribution."
            ),
        },
        "overall": {
            "row_count": row_count,
            "total_relations": sum(total_relations),
            "mean_relations_per_row": math.fsum(total_relations) / row_count,
            "relation_count_percentiles": _count_percentiles(total_relations),
            "fraction_rows_with_every_relation_type_present": sum(
                all(_relation_count(row, name) > 0 for name in active_names)
                for row in rows
            )
            / row_count,
            "fraction_rows_with_no_move_to_move_relation": sum(
                all(_relation_count(row, name) == 0 for name in move_to_move_names)
                for row in rows
            )
            / row_count,
        },
        "relation_types": relation_types,
    }


def _presence_improvement_contrast(
    absent: dict[str, object],
    present: dict[str, object],
) -> float | None:
    """Return the observational present-minus-absent gain when identifiable."""
    absent_gain = absent["relational_improvement"]
    present_gain = present["relational_improvement"]
    if absent_gain is None or present_gain is None:
        return None
    return float(cast("int | float", present_gain)) - float(
        cast("int | float", absent_gain)
    )


def build_relation_bias_report(bundle_path: Path | None) -> dict[str, object]:
    """Load and describe the typed shared relation-bias embedding on CPU."""
    if bundle_path is None:
        return {"available": False}
    model, model_args, _ = load_morpion_model_bundle(bundle_path)
    if not is_relational_entity_token_model_kind(
        model_args.model_kind
    ) or not isinstance(model.net, RelationBiasedEntityTokenTransformerValueNet):
        raise InvalidRelationBiasTableError.not_relational()
    weights = model.net.relation_bias.weight.detach().cpu().to(dtype=torch.float64)
    expected_rows = MORPION_ENTITY_RELATION_TYPE_COUNT
    if weights.ndim != 2 or weights.shape[0] != expected_rows:
        raise InvalidRelationBiasTableError.wrong_shape(
            expected_rows,
            tuple(weights.shape),
        )
    if not torch.allclose(
        weights[0], torch.zeros_like(weights[0]), atol=1e-8, rtol=0.0
    ):
        raise InvalidRelationBiasTableError.nonzero_padding()
    ranks_by_head = _absolute_ranks_by_head(weights)
    relation_payload: dict[str, object] = {}
    for relation_type in MorpionEntityRelationType:
        relation_id = int(relation_type)
        values = tuple(float(value) for value in weights[relation_id].tolist())
        maximum_head = min(
            range(len(values)), key=lambda head: (-abs(values[head]), head)
        )
        relation_payload[relation_type.name] = {
            "numeric_relation_id": relation_id,
            "name": relation_type.name,
            "heads": [
                {
                    "head": head,
                    "raw_bias": value,
                    "absolute_bias": abs(value),
                    "sign": _bias_sign(value),
                    "absolute_magnitude_rank_within_head": ranks_by_head[head][
                        relation_id
                    ],
                }
                for head, value in enumerate(values)
            ],
            "mean_bias_across_heads": math.fsum(values) / len(values),
            "mean_absolute_bias_across_heads": math.fsum(map(abs, values))
            / len(values),
            "maximum_absolute_bias": abs(values[maximum_head]),
            "head_with_maximum_absolute_bias": maximum_head,
        }
    return {
        "available": True,
        "bundle_path": str(bundle_path.resolve()),
        "relation_schema": model_args.entity_relation_schema,
        "relation_type_count": expected_rows,
        "attention_head_count": int(weights.shape[1]),
        "relation_zero_padding_validated": True,
        "shared_across_transformer_layers": True,
        "shared_table_note": (
            "The current Coral relation-bias table is shared across all "
            "Transformer layers."
        ),
        "relation_types": relation_payload,
    }


def _association_metrics(
    rows: tuple[dict[str, object], ...],
    positions: list[int],
    roles: EvaluatorRoles,
) -> dict[str, object]:
    """Return requested MSE association metrics for one row subset."""
    if not positions:
        return {
            "row_count": 0,
            "ordinary_mse": None,
            "relational_mse": None,
            "relational_improvement": None,
            "ordinary_plus_relational_ensemble_mse": None,
        }
    ordinary_errors = tuple(
        _squared_error(rows[index], roles.ordinary) for index in positions
    )
    relational_errors = tuple(
        _squared_error(rows[index], roles.relational) for index in positions
    )
    ensemble_errors = tuple(
        _ensemble_squared_error(rows[index], "ordinary_plus_relational")
        for index in positions
    )
    count = len(positions)
    ordinary_mse = math.fsum(ordinary_errors) / count
    relational_mse = math.fsum(relational_errors) / count
    return {
        "row_count": count,
        "ordinary_mse": ordinary_mse,
        "relational_mse": relational_mse,
        "relational_improvement": ordinary_mse - relational_mse,
        "ordinary_plus_relational_ensemble_mse": math.fsum(ensemble_errors) / count,
    }


def _count_percentiles(counts: tuple[int, ...]) -> dict[str, float]:
    """Return linearly interpolated count percentiles."""
    tensor = torch.tensor(counts, dtype=torch.float64)
    quantiles = torch.tensor((0.5, 0.9, 0.95, 0.99), dtype=torch.float64)
    values = torch.quantile(tensor, quantiles)
    return {
        name: float(value.item())
        for name, value in zip(
            ("p50_count", "p90_count", "p95_count", "p99_count"), values, strict=True
        )
    }


def _relation_count(row: dict[str, object], name: str) -> int:
    """Read one named active relation count."""
    relations = cast("dict[str, object]", row["relations"])
    counts = cast("dict[str, int]", relations["counts_by_type"])
    return counts[name]


def _squared_error(row: dict[str, object], evaluator: str) -> float:
    """Read one evaluator squared error."""
    evaluators = cast("dict[str, dict[str, float]]", row["evaluators"])
    return evaluators[evaluator]["squared_error"]


def _ensemble_squared_error(row: dict[str, object], ensemble: str) -> float:
    """Read one ensemble squared error."""
    ensembles = cast("dict[str, dict[str, float]]", row["ensembles"])
    return ensembles[ensemble]["squared_error"]


def _quartile_index(value: int, boundaries: tuple[float, ...]) -> int:
    """Return lower-inclusive-boundary quartile index."""
    return bisect.bisect_left(boundaries, value)


def _quartile_label(index: int, boundaries: tuple[float, ...]) -> str:
    """Return a stable present-count quartile label."""
    if not boundaries:
        return "all present rows"
    if index == 0:
        return f"count <= {boundaries[0]:g}"
    if index == len(boundaries):
        return f"count > {boundaries[-1]:g}"
    return f"{boundaries[index - 1]:g} < count <= {boundaries[index]:g}"


def _absolute_ranks_by_head(weights: torch.Tensor) -> tuple[dict[int, int], ...]:
    """Return deterministic one-based absolute-magnitude ranks per head."""
    ranks: list[dict[int, int]] = []
    for head in range(int(weights.shape[1])):
        ordered_ids = sorted(
            range(int(weights.shape[0])),
            key=lambda relation_id: (
                -abs(float(weights[relation_id, head])),
                relation_id,
            ),
        )
        ranks.append({
            relation_id: rank for rank, relation_id in enumerate(ordered_ids, start=1)
        })
    return tuple(ranks)


def _bias_sign(value: float) -> str:
    """Return a stable textual sign for one learned scalar bias."""
    if value > 0.0:
        return "positive"
    if value < 0.0:
        return "negative"
    return "zero"


__all__ = ["build_relation_bias_report", "build_relation_usage"]
