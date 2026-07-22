"""Build and atomically persist multi-seed relation-intervention reports."""
# ruff: noqa: TRY003, TRY004
# pyright: reportArgumentType=false, reportUnnecessaryCast=false

from __future__ import annotations

import bisect
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TextIO, cast

from chipiron.environments.morpion.players.evaluators.neural_networks.entity_relations import (
    MorpionEntityRelationType,
)

from .args import (
    InvalidMorpionRelationInterventionInputError,
    MorpionRelationInterventionArgs,
)
from .inference import (
    InterventionInferenceSet,
    run_relation_intervention_inference,
)
from .metrics import (
    MSE_SIGN_TOLERANCE,
    PREDICTION_CHANGE_TOLERANCE,
    intervention_metrics,
    multi_seed_metrics,
    pearson_correlation,
)

if TYPE_CHECKING:
    from collections.abc import Callable

RELATION_INTERVENTION_SCHEMA = "morpion_relation_interventions_v1"
_STRUCTURAL_FAMILIES = (
    "target_deciles",
    "state_move_count",
    "legal_action_count",
    "relation_count",
    "moves_per_new_dot_max",
    "shared_new_dot_pair_count",
    "shared_prospective_segment_pair_count",
)


@dataclass(frozen=True, slots=True)
class _StructuralInputs:
    """Validated PR 5A2 rows and report payloads."""

    rows: tuple[dict[str, object], ...]
    error_buckets: dict[str, object]
    relation_usage: dict[str, object]


@dataclass(frozen=True, slots=True)
class MorpionRelationInterventionReport:
    """Complete compact inference state and six output artifacts."""

    inference: InterventionInferenceSet
    summary: dict[str, object]
    per_seed: dict[str, object]
    structural_intervention_buckets: dict[str, object]
    relation_importance_table: dict[str, object]
    top_intervention_examples: dict[str, object]


def build_morpion_relation_interventions(
    args: MorpionRelationInterventionArgs,
) -> MorpionRelationInterventionReport:
    """Run fixed-model interventions and build all requested diagnostics."""
    inference = run_relation_intervention_inference(args)
    structural = _load_structural_inputs(
        args.structural_analysis_dir,
        expected_indices=inference.row_indices,
    )
    per_seed_payload, metrics_by_intervention = _build_per_seed(inference)
    aggregate = {
        name: multi_seed_metrics(seed_metrics)
        for name, seed_metrics in metrics_by_intervention.items()
    }
    structural_buckets = _build_structural_buckets(
        inference,
        structural,
    )
    importance = _build_relation_importance(
        inference,
        aggregate,
        metrics_by_intervention,
        structural,
    )
    top_examples = _build_top_examples(inference, structural)
    definitions = [
        {
            "name": definition.name,
            "kind": definition.kind,
            "relation_type_ids": list(definition.relation_type_ids),
            "relation_type_names": [
                MorpionEntityRelationType(relation_id).name
                for relation_id in definition.relation_type_ids
            ],
        }
        for definition in inference.definitions
    ]
    summary: dict[str, object] = {
        "schema": RELATION_INTERVENTION_SCHEMA,
        "analysis_kind": "fixed_model_inference_time_relation_intervention",
        "full_training_ablation": False,
        "interpretation_warning": (
            "Relation removal was applied only to forward inputs of fixed trained "
            "models and was not followed by model retraining."
        ),
        "sign_convention": {
            "prediction_delta": "intervened_prediction_minus_baseline_prediction",
            "squared_error_delta_vs_baseline": (
                "intervened_squared_error_minus_baseline_squared_error"
            ),
            "positive_mse_delta_meaning": (
                "disabling the relation worsened the fixed trained model"
            ),
        },
        "tolerances": {
            "prediction_change": PREDICTION_CHANGE_TOLERANCE,
            "multi_seed_mse_sign": MSE_SIGN_TOLERANCE,
        },
        "dataset": {
            "path": str(args.dataset_file.resolve()),
            "source_row_count": inference.source_row_count,
            "effective_row_count": inference.effective_row_count,
            "validation_fraction": args.validation_fraction,
            "split_policy": inference.split_policy,
            "validation_row_count": len(inference.row_indices),
            "validation_row_indices": list(inference.row_indices),
            "cache_path": str(inference.cache.paths.tensor_path.resolve()),
            "cache_rebuilt": inference.cache.rebuilt,
        },
        "requested_device": inference.requested_device,
        "resolved_device": inference.resolved_device,
        "seed_ids": [seed.seed for seed in inference.seeds],
        "intervention_definitions": definitions,
        "multi_seed_interventions": aggregate,
        "structural_analysis_available": structural is not None,
        "artifact_references": {
            "per_seed": "per_seed.json",
            "predictions": "intervention_predictions.jsonl",
            "structural_buckets": "structural_intervention_buckets.json",
            "relation_importance": "relation_importance_table.json",
            "top_examples": "top_intervention_examples.json",
        },
    }
    return MorpionRelationInterventionReport(
        inference=inference,
        summary=summary,
        per_seed=per_seed_payload,
        structural_intervention_buckets=structural_buckets,
        relation_importance_table=importance,
        top_intervention_examples=top_examples,
    )


def save_morpion_relation_interventions(
    report: MorpionRelationInterventionReport,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Stage and publish the complete six-artifact directory atomically."""
    target = output_dir.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging.", dir=target.parent)
    )
    try:
        _write_json(staging / "summary.json", report.summary)
        _write_json(staging / "per_seed.json", report.per_seed)
        _write_predictions(staging / "intervention_predictions.jsonl", report)
        _write_json(
            staging / "structural_intervention_buckets.json",
            report.structural_intervention_buckets,
        )
        _write_json(
            staging / "relation_importance_table.json",
            report.relation_importance_table,
        )
        _write_json(
            staging / "top_intervention_examples.json",
            report.top_intervention_examples,
        )
        _publish_staging(staging, target, overwrite=overwrite)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _build_per_seed(
    inference: InterventionInferenceSet,
) -> tuple[dict[str, object], dict[str, dict[int, dict[str, object]]]]:
    """Build per-seed metrics and the transposed aggregation input."""
    by_intervention: dict[str, dict[int, dict[str, object]]] = {
        definition.name: {} for definition in inference.definitions
    }
    seeds: dict[str, object] = {}
    for seed in inference.seeds:
        interventions: dict[str, object] = {}
        for definition in inference.definitions:
            metrics = intervention_metrics(
                baseline=seed.baseline_predictions,
                intervened=seed.intervention_predictions[definition.name],
                targets=seed.targets,
            )
            interventions[definition.name] = metrics
            by_intervention[definition.name][seed.seed] = metrics
        baseline = intervention_metrics(
            baseline=seed.baseline_predictions,
            intervened=seed.baseline_predictions,
            targets=seed.targets,
        )
        ordinary = (
            None
            if seed.ordinary_predictions is None
            else _regression_metrics(seed.ordinary_predictions, seed.targets)
        )
        seeds[str(seed.seed)] = {
            "seed": seed.seed,
            "relational_bundle": str(seed.relational_bundle),
            "ordinary_bundle": (
                None if seed.ordinary_bundle is None else str(seed.ordinary_bundle)
            ),
            "recomputed_baseline_metrics": {
                "count": baseline["count"],
                "mse": baseline["baseline_mse"],
                "mae": baseline["baseline_mae"],
            },
            "saved_bundle_metrics": seed.saved_bundle_metrics,
            "baseline_metric_difference": seed.saved_bundle_metrics[
                "baseline_mse_difference"
            ],
            "ordinary_context_metrics": ordinary,
            "interventions": interventions,
        }
    return (
        {
            "schema": RELATION_INTERVENTION_SCHEMA,
            "sign_convention": "positive_mse_delta_means_disabled_relation_helped_fixed_model",
            "seeds": seeds,
        },
        by_intervention,
    )


def _regression_metrics(
    predictions: tuple[float, ...], targets: tuple[float, ...]
) -> dict[str, object]:
    """Return compact ordinary-context regression metrics."""
    return {
        "count": len(targets),
        "mse": math.fsum(
            (prediction - target) ** 2
            for prediction, target in zip(predictions, targets, strict=True)
        )
        / len(targets),
        "mae": math.fsum(
            abs(prediction - target)
            for prediction, target in zip(predictions, targets, strict=True)
        )
        / len(targets),
    }


def _load_structural_inputs(
    directory: Path | None,
    *,
    expected_indices: tuple[int, ...],
) -> _StructuralInputs | None:
    """Load and validate exact PR 5A2 row order and persisted buckets."""
    if directory is None:
        return None
    try:
        rows = tuple(_iter_jsonl(directory / "structural_rows.jsonl"))
        error_buckets = _read_json(directory / "error_buckets.json")
        relation_usage = _read_json(directory / "relation_usage.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise InvalidMorpionRelationInterventionInputError.invalid(
            f"malformed structural analysis in {directory!s}"
        ) from exc
    actual_indices = tuple(int(row["row_index"]) for row in rows)
    if actual_indices != expected_indices:
        raise InvalidMorpionRelationInterventionInputError.invalid(
            "structural-analysis row indices do not exactly match validation indices"
        )
    families = cast("dict[str, object]", error_buckets.get("families"))
    if any(name not in families for name in _STRUCTURAL_FAMILIES):
        raise InvalidMorpionRelationInterventionInputError.invalid(
            "structural analysis is missing required persisted bucket families"
        )
    return _StructuralInputs(rows, error_buckets, relation_usage)


def _build_structural_buckets(
    inference: InterventionInferenceSet,
    structural: _StructuralInputs | None,
) -> dict[str, object]:
    """Apply persisted PR 5A2 bucket boundaries to every intervention."""
    if structural is None:
        return {"schema": RELATION_INTERVENTION_SCHEMA, "available": False}
    source_families = cast(
        "dict[str, dict[str, object]]", structural.error_buckets["families"]
    )
    memberships = {
        name: _family_memberships(structural.rows, source_families[name])
        for name in _STRUCTURAL_FAMILIES
    }
    interventions: dict[str, object] = {}
    for definition in inference.definitions:
        seed_payload: dict[str, object] = {}
        for seed in inference.seeds:
            family_payload: dict[str, object] = {}
            intervened = seed.intervention_predictions[definition.name]
            for family_name in _STRUCTURAL_FAMILIES:
                source_family = source_families[family_name]
                source_buckets = cast(
                    "list[dict[str, object]]", source_family["buckets"]
                )
                family_payload[family_name] = {
                    "value_path": source_family["value_path"],
                    "boundaries": source_family["boundaries"],
                    "boundary_generation_policy": source_family[
                        "boundary_generation_policy"
                    ],
                    "buckets": [
                        {
                            "label": source_bucket["label"],
                            **_subset_intervention_metrics(
                                positions=positions,
                                baseline=seed.baseline_predictions,
                                intervened=intervened,
                                targets=seed.targets,
                            ),
                        }
                        for source_bucket, positions in zip(
                            source_buckets,
                            memberships[family_name],
                            strict=True,
                        )
                    ],
                }
            seed_payload[str(seed.seed)] = family_payload
        interventions[definition.name] = seed_payload
    return {
        "schema": RELATION_INTERVENTION_SCHEMA,
        "available": True,
        "interpretation_warning": (
            "These are inference-time interventions within fixed persisted subsets, "
            "not retrained architecture ablations."
        ),
        "interventions": interventions,
    }


def _family_memberships(
    rows: tuple[dict[str, object], ...], family: dict[str, object]
) -> tuple[tuple[int, ...], ...]:
    """Assign every structural row using persisted lower-boundary semantics."""
    boundaries = tuple(
        float(value) for value in cast("list[object]", family["boundaries"])
    )
    value_path = str(family["value_path"])
    positions: list[list[int]] = [[] for _ in range(len(boundaries) + 1)]
    for position, row in enumerate(rows):
        value = float(_nested_value(row, value_path))
        positions[bisect.bisect_left(boundaries, value)].append(position)
    if sum(map(len, positions)) != len(rows):
        raise InvalidMorpionRelationInterventionInputError.invalid(
            "persisted structural buckets do not partition all rows"
        )
    return tuple(tuple(bucket) for bucket in positions)


def _subset_intervention_metrics(
    *,
    positions: tuple[int, ...],
    baseline: tuple[float, ...],
    intervened: tuple[float, ...],
    targets: tuple[float, ...],
) -> dict[str, object]:
    """Return requested intervention metrics for one fixed structural subset."""
    if not positions:
        return {
            "count": 0,
            "baseline_mse": None,
            "intervened_mse": None,
            "mse_delta": None,
            "mean_absolute_prediction_delta": None,
        }
    count = len(positions)
    baseline_mse = (
        math.fsum((baseline[index] - targets[index]) ** 2 for index in positions)
        / count
    )
    intervened_mse = (
        math.fsum((intervened[index] - targets[index]) ** 2 for index in positions)
        / count
    )
    return {
        "count": count,
        "baseline_mse": baseline_mse,
        "intervened_mse": intervened_mse,
        "mse_delta": intervened_mse - baseline_mse,
        "mean_absolute_prediction_delta": math.fsum(
            abs(intervened[index] - baseline[index]) for index in positions
        )
        / count,
    }


def _build_relation_importance(
    inference: InterventionInferenceSet,
    aggregate: dict[str, dict[str, object]],
    per_intervention: dict[str, dict[int, dict[str, object]]],
    structural: _StructuralInputs | None,
) -> dict[str, object]:
    """Join individual effects with frequency and per-seed learned biases."""
    usage_types: dict[str, dict[str, object]] = {}
    if structural is not None:
        usage_types = cast(
            "dict[str, dict[str, object]]",
            structural.relation_usage["relation_types"],
        )
    rows: list[dict[str, object]] = []
    for relation_type in MorpionEntityRelationType:
        if relation_type is MorpionEntityRelationType.NO_RELATION:
            continue
        name = relation_type.name
        intervention_name = f"disable_{name}"
        usage = usage_types.get(name, {})
        biases_by_seed = {
            str(seed.seed): cast("list[float]", seed.relation_biases[name])
            for seed in inference.seeds
        }
        flat_biases = tuple(
            value for values in biases_by_seed.values() for value in values
        )
        row = {
            "relation_id": int(relation_type),
            "relation_name": name,
            "intervention_name": intervention_name,
            "occurrence_statistics": {
                key: usage.get(key)
                for key in (
                    "total_occurrence_count",
                    "row_presence_count",
                    "row_presence_fraction",
                    "mean_count_conditional_on_presence",
                )
            },
            "learned_biases_by_seed": biases_by_seed,
            "mean_absolute_learned_bias": math.fsum(map(abs, flat_biases))
            / len(flat_biases),
            "per_seed_mse_deltas": {
                str(seed): metrics["mse_delta"]
                for seed, metrics in sorted(per_intervention[intervention_name].items())
            },
            **aggregate[intervention_name],
        }
        rows.append(row)
    by_mse = sorted(
        rows,
        key=lambda row: (-float(row["mean_mse_delta"]), str(row["relation_name"])),
    )
    by_change = sorted(
        rows,
        key=lambda row: (
            -float(row["mean_absolute_prediction_delta"]),
            str(row["relation_name"]),
        ),
    )
    mse_rank = {str(row["relation_name"]): rank for rank, row in enumerate(by_mse, 1)}
    change_rank = {
        str(row["relation_name"]): rank for rank, row in enumerate(by_change, 1)
    }
    for row in rows:
        name = str(row["relation_name"])
        row["rank_by_mean_mse_delta"] = mse_rank[name]
        row["rank_by_mean_absolute_prediction_delta"] = change_rank[name]
        occurrence = cast("dict[str, object]", row["occurrence_statistics"])
        if occurrence.get("total_occurrence_count") == 0:
            intervention_name = str(row["intervention_name"])
            if any(
                float(metrics["fraction_rows_prediction_changed_above_1e_6"]) != 0.0
                or abs(float(metrics["mse_delta"])) > MSE_SIGN_TOLERANCE
                for metrics in per_intervention[intervention_name].values()
            ):
                raise InvalidMorpionRelationInterventionInputError.invalid(
                    f"zero-occurrence relation {name} changed predictions"
                )
    correlations = _importance_correlations(rows)
    return {
        "schema": RELATION_INTERVENTION_SCHEMA,
        "individual_relations_only": True,
        "interpretation_warning": (
            "Frequency and learned bias magnitude do not independently establish "
            "importance; reported effects are fixed-model inference interventions."
        ),
        "correlations_across_relation_types": correlations,
        "relations": sorted(rows, key=lambda row: int(row["relation_id"])),
    }


def _importance_correlations(rows: list[dict[str, object]]) -> dict[str, object]:
    """Correlate individual mean MSE deltas with frequency and learned bias."""
    mse = tuple(float(row["mean_mse_delta"]) for row in rows)
    counts = tuple(
        math.log1p(
            float(
                cast("dict[str, object]", row["occurrence_statistics"])[
                    "total_occurrence_count"
                ]
                or 0
            )
        )
        for row in rows
    )
    presence = tuple(
        float(
            cast("dict[str, object]", row["occurrence_statistics"])[
                "row_presence_fraction"
            ]
            or 0
        )
        for row in rows
    )
    biases = tuple(float(row["mean_absolute_learned_bias"]) for row in rows)
    return {
        "mean_mse_delta_vs_log1p_total_occurrence_count": pearson_correlation(
            mse, counts
        ),
        "mean_mse_delta_vs_mean_absolute_learned_bias": pearson_correlation(
            mse, biases
        ),
        "mean_mse_delta_vs_row_presence_fraction": pearson_correlation(mse, presence),
    }


def _build_top_examples(
    inference: InterventionInferenceSet,
    structural: _StructuralInputs | None,
) -> dict[str, object]:
    """Build three deterministic top-50 lists per active relation type."""
    structural_rows = None if structural is None else structural.rows
    relations: dict[str, object] = {}
    for relation_type in MorpionEntityRelationType:
        if relation_type is MorpionEntityRelationType.NO_RELATION:
            continue
        name = relation_type.name
        intervention = f"disable_{name}"
        candidates: list[dict[str, object]] = []
        for seed in inference.seeds:
            predictions = seed.intervention_predictions[intervention]
            for position, row_index in enumerate(seed.row_indices):
                baseline = seed.baseline_predictions[position]
                changed = predictions[position]
                target = seed.targets[position]
                error_delta = (changed - target) ** 2 - (baseline - target) ** 2
                structural_row = (
                    None if structural_rows is None else structural_rows[position]
                )
                relation_count = None
                key_structure: dict[str, object] = {}
                if structural_row is not None:
                    relation_count = cast(
                        "dict[str, int]",
                        cast("dict[str, object]", structural_row["relations"])[
                            "counts_by_type"
                        ],
                    )[name]
                    key_structure = {
                        "state": structural_row["state"],
                        "geometry": structural_row["geometry"],
                        "relations_total": cast(
                            "dict[str, object]", structural_row["relations"]
                        )["total"],
                    }
                candidates.append({
                    "seed": seed.seed,
                    "row_index": row_index,
                    "target": target,
                    "baseline_prediction": baseline,
                    "intervened_prediction": changed,
                    "baseline_error": baseline - target,
                    "intervened_error": changed - target,
                    "prediction_delta": changed - baseline,
                    "squared_error_delta": error_delta,
                    "relation_count_for_disabled_type": relation_count,
                    "key_structural_fields": key_structure,
                })
        relations[name] = {
            "largest_performance_degradation_when_disabled": _top_records(
                candidates, lambda row: float(row["squared_error_delta"])
            ),
            "largest_performance_improvement_when_disabled": _top_records(
                candidates, lambda row: -float(row["squared_error_delta"])
            ),
            "largest_absolute_prediction_change": _top_records(
                candidates, lambda row: abs(float(row["prediction_delta"]))
            ),
        }
    return {"schema": RELATION_INTERVENTION_SCHEMA, "relations": relations}


def _top_records(
    candidates: list[dict[str, object]],
    score: Callable[[dict[str, object]], float],
) -> list[dict[str, object]]:
    """Sort top examples by score, seed, and row index."""
    ordered = sorted(
        candidates,
        key=lambda row: (
            -score(row),
            int(row["seed"]),
            int(row["row_index"]),
        ),
    )[:50]
    return [{"score": score(row), **row} for row in ordered]


def _write_predictions(path: Path, report: MorpionRelationInterventionReport) -> None:
    """Stream one nested record per seed and validation row."""
    with path.open("w", encoding="utf-8") as handle:
        for seed in report.inference.seeds:
            for position, row_index in enumerate(seed.row_indices):
                baseline = seed.baseline_predictions[position]
                target = seed.targets[position]
                interventions = {
                    definition.name: _prediction_delta_payload(
                        baseline=baseline,
                        intervened=seed.intervention_predictions[definition.name][
                            position
                        ],
                        target=target,
                    )
                    for definition in report.inference.definitions
                }
                _dump_line(
                    handle,
                    {
                        "seed": seed.seed,
                        "row_index": row_index,
                        "target": target,
                        "baseline_prediction": baseline,
                        "ordinary_prediction": (
                            None
                            if seed.ordinary_predictions is None
                            else seed.ordinary_predictions[position]
                        ),
                        "interventions": interventions,
                    },
                )


def _prediction_delta_payload(
    *, baseline: float, intervened: float, target: float
) -> dict[str, float]:
    """Return per-row prediction and signed error deltas."""
    return {
        "prediction": intervened,
        "prediction_delta": intervened - baseline,
        "squared_error_delta_vs_baseline": (
            (intervened - target) ** 2 - (baseline - target) ** 2
        ),
    }


def _nested_value(row: dict[str, object], value_path: str) -> int | float:
    """Read one persisted dotted structural value path."""
    value: object = row
    for part in value_path.split("."):
        value = cast("dict[str, object]", value)[part]
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise InvalidMorpionRelationInterventionInputError.invalid(
            f"structural value path {value_path!r} is not numeric"
        )
    return value


def _iter_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    """Read finite JSON objects from one structural JSONL file."""
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line, parse_constant=_reject_nonfinite)
            if not isinstance(payload, dict):
                raise ValueError("expected JSON object")
            rows.append(cast("dict[str, object]", payload))
    return tuple(rows)


def _read_json(path: Path) -> dict[str, object]:
    """Read one finite JSON object."""
    payload = json.loads(path.read_text(), parse_constant=_reject_nonfinite)
    if not isinstance(payload, dict):
        raise ValueError("expected JSON object")
    return cast("dict[str, object]", payload)


def _reject_nonfinite(value: str) -> object:
    """Reject non-standard JSON numeric constants."""
    raise ValueError(f"non-finite JSON value {value}")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Write deterministic finite JSON."""
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _dump_line(handle: TextIO, payload: dict[str, object]) -> None:
    """Write one deterministic finite JSONL record."""
    handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")


def _publish_staging(staging: Path, target: Path, *, overwrite: bool) -> None:
    """Publish a complete directory, rolling back overwrite swaps on failure."""
    if not target.exists():
        staging.replace(target)
        return
    if not target.is_dir():
        raise InvalidMorpionRelationInterventionInputError.invalid(
            f"output path is not a directory: {target!s}"
        )
    if not any(target.iterdir()):
        target.rmdir()
        staging.replace(target)
        return
    if not overwrite:
        raise InvalidMorpionRelationInterventionInputError.invalid(
            f"output directory is non-empty: {target!s}; pass --overwrite"
        )
    backup = Path(tempfile.mkdtemp(prefix=f".{target.name}.backup.", dir=target.parent))
    backup.rmdir()
    target.replace(backup)
    try:
        staging.replace(target)
    except Exception:
        backup.replace(target)
        raise
    shutil.rmtree(backup)


__all__ = [
    "RELATION_INTERVENTION_SCHEMA",
    "MorpionRelationInterventionReport",
    "build_morpion_relation_interventions",
    "save_morpion_relation_interventions",
]
