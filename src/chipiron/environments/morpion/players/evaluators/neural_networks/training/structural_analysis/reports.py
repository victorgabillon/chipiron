"""Build and atomically save complete Morpion structural-analysis reports."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from chipiron.environments.morpion.players.evaluators.neural_networks.entity_relations import (
    MorpionRelationalEntityTokenConverter,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.entity_tokens import (
    MorpionEntityTokenConverter,
)
from chipiron.environments.morpion.types import MorpionDynamics

from .args import InvalidStructuralAnalysisInputError, MorpionStructuralAnalysisArgs
from .buckets import build_error_buckets, regression_metric_payload
from .features import StructuralFeatureContext, build_structural_row
from .relation_usage import build_relation_bias_report, build_relation_usage
from .rows import load_comparison_inputs, load_selected_source_rows

STRUCTURAL_ANALYSIS_SCHEMA = "morpion_structural_error_analysis_v1"


@dataclass(frozen=True, slots=True)
class MorpionStructuralAnalysis:
    """Complete in-memory structural rows and six output artifact payloads."""

    rows: tuple[dict[str, object], ...]
    structural_summary: dict[str, object]
    error_buckets: dict[str, object]
    relation_usage: dict[str, object]
    relation_biases: dict[str, object]
    top_structural_examples: dict[str, object]


def build_morpion_structural_analysis(
    args: MorpionStructuralAnalysisArgs,
) -> MorpionStructuralAnalysis:
    """Associate canonical paired predictions with selected state structure."""
    comparison = load_comparison_inputs(args)
    source_rows = load_selected_source_rows(args.dataset_file, comparison)
    dynamics = MorpionDynamics()
    context = StructuralFeatureContext(
        dynamics=dynamics,
        token_converter=MorpionEntityTokenConverter(dynamics=dynamics),
        relational_converter=MorpionRelationalEntityTokenConverter(dynamics=dynamics),
    )
    structural_rows = tuple(
        build_structural_row(
            row=row,
            paired=paired,
            evaluator_names=comparison.evaluator_names,
            roles=comparison.roles,
            context=context,
        )
        for row, paired in zip(source_rows, comparison.predictions, strict=True)
    )
    error_buckets = build_error_buckets(
        rows=structural_rows,
        evaluator_names=comparison.evaluator_names,
        roles=comparison.roles,
    )
    relation_usage = build_relation_usage(
        rows=structural_rows,
        roles=comparison.roles,
    )
    relation_biases = build_relation_bias_report(args.relational_bundle)
    top_examples = build_top_structural_examples(
        rows=structural_rows,
        ordinary_name=comparison.roles.ordinary,
        relational_name=comparison.roles.relational,
    )
    global_evaluators = {
        name: regression_metric_payload(
            tuple(_prediction(row, "evaluators", name) for row in structural_rows),
            tuple(float(cast("int | float", row["target"])) for row in structural_rows),
        )
        for name in comparison.evaluator_names
    }
    ensemble_names = (
        "ordinary_plus_relational",
        "mlp_plus_relational",
        "all_evaluators",
    )
    global_ensembles = {
        name: regression_metric_payload(
            tuple(_prediction(row, "ensembles", name) for row in structural_rows),
            tuple(float(cast("int | float", row["target"])) for row in structural_rows),
        )
        for name in ensemble_names
    }
    residual_correlation = _ordinary_relational_residual_correlation(
        structural_rows,
        comparison.roles.ordinary,
        comparison.roles.relational,
    )
    mean_relational_gain = math.fsum(
        _squared_error(row, comparison.roles.ordinary)
        - _squared_error(row, comparison.roles.relational)
        for row in structural_rows
    ) / len(structural_rows)
    bucket_manifest = _bucket_manifest(error_buckets)
    relation_headlines = _relation_headlines(relation_usage)
    summary: dict[str, object] = {
        "schema": STRUCTURAL_ANALYSIS_SCHEMA,
        "source_comparison_schema": comparison.summary["schema"],
        "residual_convention": "prediction_minus_target",
        "analysis_kind": "observational_structural_error_attribution",
        "causal_relation_ablation": False,
        "dataset": {
            "dataset_file": str(args.dataset_file.resolve()),
            "comparison_dir": str(args.comparison_dir.resolve()),
            "selected_validation_row_count": len(structural_rows),
            "validation_row_indices": list(comparison.row_indices),
        },
        "evaluator_names": list(comparison.evaluator_names),
        "evaluator_roles": {
            "ordinary_transformer": comparison.roles.ordinary,
            "relational_transformer": comparison.roles.relational,
            "mlp": comparison.roles.mlp,
        },
        "global_evaluator_metrics": global_evaluators,
        "global_ensemble_metrics": global_ensembles,
        "ordinary_relational_residual_correlation": residual_correlation,
        "mean_relational_squared_error_gain": mean_relational_gain,
        "bucket_manifest": bucket_manifest,
        "relation_usage_headline_findings": relation_headlines,
        "artifact_references": {
            "rows": "structural_rows.jsonl",
            "buckets": "error_buckets.json",
            "relation_usage": "relation_usage.json",
            "relation_biases": "relation_biases.json",
            "top_examples": "top_structural_examples.json",
        },
    }
    return MorpionStructuralAnalysis(
        rows=structural_rows,
        structural_summary=summary,
        error_buckets=error_buckets,
        relation_usage=relation_usage,
        relation_biases=relation_biases,
        top_structural_examples=top_examples,
    )


def save_morpion_structural_analysis(
    analysis: MorpionStructuralAnalysis,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Stage and publish all six artifacts as one complete directory set."""
    target = output_dir.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging.", dir=target.parent)
    )
    try:
        _write_rows(staging / "structural_rows.jsonl", analysis.rows)
        _write_json(staging / "structural_summary.json", analysis.structural_summary)
        _write_json(staging / "error_buckets.json", analysis.error_buckets)
        _write_json(staging / "relation_usage.json", analysis.relation_usage)
        _write_json(staging / "relation_biases.json", analysis.relation_biases)
        _write_json(
            staging / "top_structural_examples.json",
            analysis.top_structural_examples,
        )
        _publish_staging(staging=staging, target=target, overwrite=overwrite)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def build_top_structural_examples(
    *,
    rows: tuple[dict[str, object], ...],
    ordinary_name: str,
    relational_name: str,
) -> dict[str, object]:
    """Build all deterministic top-100 structural example lists."""
    score_functions = {
        "largest_relational_gain_over_ordinary": lambda row: (
            _squared_error(row, ordinary_name) - _squared_error(row, relational_name)
        ),
        "largest_ordinary_gain_over_relational": lambda row: (
            _squared_error(row, relational_name) - _squared_error(row, ordinary_name)
        ),
        "largest_ordinary_relational_prediction_disagreement": lambda row: abs(
            _prediction(row, "evaluators", ordinary_name)
            - _prediction(row, "evaluators", relational_name)
        ),
        "largest_ensemble_gain_over_best_individual_transformer": lambda row: (
            min(
                _squared_error(row, ordinary_name),
                _squared_error(row, relational_name),
            )
            - _ensemble_squared_error(row, "ordinary_plus_relational")
        ),
        "largest_shared_prospective_segment_pair_count": lambda row: _nested_number(
            row, "geometry", "move_pairs_sharing_prospective_segment"
        ),
        "largest_shared_window_overlap_count": lambda row: _nested_number(
            row, "geometry", "move_pairs_sharing_any_window_dot"
        ),
        "largest_relation_count": lambda row: _nested_number(row, "relations", "total"),
    }
    return {
        "schema": "morpion_top_structural_examples_v1",
        "lists": {
            name: [
                _top_example(row, score_function(row))
                for row in sorted(
                    rows,
                    key=lambda item: (
                        -score_function(item),
                        int(cast("int | float", item["row_index"])),
                    ),
                )[:100]
            ]
            for name, score_function in score_functions.items()
        },
    }


def _top_example(row: dict[str, object], score: float) -> dict[str, object]:
    """Return one compact structural example with all predictions and relations."""
    evaluators = cast("dict[str, dict[str, float]]", row["evaluators"])
    relations = cast("dict[str, object]", row["relations"])
    return {
        "row_index": row["row_index"],
        "score": score,
        "target": row["target"],
        "predictions": {
            name: metrics["prediction"] for name, metrics in evaluators.items()
        },
        "errors": {
            name: {
                "residual": metrics["residual"],
                "absolute_error": metrics["absolute_error"],
                "squared_error": metrics["squared_error"],
            }
            for name, metrics in evaluators.items()
        },
        "state": row["state"],
        "tokens": row["tokens"],
        "geometry": row["geometry"],
        "relation_counts": relations["counts_by_type"],
    }


def _ordinary_relational_residual_correlation(
    rows: tuple[dict[str, object], ...],
    ordinary: str,
    relational: str,
) -> float | None:
    """Return global residual Pearson correlation from canonical predictions."""
    ordinary_residuals = tuple(_residual(row, ordinary) for row in rows)
    relational_residuals = tuple(_residual(row, relational) for row in rows)
    metrics = regression_metric_payload(ordinary_residuals, relational_residuals)
    return cast("float | None", metrics["pearson_correlation"])


def _relation_headlines(relation_usage: dict[str, object]) -> dict[str, object]:
    """Return numerical relation-frequency and presence-association headlines."""
    relation_types = cast(
        "dict[str, dict[str, object]]", relation_usage["relation_types"]
    )
    by_occurrence = sorted(
        relation_types,
        key=lambda name: (
            -cast("int", relation_types[name]["total_occurrence_count"]),
            name,
        ),
    )
    comparable_relation_types = tuple(
        name
        for name, payload in relation_types.items()
        if _presence_gain_contrast(payload) is not None
    )
    by_presence_contrast = sorted(
        comparable_relation_types,
        key=lambda name: (
            -cast("float", _presence_gain_contrast(relation_types[name])),
            name,
        ),
    )
    return {
        "interpretation": "observational_frequency_association_not_causal_ablation",
        "overall": relation_usage["overall"],
        "most_frequent_relation_types": by_occurrence[:5],
        "largest_present_minus_absent_relational_improvement_contrasts": [
            {
                "name": name,
                "present_minus_absent_relational_improvement": cast(
                    "float", _presence_gain_contrast(relation_types[name])
                ),
            }
            for name in by_presence_contrast[:5]
        ],
    }


def _presence_gain_contrast(
    relation_payload: dict[str, object],
) -> float | None:
    """Read a comparable present-minus-absent relational-gain contrast."""
    association = cast(
        "dict[str, object]", relation_payload["presence_association"]
    )
    value = association["present_minus_absent_relational_improvement"]
    return None if value is None else float(cast("int | float", value))


def _bucket_manifest(error_buckets: dict[str, object]) -> dict[str, object]:
    """Return compact persisted family policies and boundaries."""
    families = cast("dict[str, dict[str, object]]", error_buckets["families"])
    return {
        name: {
            "value_path": payload["value_path"],
            "boundary_generation_policy": payload["boundary_generation_policy"],
            "boundaries": payload["boundaries"],
        }
        for name, payload in families.items()
    }


def _publish_staging(*, staging: Path, target: Path, overwrite: bool) -> None:
    """Publish a complete staged directory, with rollback for overwrite swaps."""
    if not target.exists():
        staging.replace(target)
        return
    if not target.is_dir():
        raise InvalidStructuralAnalysisInputError.invalid_output(target)
    if not any(target.iterdir()):
        target.rmdir()
        staging.replace(target)
        return
    if not overwrite:
        raise InvalidStructuralAnalysisInputError.output_not_empty(target)
    backup = Path(tempfile.mkdtemp(prefix=f".{target.name}.backup.", dir=target.parent))
    backup.rmdir()
    target.replace(backup)
    try:
        staging.replace(target)
    except Exception:
        backup.replace(target)
        raise
    shutil.rmtree(backup)


def _write_rows(path: Path, rows: tuple[dict[str, object], ...]) -> None:
    """Write streaming-friendly finite deterministic structural JSONL."""
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            json.dump(
                row, stream, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            stream.write("\n")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Write one finite deterministic JSON report."""
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _prediction(row: dict[str, object], section: str, name: str) -> float:
    """Read one prediction from an evaluator or ensemble section."""
    payload = cast("dict[str, dict[str, float]]", row[section])
    return payload[name]["prediction"]


def _squared_error(row: dict[str, object], name: str) -> float:
    """Read one evaluator squared error."""
    payload = cast("dict[str, dict[str, float]]", row["evaluators"])
    return payload[name]["squared_error"]


def _ensemble_squared_error(row: dict[str, object], name: str) -> float:
    """Read one ensemble squared error."""
    payload = cast("dict[str, dict[str, float]]", row["ensembles"])
    return payload[name]["squared_error"]


def _residual(row: dict[str, object], name: str) -> float:
    """Read one evaluator residual."""
    payload = cast("dict[str, dict[str, float]]", row["evaluators"])
    return payload[name]["residual"]


def _nested_number(row: dict[str, object], section: str, name: str) -> float:
    """Read one numeric structural field."""
    payload = cast("dict[str, object]", row[section])
    return float(cast("int | float", payload[name]))


__all__ = [
    "STRUCTURAL_ANALYSIS_SCHEMA",
    "MorpionStructuralAnalysis",
    "build_morpion_structural_analysis",
    "build_top_structural_examples",
    "save_morpion_structural_analysis",
]
