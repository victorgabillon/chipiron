"""CLI for analysis-only Morpion structural error attribution."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

from .args import MorpionStructuralAnalysisArgs, MorpionStructuralAnalysisError
from .reports import (
    MorpionStructuralAnalysis,
    build_morpion_structural_analysis,
    save_morpion_structural_analysis,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def _parser() -> argparse.ArgumentParser:
    """Build the standalone structural-analysis parser."""
    parser = argparse.ArgumentParser(
        description="Attribute paired Morpion evaluator errors to state structure."
    )
    parser.add_argument("--dataset-file", type=Path, required=True)
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--relational-bundle", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-ranked-bucket-count", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build, save, and summarize one structural-analysis run."""
    namespace = _parser().parse_args(argv)
    args = MorpionStructuralAnalysisArgs(
        dataset_file=namespace.dataset_file,
        comparison_dir=namespace.comparison_dir,
        output_dir=namespace.output_dir,
        relational_bundle=namespace.relational_bundle,
        minimum_ranked_bucket_count=namespace.minimum_ranked_bucket_count,
        overwrite=namespace.overwrite,
    )
    try:
        analysis = build_morpion_structural_analysis(args)
        save_morpion_structural_analysis(
            analysis,
            args.output_dir,
            overwrite=args.overwrite,
        )
    except MorpionStructuralAnalysisError as exc:
        print(f"Morpion structural analysis failed: {exc}", file=sys.stderr)
        return 2
    print_structural_summary(
        analysis,
        minimum_bucket_count=args.minimum_ranked_bucket_count,
    )
    return 0


def print_structural_summary(
    analysis: MorpionStructuralAnalysis,
    *,
    minimum_bucket_count: int,
) -> None:
    """Print the requested concise numerical association summary."""
    summary = analysis.structural_summary
    roles = cast("dict[str, str]", summary["evaluator_roles"])
    evaluators = cast(
        "dict[str, dict[str, object]]", summary["global_evaluator_metrics"]
    )
    ensembles = cast("dict[str, dict[str, object]]", summary["global_ensemble_metrics"])
    ordinary = roles["ordinary_transformer"]
    relational = roles["relational_transformer"]
    print(f"Selected validation rows: {len(analysis.rows)}")
    print(f"Ordinary MSE: {evaluators[ordinary]['mse']:.8g}")
    print(f"Relational MSE: {evaluators[relational]['mse']:.8g}")
    print(
        "Ordinary + relational ensemble MSE: "
        f"{ensembles['ordinary_plus_relational']['mse']:.8g}"
    )
    print(
        "Ordinary-relational residual correlation: "
        f"{summary['ordinary_relational_residual_correlation']}"
    )
    print(f"Mean relational gain: {summary['mean_relational_squared_error_gain']:.8g}")
    print(f"Top buckets (minimum count {minimum_bucket_count}):")
    for item in _ranked_buckets(analysis, minimum_bucket_count)[:5]:
        print(
            f"  {item['family']} / {item['label']}: "
            f"count={item['count']} gain={item['gain']:.8g}"
        )
    print("Relation presence-minus-absence improvement associations:")
    for item in _ranked_relation_associations(analysis)[:5]:
        print(
            f"  {item['name']}: present={item['present_count']} "
            f"absent={item['absent_count']} contrast={item['contrast']:.8g}"
        )
    print("Largest learned relation biases:")
    for item in _ranked_biases(analysis)[:5]:
        print(f"  {item['name']} head={item['head']} bias={item['bias']:.8g}")


def _ranked_buckets(
    analysis: MorpionStructuralAnalysis,
    minimum_count: int,
) -> list[dict[str, object]]:
    """Return eligible buckets sorted by mean relational improvement."""
    families = cast(
        "dict[str, dict[str, object]]",
        analysis.error_buckets["families"],
    )
    ranked: list[dict[str, object]] = []
    for family_name, family in families.items():
        buckets = cast("list[dict[str, object]]", family["buckets"])
        for bucket in buckets:
            count = cast("int", bucket["count"])
            paired = cast("dict[str, object]", bucket["ordinary_vs_relational"])
            gain = paired["mean_relational_mse_improvement"]
            if count >= minimum_count and gain is not None:
                ranked.append({
                    "family": family_name,
                    "label": bucket["label"],
                    "count": count,
                    "gain": float(cast("int | float", gain)),
                })
    return sorted(
        ranked,
        key=lambda item: (
            -cast("float", item["gain"]),
            cast("str", item["family"]),
            cast("str", item["label"]),
        ),
    )


def _ranked_relation_associations(
    analysis: MorpionStructuralAnalysis,
) -> list[dict[str, object]]:
    """Return comparable observational present-minus-absent associations."""
    relation_types = cast(
        "dict[str, dict[str, object]]",
        analysis.relation_usage["relation_types"],
    )
    ranked: list[dict[str, object]] = []
    for name, payload in relation_types.items():
        associations = cast(
            "dict[str, object]",
            payload["presence_association"],
        )
        present = cast("dict[str, object]", associations["present"])
        absent = cast("dict[str, object]", associations["absent"])
        contrast = associations["present_minus_absent_relational_improvement"]
        if contrast is not None:
            ranked.append({
                "name": name,
                "present_count": present["row_count"],
                "absent_count": absent["row_count"],
                "contrast": float(cast("int | float", contrast)),
            })
    return sorted(
        ranked,
        key=lambda item: (
            -cast("float", item["contrast"]),
            cast("str", item["name"]),
        ),
    )


def _ranked_biases(analysis: MorpionStructuralAnalysis) -> list[dict[str, object]]:
    """Return individual relation/head biases sorted by absolute magnitude."""
    if analysis.relation_biases.get("available") is not True:
        return []
    relation_types = cast(
        "dict[str, dict[str, object]]",
        analysis.relation_biases["relation_types"],
    )
    ranked: list[dict[str, object]] = []
    for name, payload in relation_types.items():
        heads = cast("list[dict[str, object]]", payload["heads"])
        ranked.extend(
            {
                "name": name,
                "head": head["head"],
                "bias": head["raw_bias"],
                "absolute_bias": head["absolute_bias"],
            }
            for head in heads
        )
    return sorted(
        ranked,
        key=lambda item: (
            -cast("float", item["absolute_bias"]),
            cast("str", item["name"]),
            cast("int", item["head"]),
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "print_structural_summary"]
