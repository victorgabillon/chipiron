"""CLI for multi-seed fixed-model Morpion relation interventions."""
# ruff: noqa: TRY003
# pyright: reportArgumentType=false

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

from .args import (
    InvalidMorpionRelationInterventionInputError,
    MorpionRelationInterventionArgs,
    MorpionRelationInterventionBundle,
    MorpionRelationInterventionError,
)
from .reports import (
    MorpionRelationInterventionReport,
    build_morpion_relation_interventions,
    save_morpion_relation_interventions,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def _parser() -> argparse.ArgumentParser:
    """Build the standalone relation-intervention parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run inference-time relation removals on fixed trained Morpion models."
        )
    )
    parser.add_argument("--dataset-file", type=Path, required=True)
    parser.add_argument("--relational-bundle", action="append", required=True)
    parser.add_argument("--ordinary-bundle", action="append", default=[])
    parser.add_argument("--structural-analysis-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=50_000)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse, build, atomically save, and summarize one intervention run."""
    namespace = _parser().parse_args(argv)
    try:
        relational = _parse_seed_paths(namespace.relational_bundle, "relational")
        ordinary = _parse_seed_paths(namespace.ordinary_bundle, "ordinary")
        if ordinary and set(ordinary) != set(relational):
            raise InvalidMorpionRelationInterventionInputError.invalid(
                "ordinary bundle seed set must exactly match relational seeds"
            )
        bundles = tuple(
            MorpionRelationInterventionBundle(
                seed=seed,
                relational_bundle=relational[seed],
                ordinary_bundle=ordinary.get(seed),
            )
            for seed in sorted(relational)
        )
        args = MorpionRelationInterventionArgs(
            dataset_file=namespace.dataset_file,
            bundles=bundles,
            output_dir=namespace.output_dir,
            structural_analysis_dir=namespace.structural_analysis_dir,
            max_rows=namespace.max_rows,
            validation_fraction=namespace.validation_fraction,
            batch_size=namespace.batch_size,
            device=namespace.device,
            overwrite=namespace.overwrite,
        )
        report = build_morpion_relation_interventions(args)
        save_morpion_relation_interventions(
            report,
            args.output_dir,
            overwrite=args.overwrite,
        )
    except MorpionRelationInterventionError as exc:
        print(f"Morpion relation interventions failed: {exc}", file=sys.stderr)
        return 2
    print_relation_intervention_summary(report)
    return 0


def _parse_seed_paths(values: list[str], label: str) -> dict[int, Path]:
    """Parse exact repeatable ``SEED=PATH`` values with duplicate rejection."""
    parsed: dict[int, Path] = {}
    for value in values:
        if value.count("=") != 1:
            raise InvalidMorpionRelationInterventionInputError.invalid(
                f"malformed {label} bundle {value!r}; expected SEED=PATH"
            )
        raw_seed, raw_path = value.split("=", 1)
        try:
            seed = int(raw_seed)
        except ValueError as exc:
            raise InvalidMorpionRelationInterventionInputError.invalid(
                f"malformed {label} seed {raw_seed!r}"
            ) from exc
        if seed < 0 or not raw_path:
            raise InvalidMorpionRelationInterventionInputError.invalid(
                f"malformed {label} bundle {value!r}"
            )
        if seed in parsed:
            raise InvalidMorpionRelationInterventionInputError.invalid(
                f"duplicate {label} seed {seed}"
            )
        path = Path(raw_path)
        if not path.is_dir():
            raise InvalidMorpionRelationInterventionInputError.invalid(
                f"missing {label} bundle path {path!s}"
            )
        parsed[seed] = path
    return parsed


def print_relation_intervention_summary(
    report: MorpionRelationInterventionReport,
) -> None:
    """Print baseline, ranked effects, inactive types, and structural hotspots."""
    seeds = cast("dict[str, dict[str, object]]", report.per_seed["seeds"])
    print("Fixed-model inference-time relation interventions (not a training ablation)")
    for seed_id, payload in seeds.items():
        baseline = cast("dict[str, object]", payload["recomputed_baseline_metrics"])
        interventions = cast("dict[str, dict[str, object]]", payload["interventions"])
        print(
            f"Seed {seed_id}: baseline MSE={baseline['mse']:.8g}; "
            "disable-all delta="
            f"{interventions['disable_all_relations']['mse_delta']:.8g}"
        )
    aggregate = cast(
        "dict[str, dict[str, object]]", report.summary["multi_seed_interventions"]
    )
    relations = cast(
        "list[dict[str, object]]", report.relation_importance_table["relations"]
    )
    print("Individual relations by mean MSE delta:")
    for row in sorted(
        relations,
        key=lambda item: (-float(item["mean_mse_delta"]), str(item["relation_name"])),
    ):
        print(
            f"  {row['relation_name']}: delta={row['mean_mse_delta']:.8g} "
            f"class={row['consistency_classification']}"
        )
    print("Relation groups by mean MSE delta:")
    group_names = [
        definition.name
        for definition in report.inference.definitions
        if definition.kind == "group"
    ]
    for name in sorted(
        group_names,
        key=lambda item: (-float(aggregate[item]["mean_mse_delta"]), item),
    ):
        print(f"  {name}: delta={aggregate[name]['mean_mse_delta']:.8g}")
    inactive = [
        str(row["relation_name"])
        for row in relations
        if row["consistency_classification"] == "inactive"
    ]
    mixed = [
        str(row["relation_name"])
        for row in relations
        if row["consistency_classification"] == "mixed"
    ]
    print("Inactive relation types: " + (", ".join(inactive) or "none"))
    print("Mixed relation types: " + (", ".join(mixed) or "none"))
    hotspots = _ranked_structural_hotspots(report)[:5]
    print("Largest structural-bucket intervention effects:")
    for row in hotspots:
        print(
            f"  {row['intervention']} / {row['family']} / {row['label']}: "
            f"mean delta={row['mean_delta']:.8g} count={row['count']}"
        )


def _ranked_structural_hotspots(
    report: MorpionRelationInterventionReport,
) -> list[dict[str, object]]:
    """Rank structural buckets by cross-seed absolute mean MSE effect."""
    artifact = report.structural_intervention_buckets
    if artifact.get("available") is not True:
        return []
    interventions = cast("dict[str, dict[str, object]]", artifact["interventions"])
    rows: list[dict[str, object]] = []
    for intervention, seed_payload in interventions.items():
        seed_ids = sorted(seed_payload, key=int)
        first_seed = cast("dict[str, dict[str, object]]", seed_payload[seed_ids[0]])
        for family_name, family in first_seed.items():
            buckets = cast("list[dict[str, object]]", family["buckets"])
            for bucket_index, bucket in enumerate(buckets):
                deltas: list[float] = []
                for seed_id in seed_ids:
                    seed_families = cast(
                        "dict[str, dict[str, object]]", seed_payload[seed_id]
                    )
                    seed_buckets = cast(
                        "list[dict[str, object]]",
                        seed_families[family_name]["buckets"],
                    )
                    delta = seed_buckets[bucket_index]["mse_delta"]
                    if delta is not None:
                        deltas.append(float(delta))
                if deltas:
                    rows.append({
                        "intervention": intervention,
                        "family": family_name,
                        "label": bucket["label"],
                        "count": bucket["count"],
                        "mean_delta": sum(deltas) / len(deltas),
                    })
    return sorted(
        rows,
        key=lambda row: (
            -abs(float(row["mean_delta"])),
            str(row["intervention"]),
            str(row["family"]),
            str(row["label"]),
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "print_relation_intervention_summary"]
