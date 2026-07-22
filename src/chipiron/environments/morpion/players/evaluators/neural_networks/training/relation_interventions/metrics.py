"""Metrics for fixed-model Morpion relation interventions."""
# ruff: noqa: TRY003
# pyright: reportArgumentType=false

from __future__ import annotations

import math

MSE_SIGN_TOLERANCE = 1e-8
PREDICTION_CHANGE_TOLERANCE = 1e-6
ROW_ERROR_TIE_TOLERANCE = 1e-12


def intervention_metrics(
    *,
    baseline: tuple[float, ...],
    intervened: tuple[float, ...],
    targets: tuple[float, ...],
) -> dict[str, object]:
    """Return all requested per-seed intervention metrics."""
    if not (len(baseline) == len(intervened) == len(targets)) or not targets:
        raise ValueError("Intervention metrics require equal non-empty inputs.")
    base_errors = tuple((p - t) ** 2 for p, t in zip(baseline, targets, strict=True))
    alt_errors = tuple((p - t) ** 2 for p, t in zip(intervened, targets, strict=True))
    base_abs = tuple(abs(p - t) for p, t in zip(baseline, targets, strict=True))
    alt_abs = tuple(abs(p - t) for p, t in zip(intervened, targets, strict=True))
    deltas = tuple(a - b for a, b in zip(intervened, baseline, strict=True))
    error_deltas = tuple(a - b for a, b in zip(alt_errors, base_errors, strict=True))
    count = len(targets)
    baseline_mse = math.fsum(base_errors) / count
    intervened_mse = math.fsum(alt_errors) / count
    mse_delta = intervened_mse - baseline_mse
    improved = sum(delta < -ROW_ERROR_TIE_TOLERANCE for delta in error_deltas)
    worsened = sum(delta > ROW_ERROR_TIE_TOLERANCE for delta in error_deltas)
    tied = count - improved - worsened
    mean_delta = math.fsum(deltas) / count
    variance = math.fsum((delta - mean_delta) ** 2 for delta in deltas) / count
    return {
        "count": count,
        "baseline_mse": baseline_mse,
        "intervened_mse": intervened_mse,
        "mse_delta": mse_delta,
        "relative_mse_delta": None if baseline_mse == 0.0 else mse_delta / baseline_mse,
        "baseline_mae": math.fsum(base_abs) / count,
        "intervened_mae": math.fsum(alt_abs) / count,
        "mae_delta": (math.fsum(alt_abs) - math.fsum(base_abs)) / count,
        "prediction_delta_mean": mean_delta,
        "prediction_delta_population_standard_deviation": math.sqrt(variance),
        "mean_absolute_prediction_delta": math.fsum(map(abs, deltas)) / count,
        "maximum_absolute_prediction_delta": max(map(abs, deltas)),
        "fraction_rows_prediction_changed_above_1e_6": sum(
            abs(delta) > PREDICTION_CHANGE_TOLERANCE for delta in deltas
        )
        / count,
        "fraction_rows_improved_by_intervention": improved / count,
        "fraction_rows_worsened_by_intervention": worsened / count,
        "fraction_rows_tied": tied / count,
    }


def multi_seed_metrics(per_seed: dict[int, dict[str, object]]) -> dict[str, object]:
    """Aggregate intervention effects across independently trained seeds."""
    seeds = tuple(sorted(per_seed))
    mse_deltas = tuple(float(per_seed[s]["mse_delta"]) for s in seeds)
    mae_deltas = tuple(float(per_seed[s]["mae_delta"]) for s in seeds)
    abs_deltas = tuple(
        float(per_seed[s]["mean_absolute_prediction_delta"]) for s in seeds
    )
    mean = math.fsum(mse_deltas) / len(mse_deltas)
    std = math.sqrt(
        math.fsum((value - mean) ** 2 for value in mse_deltas) / len(mse_deltas)
    )
    return {
        "completed_seed_count": len(seeds),
        "seed_ids": list(seeds),
        "mean_mse_delta": mean,
        "population_standard_deviation_mse_delta": std,
        "minimum_mse_delta": min(mse_deltas),
        "maximum_mse_delta": max(mse_deltas),
        "seeds_with_positive_mse_delta": sum(
            value > MSE_SIGN_TOLERANCE for value in mse_deltas
        ),
        "seeds_with_negative_mse_delta": sum(
            value < -MSE_SIGN_TOLERANCE for value in mse_deltas
        ),
        "mean_mae_delta": math.fsum(mae_deltas) / len(mae_deltas),
        "mean_absolute_prediction_delta": math.fsum(abs_deltas) / len(abs_deltas),
        "consistency_classification": consistency_classification(mse_deltas),
    }


def consistency_classification(mse_deltas: tuple[float, ...]) -> str:
    """Classify cross-seed delta signs using the named MSE tolerance."""
    if all(abs(value) <= MSE_SIGN_TOLERANCE for value in mse_deltas):
        return "inactive"
    if all(value > MSE_SIGN_TOLERANCE for value in mse_deltas):
        return "consistently_helpful"
    if all(value < -MSE_SIGN_TOLERANCE for value in mse_deltas):
        return "consistently_harmful"
    return "mixed"


def pearson_correlation(
    first: tuple[float, ...], second: tuple[float, ...]
) -> float | None:
    """Return Pearson correlation, or null when either input is constant."""
    if len(first) != len(second) or len(first) < 2:
        return None
    first_mean = math.fsum(first) / len(first)
    second_mean = math.fsum(second) / len(second)
    first_centered = tuple(value - first_mean for value in first)
    second_centered = tuple(value - second_mean for value in second)
    denominator = math.sqrt(
        math.fsum(value * value for value in first_centered)
        * math.fsum(value * value for value in second_centered)
    )
    if denominator == 0.0:
        return None
    return (
        math.fsum(
            left * right
            for left, right in zip(first_centered, second_centered, strict=True)
        )
        / denominator
    )


__all__ = [
    "MSE_SIGN_TOLERANCE",
    "PREDICTION_CHANGE_TOLERANCE",
    "ROW_ERROR_TIE_TOLERANCE",
    "consistency_classification",
    "intervention_metrics",
    "multi_seed_metrics",
    "pearson_correlation",
]
