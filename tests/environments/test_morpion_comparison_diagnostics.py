"""Tests for paired Morpion evaluator comparison diagnostics."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import pytest
import torch
from torch import nn

from chipiron.environments.morpion.players.evaluators.neural_networks import (
    MORPION_ENTITY_RELATION_SCHEMA,
    MORPION_ENTITY_RELATION_TYPE_COUNT,
    MORPION_ENTITY_TOKEN_MODEL_KIND,
    MORPION_RELATION_BIASED_ENTITY_TOKEN_MODEL_KIND,
    MorpionRegressorArgs,
    build_morpion_regressor,
    save_morpion_model_bundle,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.model import (
    MorpionRegressor,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training.cached_index_schedule import (
    cached_index_schedule,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training.comparison_diagnostics import (
    InvalidMorpionComparisonInputError,
    MorpionComparisonBundle,
    MorpionComparisonDiagnosticsArgs,
    MorpionComparisonEvaluationError,
    _ensemble_metrics,
    _pairwise_metrics,
    _parse_bundle_argument,
    _regression_metrics,
    _worst_rows_artifact,
    build_morpion_comparison_diagnostics,
    save_morpion_comparison_diagnostics,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training.flat_tensor_cache import (
    flat_cache_batch,
    load_or_materialize_flat_tensor_cache,
)
from tests.environments.test_morpion_flat_tensor_cache import (
    _build_jsonl_rows_file,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def small_rows_path(tmp_path: Path) -> Path:
    """Return ten real streaming rows with easy-to-identify targets."""
    return _build_jsonl_rows_file(
        tmp_path,
        target_values=tuple(float(value) for value in range(10)),
    )


def _save_constant_flat_bundle(
    tmp_path: Path,
    *,
    name: str,
    constant: float,
) -> MorpionComparisonBundle:
    """Save one real linear bundle with a constant scalar prediction."""
    model_args = MorpionRegressorArgs(model_kind="linear")
    model = build_morpion_regressor(model_args)
    linear = cast("nn.Linear", model.net)
    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.fill_(constant)
    bundle_path = tmp_path / name
    save_morpion_model_bundle(model, bundle_path, model_args=model_args)
    return MorpionComparisonBundle(name, bundle_path)


def _tiny_entity_args(*, relational: bool) -> MorpionRegressorArgs:
    """Return minimal real ordinary or relational Transformer arguments."""
    common: dict[str, object] = {
        "model_kind": (
            MORPION_RELATION_BIASED_ENTITY_TOKEN_MODEL_KIND
            if relational
            else MORPION_ENTITY_TOKEN_MODEL_KIND
        ),
        "entity_max_tokens": 128,
        "entity_d_model": 4,
        "entity_n_head": 1,
        "entity_n_layer": 0,
        "entity_dim_feedforward": 8,
        "entity_dropout_ratio": 0.0,
    }
    if relational:
        common.update(
            entity_relation_schema=MORPION_ENTITY_RELATION_SCHEMA,
            entity_relation_type_count=MORPION_ENTITY_RELATION_TYPE_COUNT,
        )
    return MorpionRegressorArgs(**common)  # type: ignore[arg-type]


def _save_entity_bundle(
    tmp_path: Path,
    *,
    name: str,
    relational: bool,
) -> MorpionComparisonBundle:
    """Save one tiny deterministic real entity-token model bundle."""
    model_args = _tiny_entity_args(relational=relational)
    model = build_morpion_regressor(model_args)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    bundle_path = tmp_path / name
    save_morpion_model_bundle(model, bundle_path, model_args=model_args)
    return MorpionComparisonBundle(name, bundle_path)


def _comparison_args(
    tmp_path: Path,
    rows_path: Path,
    bundles: tuple[MorpionComparisonBundle, ...],
    *,
    validation_fraction: float = 0.2,
) -> MorpionComparisonDiagnosticsArgs:
    """Return CPU-only comparison args for a tiny test run."""
    return MorpionComparisonDiagnosticsArgs(
        dataset_file=rows_path,
        bundles=bundles,
        output_dir=tmp_path / "diagnostics",
        max_rows=10,
        validation_fraction=validation_fraction,
        batch_size=2,
        device="cpu",
    )


def test_two_flat_evaluators_share_canonical_validation_indices(
    tmp_path: Path,
    small_rows_path: Path,
) -> None:
    """Two flat bundles should predict the exact cached streaming validation rows."""
    bundles = (
        _save_constant_flat_bundle(tmp_path, name="flat_a", constant=1.0),
        _save_constant_flat_bundle(tmp_path, name="flat_b", constant=2.0),
    )
    args = _comparison_args(tmp_path, small_rows_path, bundles)

    diagnostics = build_morpion_comparison_diagnostics(args)
    expected = cached_index_schedule(
        row_count=10,
        validation_fraction=0.2,
    ).validation_indices

    assert diagnostics.validation_row_indices == expected == (4, 9)
    assert diagnostics.targets == (4.0, 9.0)
    assert diagnostics.predictions["flat_a"] == (1.0, 1.0)
    assert diagnostics.predictions["flat_b"] == (2.0, 2.0)
    assert diagnostics.split_policy == "index_modulo_5"


def test_flat_and_ordinary_entity_bundles_share_targets(
    tmp_path: Path,
    small_rows_path: Path,
) -> None:
    """Flat and ordinary entity adapters should agree exactly on paired targets."""
    bundles = (
        _save_constant_flat_bundle(tmp_path, name="flat", constant=1.0),
        _save_entity_bundle(
            tmp_path,
            name="ordinary",
            relational=False,
        ),
    )

    diagnostics = build_morpion_comparison_diagnostics(
        _comparison_args(tmp_path, small_rows_path, bundles)
    )

    assert diagnostics.validation_row_indices == (4, 9)
    assert diagnostics.targets == (4.0, 9.0)
    assert set(diagnostics.predictions) == {"flat", "ordinary"}
    assert all(len(values) == 2 for values in diagnostics.predictions.values())


def test_flat_ordinary_and_relational_adapters_use_paired_rows_and_relations(
    tmp_path: Path,
    small_rows_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A relational comparison should forward both tokens and relation triples."""
    bundles = (
        _save_constant_flat_bundle(tmp_path, name="flat", constant=1.0),
        _save_entity_bundle(tmp_path, name="ordinary", relational=False),
        _save_entity_bundle(tmp_path, name="relational", relational=True),
    )
    original_forward = MorpionRegressor.forward
    relational_auxiliary_counts: list[int] = []

    def spying_forward(
        self: MorpionRegressor,
        input_tensor: torch.Tensor,
        *auxiliary_input_tensors: torch.Tensor,
    ) -> torch.Tensor:
        if self.args.model_kind == MORPION_RELATION_BIASED_ENTITY_TOKEN_MODEL_KIND:
            relational_auxiliary_counts.append(len(auxiliary_input_tensors))
            assert auxiliary_input_tensors[0].shape[-1] == 3
        return original_forward(self, input_tensor, *auxiliary_input_tensors)

    monkeypatch.setattr(MorpionRegressor, "forward", spying_forward)

    diagnostics = build_morpion_comparison_diagnostics(
        _comparison_args(tmp_path, small_rows_path, bundles)
    )

    assert diagnostics.targets == (4.0, 9.0)
    assert set(diagnostics.predictions) == {"flat", "ordinary", "relational"}
    assert relational_auxiliary_counts == [1]
    ensembles = cast("dict[str, object]", diagnostics.summary["ensembles"])
    assert "all_evaluators_equal_weight" in ensembles


def test_scalar_metrics_residuals_and_percentiles_are_hand_computable() -> None:
    """Core quality metrics should use prediction-minus-target residuals."""
    metrics = _regression_metrics(
        predictions=(1.0, 3.0, 5.0),
        targets=(1.0, 2.0, 3.0),
    )

    assert metrics["mse"] == pytest.approx(5.0 / 3.0)
    assert metrics["rmse"] == pytest.approx(math.sqrt(5.0 / 3.0))
    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["mean_error"] == pytest.approx(1.0)
    assert metrics["pearson_correlation"] == pytest.approx(1.0)
    assert metrics["r2_vs_mean_baseline"] == pytest.approx(-1.5)
    absolute_percentiles = cast(
        "dict[str, float]",
        metrics["absolute_error_percentiles"],
    )
    squared_percentiles = cast(
        "dict[str, float]",
        metrics["squared_error_percentiles"],
    )
    assert absolute_percentiles["p50"] == pytest.approx(1.0)
    assert absolute_percentiles["p75"] == pytest.approx(1.5)
    assert squared_percentiles["p90"] == pytest.approx(3.4)


def test_pairwise_correlations_wins_ties_and_ensemble_metrics() -> None:
    """Paired metrics should distinguish wins, ties, and equal-weight gains."""
    names = ("a", "b", "c")
    targets = (0.0, 0.0, 0.0)
    predictions = {
        "a": (0.0, 1.0, 2.0),
        "b": (1.0, 1.0, 0.0),
        "c": (3.0, 3.0, 3.0),
    }

    pairwise = cast(
        "dict[str, dict[str, object]]",
        _pairwise_metrics(
            evaluator_names=names,
            predictions=predictions,
            targets=targets,
        ),
    )
    a_vs_b = pairwise["a__vs__b"]
    assert a_vs_b["residual_pearson_correlation"] == pytest.approx(
        -math.sqrt(3.0) / 2.0
    )
    assert a_vs_b["fraction_a_lower_absolute_error"] == pytest.approx(1.0 / 3.0)
    assert a_vs_b["fraction_b_lower_absolute_error"] == pytest.approx(1.0 / 3.0)
    assert a_vs_b["fraction_tied"] == pytest.approx(1.0 / 3.0)
    ensembles = cast(
        "dict[str, dict[str, object]]",
        _ensemble_metrics(
            evaluator_names=names,
            predictions=predictions,
            targets=targets,
        ),
    )
    assert ensembles["a__vs__b"]["mse"] == pytest.approx(0.75)
    assert "all_evaluators_equal_weight" in ensembles
    oracle = ensembles["per_row_oracle"]
    assert oracle["mse"] == pytest.approx(1.0 / 3.0)
    assert oracle["mae"] == pytest.approx(1.0 / 3.0)
    assert oracle["winning_row_count_per_evaluator"] == {"a": 2, "b": 1, "c": 0}
    assert oracle["deployable"] is False


def test_worst_rows_have_deterministic_error_and_disagreement_order() -> None:
    """Equal-error tail rows should use ascending source row index as tie-breaker."""
    worst = _worst_rows_artifact(
        evaluator_names=("a", "b"),
        row_indices=(9, 3, 5),
        targets=(0.0, 0.0, 0.0),
        predictions={"a": (1.0, 1.0, 1.0), "b": (0.0, 0.0, 0.0)},
    )
    by_evaluator = cast(
        "dict[str, list[dict[str, object]]]",
        worst["worst_rows_by_evaluator"],
    )
    disagreements = cast(
        "list[dict[str, object]]",
        worst["largest_disagreements"],
    )

    assert [row["row_index"] for row in by_evaluator["a"]] == [3, 5, 9]
    assert [row["row_index"] for row in disagreements] == [3, 5, 9]
    assert by_evaluator["a"][0]["other_predictions"] == {"b": 0.0}
    assert by_evaluator["a"][0]["other_residuals"] == {"b": 0.0}


def test_saved_jsonl_and_csv_have_one_row_per_validation_item(
    tmp_path: Path,
    small_rows_path: Path,
) -> None:
    """Both streaming prediction formats should contain the same paired row count."""
    bundles = (
        _save_constant_flat_bundle(tmp_path, name="flat one", constant=1.0),
        _save_constant_flat_bundle(tmp_path, name="flat-two", constant=2.0),
    )
    args = _comparison_args(tmp_path, small_rows_path, bundles)
    diagnostics = build_morpion_comparison_diagnostics(args)

    save_morpion_comparison_diagnostics(diagnostics, args.output_dir)

    jsonl_rows = [
        json.loads(line)
        for line in (args.output_dir / "predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    with (args.output_dir / "predictions.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        csv_rows = list(csv.DictReader(stream))
    summary = json.loads((args.output_dir / "summary.json").read_text())

    assert len(jsonl_rows) == len(csv_rows) == 2
    assert jsonl_rows[0]["residuals"]["flat one"] == -3.0
    assert "flat_one_prediction" in csv_rows[0]
    assert "flat_two_prediction" in csv_rows[0]
    assert summary["dataset"]["validation_row_indices"] == [4, 9]
    assert summary["residual_convention"] == "prediction_minus_target"


def test_duplicate_names_malformed_cli_and_missing_bundle_are_rejected(
    tmp_path: Path,
    small_rows_path: Path,
) -> None:
    """Ambiguous comparison inputs should fail before evaluation."""
    existing = _save_constant_flat_bundle(tmp_path, name="bundle", constant=0.0)
    duplicate_args = _comparison_args(
        tmp_path,
        small_rows_path,
        (
            MorpionComparisonBundle("same", existing.bundle_path),
            MorpionComparisonBundle("same", existing.bundle_path),
        ),
    )
    missing_args = _comparison_args(
        tmp_path,
        small_rows_path,
        (existing, MorpionComparisonBundle("missing", tmp_path / "absent")),
    )

    with pytest.raises(InvalidMorpionComparisonInputError, match="Duplicate"):
        build_morpion_comparison_diagnostics(duplicate_args)
    with pytest.raises(InvalidMorpionComparisonInputError, match="does not exist"):
        build_morpion_comparison_diagnostics(missing_args)
    with pytest.raises(Exception, match="NAME=PATH"):
        _parse_bundle_argument("not-a-pair")
    with pytest.raises(Exception, match="NAME=PATH"):
        _parse_bundle_argument("name=")


def test_non_finite_predictions_are_rejected(
    tmp_path: Path,
    small_rows_path: Path,
) -> None:
    """A bundle producing NaN must not result in non-standard JSON numbers."""
    finite = _save_constant_flat_bundle(tmp_path, name="finite", constant=0.0)
    non_finite = _save_constant_flat_bundle(tmp_path, name="nan", constant=math.nan)

    with pytest.raises(MorpionComparisonEvaluationError, match="non-finite"):
        build_morpion_comparison_diagnostics(
            _comparison_args(tmp_path, small_rows_path, (finite, non_finite))
        )


def test_inference_does_not_change_model_parameter_values(tmp_path: Path) -> None:
    """The canonical batch forward used by diagnostics should be read-only."""
    rows_path = _build_jsonl_rows_file(
        tmp_path,
        target_values=(0.0, 1.0, 2.0, 3.0),
    )
    cache = load_or_materialize_flat_tensor_cache(
        rows_path=rows_path,
        row_chunk_size=2,
        max_rows=4,
    )
    model_args = MorpionRegressorArgs(model_kind="linear")
    model = build_morpion_regressor(model_args)
    before = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }
    batch = flat_cache_batch(
        cache=cache,
        row_indices=(1, 3),
        requested_feature_names=model_args.feature_names,
    )

    model.eval()
    with torch.no_grad():
        model(*batch.get_model_input_tensors())

    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name])


def test_repeated_runs_write_deterministic_artifacts(
    tmp_path: Path,
    small_rows_path: Path,
) -> None:
    """Identical inputs should produce byte-identical comparison artifacts."""
    bundles = (
        _save_constant_flat_bundle(tmp_path, name="first", constant=1.0),
        _save_constant_flat_bundle(tmp_path, name="second", constant=2.0),
    )
    first_args = _comparison_args(tmp_path, small_rows_path, bundles)
    second_args = replace(
        first_args,
        output_dir=tmp_path / "diagnostics_second",
    )
    first = build_morpion_comparison_diagnostics(first_args)
    second = build_morpion_comparison_diagnostics(second_args)
    save_morpion_comparison_diagnostics(first, first_args.output_dir)
    save_morpion_comparison_diagnostics(second, second_args.output_dir)

    for file_name in (
        "summary.json",
        "predictions.jsonl",
        "predictions.csv",
        "worst_rows.json",
    ):
        assert (first_args.output_dir / file_name).read_bytes() == (
            second_args.output_dir / file_name
        ).read_bytes()
