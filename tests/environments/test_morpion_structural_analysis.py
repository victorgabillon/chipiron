"""Tests for Morpion structural error attribution and relation usage."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import pytest
import torch

from chipiron.environments.morpion.players.evaluators.neural_networks import (
    MorpionRegressorArgs,
    build_morpion_regressor,
    save_morpion_model_bundle,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.entity_relations import (
    MorpionEntityRelationType,
    MorpionRelationalEntityTokenConverter,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training.comparison_diagnostics import (
    MorpionComparisonBundle,
    build_morpion_comparison_diagnostics,
    save_morpion_comparison_diagnostics,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training.structural_analysis import (
    InvalidRelationBiasTableError,
    InvalidStructuralAnalysisInputError,
    MorpionStructuralAnalysisArgs,
    build_morpion_structural_analysis,
    save_morpion_structural_analysis,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training.structural_analysis.buckets import (
    build_error_buckets,
    deterministic_quantile_boundaries,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training.structural_analysis.cli import (
    main,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training.structural_analysis.features import (
    legal_action_geometry,
    relation_counts_by_name,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training.structural_analysis.relation_usage import (
    build_relation_bias_report,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training.structural_analysis.reports import (
    build_top_structural_examples,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training.structural_analysis.rows import (
    EvaluatorRoles,
)
from chipiron.environments.morpion.types import MorpionDynamics
from tests.environments.test_morpion_comparison_diagnostics import (
    _comparison_args,
    _save_entity_bundle,
    _tiny_entity_args,
)
from tests.environments.test_morpion_entity_tokens import _make_one_step_state
from tests.environments.test_morpion_flat_tensor_cache import _build_jsonl_rows_file

if TYPE_CHECKING:
    from pathlib import Path

    from coral.neural_networks.models.relation_biased_entity_token_transformer_value_net import (
        RelationBiasedEntityTokenTransformerValueNet,
    )


@pytest.fixture
def structural_inputs(tmp_path: Path) -> MorpionStructuralAnalysisArgs:
    """Build a tiny real flat/ordinary/relational PR 5A1 artifact."""
    rows_path = _build_jsonl_rows_file(
        tmp_path,
        target_values=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
    )
    bundles = (
        _save_constant_mlp_bundle(tmp_path, name="mlp", constant=1.0),
        _save_entity_bundle(tmp_path, name="ordinary", relational=False),
        _save_entity_bundle(tmp_path, name="relational", relational=True),
    )
    comparison_args = replace(
        _comparison_args(
            tmp_path,
            rows_path,
            bundles,
            validation_fraction=0.5,
        ),
        max_rows=6,
        output_dir=tmp_path / "comparison",
    )
    comparison = build_morpion_comparison_diagnostics(comparison_args)
    save_morpion_comparison_diagnostics(comparison, comparison_args.output_dir)
    return MorpionStructuralAnalysisArgs(
        dataset_file=rows_path,
        comparison_dir=comparison_args.output_dir,
        output_dir=tmp_path / "structural",
        minimum_ranked_bucket_count=1,
    )


def _save_constant_mlp_bundle(
    tmp_path: Path,
    *,
    name: str,
    constant: float,
) -> MorpionComparisonBundle:
    """Save one tiny real MLP with a constant output."""
    model_args = MorpionRegressorArgs(model_kind="mlp", hidden_sizes=(4,))
    model = build_morpion_regressor(model_args)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        output_layer = cast(
            "torch.nn.Linear", cast("torch.nn.Sequential", model.net)[-1]
        )
        output_layer.bias.fill_(constant)
    bundle_path = tmp_path / name
    save_morpion_model_bundle(model, bundle_path, model_args=model_args)
    return MorpionComparisonBundle(name, bundle_path)


def test_selected_rows_and_source_targets_match_comparison(
    structural_inputs: MorpionStructuralAnalysisArgs,
) -> None:
    """Structural rows should preserve exact PR 5A1 validation ordering and targets."""
    analysis = build_morpion_structural_analysis(structural_inputs)

    assert [row["row_index"] for row in analysis.rows] == [1, 3, 5]
    assert [row["target"] for row in analysis.rows] == [1.0, 3.0, 5.0]
    assert analysis.structural_summary["source_comparison_schema"] == (
        "morpion_evaluator_comparison_v1"
    )


@pytest.mark.parametrize("mutation", ("missing", "duplicate", "unexpected", "nan"))
def test_prediction_row_integrity_failures_are_rejected(
    tmp_path: Path,
    structural_inputs: MorpionStructuralAnalysisArgs,
    mutation: str,
) -> None:
    """Missing, duplicate, unexpected, and non-finite prediction rows should fail."""
    comparison_dir = tmp_path / f"comparison_{mutation}"
    shutil.copytree(structural_inputs.comparison_dir, comparison_dir)
    prediction_path = comparison_dir / "predictions.jsonl"
    rows = [json.loads(line) for line in prediction_path.read_text().splitlines()]
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "unexpected":
        rows[0]["row_index"] = 2
    else:
        rows[0]["predictions"]["mlp"] = math.nan
    prediction_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    args = replace(structural_inputs, comparison_dir=comparison_dir)

    with pytest.raises(InvalidStructuralAnalysisInputError):
        build_morpion_structural_analysis(args)


def test_structural_counts_and_relations_match_canonical_converter(
    structural_inputs: MorpionStructuralAnalysisArgs,
) -> None:
    """Per-row token, state, and named relation totals should be internally exact."""
    row = build_morpion_structural_analysis(structural_inputs).rows[0]
    state = cast("dict[str, int]", row["state"])
    tokens = cast("dict[str, int]", row["tokens"])
    relations = cast("dict[str, object]", row["relations"])
    counts = cast("dict[str, int]", relations["counts_by_type"])

    assert state["moves"] == state["move_count"]
    assert tokens["total"] == (1 + tokens["dots"] + tokens["edges"] + tokens["moves"])
    assert sum(counts.values()) == relations["total"]
    assert "NO_RELATION" not in counts
    assert set(counts) == {
        relation.name
        for relation in MorpionEntityRelationType
        if relation is not MorpionEntityRelationType.NO_RELATION
    }


def test_prospective_and_shared_action_geometry_is_hand_computable() -> None:
    """Prospective edges and pair overlaps should include currently undrawn windows."""
    actions = (
        (0, 0, 0, 2),
        (0, 1, 0, 1),
        (1, 2, -2, 2),
    )

    geometry = legal_action_geometry(actions)

    assert geometry["move_pair_count"] == 3
    assert geometry["unique_new_dot_count"] == 1
    assert geometry["move_pairs_sharing_new_dot"] == 3
    assert geometry["move_pairs_sharing_any_window_dot"] == 3
    assert geometry["move_pairs_sharing_two_or_more_window_dots"] == 1
    assert geometry["move_pairs_sharing_prospective_segment"] == 1
    assert geometry["unique_prospective_segment_count"] == 9
    assert geometry["prospective_segment_reuse_count"] == 3
    assert geometry["maximum_prospective_segment_multiplicity"] == 2


def test_named_relation_counts_equal_real_converter_tensor() -> None:
    """Named usage counts should be a lossless view of canonical relation triples."""
    state = _make_one_step_state()
    triples = (
        MorpionRelationalEntityTokenConverter(
            dynamics=MorpionDynamics(), max_tokens=128
        )
        .state_to_tensors(state)
        .relation_triples
    )

    counts = relation_counts_by_name(triples)

    assert sum(counts.values()) == int(triples.shape[0])
    for relation in MorpionEntityRelationType:
        if relation is MorpionEntityRelationType.NO_RELATION:
            continue
        assert counts[relation.name] == int(
            torch.sum(triples[:, 2] == int(relation)).item()
        )


def test_bucket_boundaries_membership_metrics_and_wins_are_deterministic(
    structural_inputs: MorpionStructuralAnalysisArgs,
) -> None:
    """Every row should enter one bucket per family with reproducible metrics."""
    analysis = build_morpion_structural_analysis(structural_inputs)
    roles = EvaluatorRoles(ordinary="ordinary", relational="relational", mlp="mlp")
    first = build_error_buckets(
        rows=analysis.rows,
        evaluator_names=("mlp", "ordinary", "relational"),
        roles=roles,
    )
    second = build_error_buckets(
        rows=analysis.rows,
        evaluator_names=("mlp", "ordinary", "relational"),
        roles=roles,
    )
    families = cast("dict[str, dict[str, object]]", first["families"])

    assert first == second
    assert deterministic_quantile_boundaries((1.0, 2.0, 3.0, 4.0), 4) == (
        1.0,
        2.0,
        3.0,
    )
    for family in families.values():
        buckets = cast("list[dict[str, object]]", family["buckets"])
        assert sum(cast("int", bucket["count"]) for bucket in buckets) == 3
    target_buckets = cast(
        "list[dict[str, object]]", families["target_deciles"]["buckets"]
    )
    nonempty = [bucket for bucket in target_buckets if bucket["count"]]
    metrics = cast("dict[str, dict[str, object]]", nonempty[0]["evaluators"])
    assert metrics["mlp"]["mae"] is not None
    paired = cast("dict[str, object]", nonempty[0]["ordinary_vs_relational"])
    assert cast("float", paired["ordinary_win_fraction"]) + cast(
        "float", paired["relational_win_fraction"]
    ) + cast("float", paired["tie_fraction"]) == pytest.approx(1.0)


def test_relation_presence_subsets_and_no_relation_exclusion(
    structural_inputs: MorpionStructuralAnalysisArgs,
) -> None:
    """Every active relation should partition rows into absent and present subsets."""
    usage = build_morpion_structural_analysis(structural_inputs).relation_usage
    relation_types = cast("dict[str, dict[str, object]]", usage["relation_types"])

    assert "NO_RELATION" not in relation_types
    for payload in relation_types.values():
        associations = cast("dict[str, object]", payload["presence_association"])
        absent = cast("dict[str, object]", associations["absent"])
        present = cast("dict[str, object]", associations["present"])
        assert (
            cast("int", absent["row_count"])
            + cast("int", present["row_count"])
            == 3
        )
        contrast = associations["present_minus_absent_relational_improvement"]
        if (
            cast("int", absent["row_count"]) == 0
            or cast("int", present["row_count"]) == 0
        ):
            assert contrast is None


def test_learned_relation_biases_map_names_heads_and_validate_padding(
    tmp_path: Path,
) -> None:
    """Typed learned biases should retain relation/head identity and neutral row zero."""
    model_args = _tiny_entity_args(relational=True)
    model = build_morpion_regressor(model_args)
    typed_net = cast("RelationBiasedEntityTokenTransformerValueNet", model.net)
    with torch.no_grad():
        typed_net.relation_bias.weight.zero_()
        typed_net.relation_bias.weight[1, 0] = 0.75
    bundle = tmp_path / "relational_bias_bundle"
    save_morpion_model_bundle(model, bundle, model_args=model_args)
    before = (bundle / "param.pt").read_bytes()

    report = build_relation_bias_report(bundle)

    relation_types = cast("dict[str, dict[str, object]]", report["relation_types"])
    slot_zero = relation_types["MOVE_TO_DOT_SLOT_0"]
    heads = cast("list[dict[str, object]]", slot_zero["heads"])
    assert heads[0]["raw_bias"] == pytest.approx(0.75)
    assert heads[0]["absolute_magnitude_rank_within_head"] == 1
    assert report["relation_zero_padding_validated"] is True
    assert report["shared_across_transformer_layers"] is True
    assert (bundle / "param.pt").read_bytes() == before

    with torch.no_grad():
        typed_net.relation_bias.weight[0, 0] = 1.0
    invalid_bundle = tmp_path / "invalid_bias_bundle"
    save_morpion_model_bundle(model, invalid_bundle, model_args=model_args)
    with pytest.raises(InvalidRelationBiasTableError, match="row zero"):
        build_relation_bias_report(invalid_bundle)


def test_top_example_ordering_is_score_then_row_index(
    structural_inputs: MorpionStructuralAnalysisArgs,
) -> None:
    """Top structural examples should have deterministic score/index ordering."""
    rows = build_morpion_structural_analysis(structural_inputs).rows
    report = build_top_structural_examples(
        rows=rows,
        ordinary_name="ordinary",
        relational_name="relational",
    )
    lists = cast("dict[str, list[dict[str, object]]]", report["lists"])

    for examples in lists.values():
        ordering = [
            (-cast("float", item["score"]), cast("int", item["row_index"]))
            for item in examples
        ]
        assert ordering == sorted(ordering)


def test_outputs_are_complete_deterministic_and_require_overwrite(
    tmp_path: Path,
    structural_inputs: MorpionStructuralAnalysisArgs,
) -> None:
    """Six finite artifacts should be deterministic and safe from mixed-run writes."""
    analysis = build_morpion_structural_analysis(structural_inputs)
    first_dir = tmp_path / "first_output"
    second_dir = tmp_path / "second_output"
    save_morpion_structural_analysis(analysis, first_dir)
    save_morpion_structural_analysis(analysis, second_dir)
    file_names = (
        "structural_rows.jsonl",
        "structural_summary.json",
        "error_buckets.json",
        "relation_usage.json",
        "relation_biases.json",
        "top_structural_examples.json",
    )

    assert len((first_dir / "structural_rows.jsonl").read_text().splitlines()) == 3
    for file_name in file_names:
        assert (first_dir / file_name).read_bytes() == (
            second_dir / file_name
        ).read_bytes()
    with pytest.raises(InvalidStructuralAnalysisInputError, match="not empty"):
        save_morpion_structural_analysis(analysis, first_dir)
    save_morpion_structural_analysis(analysis, first_dir, overwrite=True)


def test_cli_succeeds_on_tiny_real_comparison(
    tmp_path: Path,
    structural_inputs: MorpionStructuralAnalysisArgs,
) -> None:
    """The public CLI should consume a real three-adapter comparison artifact."""
    output_dir = tmp_path / "cli_output"

    result = main((
        "--dataset-file",
        str(structural_inputs.dataset_file),
        "--comparison-dir",
        str(structural_inputs.comparison_dir),
        "--output-dir",
        str(output_dir),
        "--minimum-ranked-bucket-count",
        "1",
    ))

    assert result == 0
    assert (output_dir / "structural_summary.json").is_file()
