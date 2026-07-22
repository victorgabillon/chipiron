"""Tests for fixed-model multi-seed Morpion relation interventions."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import pytest
import torch

from chipiron.environments.morpion.players.evaluators.neural_networks import (
    MorpionRelationalEntityTokenConverter,
    build_morpion_regressor,
    load_morpion_model_bundle,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.entity_relations import (
    MORPION_ENTITY_RELATION_TYPE_COUNT,
    MorpionEntityRelationType,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training.comparison_diagnostics import (
    build_morpion_comparison_diagnostics,
    save_morpion_comparison_diagnostics,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training.relation_interventions import (
    InvalidMorpionRelationInterventionInputError,
    MorpionRelationInterventionArgs,
    MorpionRelationInterventionBundle,
    RelationInterventionDefinition,
    all_relation_interventions,
    build_morpion_relation_interventions,
    disable_relation_types,
    save_morpion_relation_interventions,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training.relation_interventions.cli import (
    main,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training.relation_interventions.definitions import (
    group_relation_interventions,
    individual_relation_interventions,
    validate_intervention_definition,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training.relation_interventions.inference import (
    _validate_prediction_tensor,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training.relation_interventions.metrics import (
    consistency_classification,
    intervention_metrics,
    multi_seed_metrics,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training.structural_analysis import (
    MorpionStructuralAnalysisArgs,
    build_morpion_structural_analysis,
    save_morpion_structural_analysis,
)
from chipiron.environments.morpion.types import MorpionDynamics
from tests.environments.test_morpion_comparison_diagnostics import (
    _comparison_args,
    _save_entity_bundle,
    _tiny_entity_args,
)
from tests.environments.test_morpion_entity_tokens import _make_one_step_state
from tests.environments.test_morpion_flat_tensor_cache import _build_jsonl_rows_file
from tests.environments.test_morpion_structural_analysis import (
    _save_constant_mlp_bundle,
)

if TYPE_CHECKING:
    from pathlib import Path

    from coral.neural_networks.models.relation_biased_entity_token_transformer_value_net import (
        RelationBiasedEntityTokenTransformerValueNet,
    )


@pytest.fixture
def intervention_args(tmp_path: Path) -> MorpionRelationInterventionArgs:
    """Build tiny real bundles, comparison, and PR 5A2 structural artifacts."""
    rows_path = _build_jsonl_rows_file(
        tmp_path,
        target_values=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
    )
    relational = _save_entity_bundle(tmp_path, name="relational", relational=True)
    ordinary = _save_entity_bundle(tmp_path, name="ordinary", relational=False)
    mlp = _save_constant_mlp_bundle(tmp_path, name="mlp", constant=0.0)
    comparison_args = replace(
        _comparison_args(
            tmp_path,
            rows_path,
            (mlp, ordinary, relational),
            validation_fraction=0.5,
        ),
        max_rows=6,
        output_dir=tmp_path / "comparison",
    )
    comparison = build_morpion_comparison_diagnostics(comparison_args)
    save_morpion_comparison_diagnostics(comparison, comparison_args.output_dir)
    structural_args = MorpionStructuralAnalysisArgs(
        dataset_file=rows_path,
        comparison_dir=comparison_args.output_dir,
        output_dir=tmp_path / "structural",
        minimum_ranked_bucket_count=1,
    )
    structural = build_morpion_structural_analysis(structural_args)
    save_morpion_structural_analysis(structural, structural_args.output_dir)
    return MorpionRelationInterventionArgs(
        dataset_file=rows_path,
        bundles=(
            MorpionRelationInterventionBundle(
                seed=0,
                relational_bundle=relational.bundle_path,
                ordinary_bundle=ordinary.bundle_path,
            ),
        ),
        output_dir=tmp_path / "interventions",
        structural_analysis_dir=structural_args.output_dir,
        max_rows=6,
        validation_fraction=0.5,
        batch_size=2,
        device="cpu",
    )


def test_definitions_are_complete_deterministic_and_validated() -> None:
    """Individual/group definitions should be stable and reject invalid IDs."""
    first = all_relation_interventions()
    second = all_relation_interventions()
    individuals = individual_relation_interventions()
    groups = group_relation_interventions()

    assert first == second
    assert len(individuals) == MORPION_ENTITY_RELATION_TYPE_COUNT - 1
    assert {definition.name for definition in groups} == {
        "disable_group_move_to_dot_slots",
        "disable_group_dot_to_move_slots",
        "disable_group_edge_dot_connectivity",
        "disable_group_existing_move_window_edges",
        "disable_group_move_to_move",
        "disable_group_all_bidirectional_move_dot",
    }
    with pytest.raises(InvalidMorpionRelationInterventionInputError, match="duplicate"):
        validate_intervention_definition(
            RelationInterventionDefinition("bad", (1, 1), "test")
        )
    with pytest.raises(InvalidMorpionRelationInterventionInputError, match="unknown"):
        validate_intervention_definition(
            RelationInterventionDefinition("bad", (999,), "test")
        )


def test_relation_disabling_clones_and_changes_only_type_column() -> None:
    """Single/all disabling must preserve cache inputs, endpoints, and padding."""
    original = torch.tensor([[[2, 3, 1], [4, 5, 2], [0, 0, 0]]])
    before = original.clone()

    single = disable_relation_types(original, (1,))
    all_disabled = disable_relation_types(original, tuple(range(1, 16)))

    torch.testing.assert_close(original, before)
    torch.testing.assert_close(single[..., :2], original[..., :2])
    assert single[..., 2].tolist() == [[0, 2, 0]]
    torch.testing.assert_close(all_disabled[..., :2], original[..., :2])
    assert torch.count_nonzero(all_disabled[..., 2]).item() == 0
    assert all_disabled[0, 2].tolist() == [0, 0, 0]


def test_metrics_sign_multi_seed_statistics_and_classifications() -> None:
    """Signed deltas and population aggregation should be hand-computable."""
    metrics = intervention_metrics(
        baseline=(0.0, 2.0),
        intervened=(1.0, 1.0),
        targets=(0.0, 0.0),
    )
    assert metrics["baseline_mse"] == pytest.approx(2.0)
    assert metrics["intervened_mse"] == pytest.approx(1.0)
    assert metrics["mse_delta"] == pytest.approx(-1.0)
    assert metrics["fraction_rows_improved_by_intervention"] == 0.5
    aggregate = multi_seed_metrics({
        0: {**metrics, "mse_delta": 1.0, "mae_delta": 0.5},
        1: {**metrics, "mse_delta": 3.0, "mae_delta": 1.5},
    })
    assert aggregate["mean_mse_delta"] == pytest.approx(2.0)
    assert aggregate["population_standard_deviation_mse_delta"] == pytest.approx(1.0)
    assert aggregate["consistency_classification"] == "consistently_helpful"
    assert consistency_classification((-1.0, -2.0)) == "consistently_harmful"
    assert consistency_classification((-1.0, 1.0)) == "mixed"
    assert consistency_classification((0.0, 0.0)) == "inactive"


def test_real_nonzero_relation_bias_changes_fixed_model_prediction() -> None:
    """A present relation with learned bias should affect a real Coral forward."""
    torch.manual_seed(7)
    # Two layers let relation-biased entity updates propagate to the value token.
    model_args = replace(_tiny_entity_args(relational=True), entity_n_layer=2)
    model = build_morpion_regressor(model_args).eval()
    relational = MorpionRelationalEntityTokenConverter(
        dynamics=MorpionDynamics(), max_tokens=128
    ).state_to_tensors(_make_one_step_state())
    relation_id = int(relational.relation_triples[0, 2].item())
    relational_net = cast(
        "RelationBiasedEntityTokenTransformerValueNet",
        model.net,
    )
    with torch.no_grad():
        relational_net.relation_bias.weight[relation_id].fill_(2.0)
        baseline = model(relational.token_tensor, relational.relation_triples)
        disabled = model(
            relational.token_tensor,
            disable_relation_types(relational.relation_triples, (relation_id,)),
        )
    assert not torch.equal(baseline, disabled)


def test_absent_relation_has_exactly_zero_direct_effect() -> None:
    """Disabling a type absent from the forward input should be an identity."""
    model = build_morpion_regressor(_tiny_entity_args(relational=True)).eval()
    relational = MorpionRelationalEntityTokenConverter(max_tokens=128).state_to_tensors(
        _make_one_step_state()
    )
    absent = int(MorpionEntityRelationType.MOVE_WINDOW_CONTAINS_EDGE)
    assert not bool(torch.any(relational.relation_triples[:, 2] == absent))
    with torch.inference_mode():
        baseline = model(relational.token_tensor, relational.relation_triples)
        intervened = model(
            relational.token_tensor,
            disable_relation_types(relational.relation_triples, (absent,)),
        )
    torch.testing.assert_close(intervened, baseline, rtol=0.0, atol=0.0)


def test_end_to_end_indices_baseline_ordinary_and_immutability(
    intervention_args: MorpionRelationInterventionArgs,
) -> None:
    """Canonical cached rows and fixed parameters should survive the full run."""
    bundle = intervention_args.bundles[0]
    weights_before = (bundle.relational_bundle / "param.pt").read_bytes()
    report = build_morpion_relation_interventions(intervention_args)
    seed = report.inference.seeds[0]
    cache = report.inference.cache

    assert seed.row_indices == (1, 3, 5)
    assert seed.targets == (1.0, 3.0, 5.0)
    assert seed.ordinary_predictions is not None
    assert len(seed.ordinary_predictions) == len(seed.row_indices)
    assert (bundle.relational_bundle / "param.pt").read_bytes() == weights_before
    assert cache.packed_relation_triples.device.type == "cpu"
    assert report.summary["full_training_ablation"] is False
    model, _, _ = load_morpion_model_bundle(bundle.relational_bundle)
    batch = report.inference.cache
    from chipiron.environments.morpion.players.evaluators.neural_networks.training.relational_entity_token_cache import (
        relational_entity_token_cache_batch,
    )

    sample = relational_entity_token_cache_batch(cache=batch, row_indices=(1, 3, 5))
    with torch.inference_mode():
        direct = model(*sample.get_model_input_tensors()).reshape(-1).tolist()
    assert seed.baseline_predictions == pytest.approx(direct)


def test_structural_join_reuses_boundaries_and_frequency_metadata(
    intervention_args: MorpionRelationInterventionArgs,
) -> None:
    """PR 5A2 row order, bucket definitions, usage, and dead types should join."""
    report = build_morpion_relation_interventions(intervention_args)
    artifact = report.structural_intervention_buckets
    structural_dir = cast("Path", intervention_args.structural_analysis_dir)
    source = json.loads(
        (structural_dir / "error_buckets.json").read_text()
    )
    intervention = cast("dict[str, dict[str, object]]", artifact["interventions"])[
        "disable_all_relations"
    ]["0"]
    target = cast("dict[str, object]", intervention)["target_deciles"]
    assert (
        cast("dict[str, object]", target)["boundaries"]
        == source["families"]["target_deciles"]["boundaries"]
    )
    relations = cast(
        "list[dict[str, object]]", report.relation_importance_table["relations"]
    )
    by_name = {row["relation_name"]: row for row in relations}
    for dead_name in ("MOVE_WINDOW_CONTAINS_EDGE", "EDGE_IN_MOVE_WINDOW"):
        dead = by_name[dead_name]
        occurrence = cast("dict[str, object]", dead["occurrence_statistics"])
        assert occurrence["total_occurrence_count"] == 0
        assert dead["consistency_classification"] == "inactive"
        assert dead["mean_mse_delta"] == 0.0


def test_structural_row_mismatch_is_rejected(
    tmp_path: Path,
    intervention_args: MorpionRelationInterventionArgs,
) -> None:
    """Structural rows must join the validation schedule by exact order and index."""
    structural = tmp_path / "bad_structural"
    structural.mkdir()
    source = cast("Path", intervention_args.structural_analysis_dir)
    for name in ("error_buckets.json", "relation_usage.json"):
        (structural / name).write_bytes((source / name).read_bytes())
    rows = (source / "structural_rows.jsonl").read_text().splitlines()
    (structural / "structural_rows.jsonl").write_text("\n".join(reversed(rows)) + "\n")
    with pytest.raises(InvalidMorpionRelationInterventionInputError, match="indices"):
        build_morpion_relation_interventions(
            replace(intervention_args, structural_analysis_dir=structural)
        )


def test_nonfinite_predictions_are_rejected() -> None:
    """Inference validation should reject non-finite model outputs."""
    with pytest.raises(Exception, match="non-finite predictions"):
        _validate_prediction_tensor(0, "test", torch.tensor([float("nan")]))


def test_wrong_bundle_kind_and_duplicate_seed_are_rejected(
    intervention_args: MorpionRelationInterventionArgs,
) -> None:
    """Relational bundle identity and seed uniqueness should be strict."""
    bundle = intervention_args.bundles[0]
    wrong = replace(bundle, relational_bundle=cast("Path", bundle.ordinary_bundle))
    with pytest.raises(
        InvalidMorpionRelationInterventionInputError, match="relational"
    ):
        build_morpion_relation_interventions(
            replace(intervention_args, bundles=(wrong,))
        )
    with pytest.raises(InvalidMorpionRelationInterventionInputError, match="duplicate"):
        build_morpion_relation_interventions(
            replace(intervention_args, bundles=(bundle, bundle))
        )


def test_atomic_outputs_are_deterministic_and_require_overwrite(
    tmp_path: Path,
    intervention_args: MorpionRelationInterventionArgs,
) -> None:
    """Complete artifact sets should be byte-stable and protected by default."""
    report = build_morpion_relation_interventions(intervention_args)
    first = tmp_path / "first"
    second = tmp_path / "second"
    save_morpion_relation_interventions(report, first)
    save_morpion_relation_interventions(report, second)
    expected = {
        "summary.json",
        "per_seed.json",
        "intervention_predictions.jsonl",
        "structural_intervention_buckets.json",
        "relation_importance_table.json",
        "top_intervention_examples.json",
    }
    assert {path.name for path in first.iterdir()} == expected
    assert (first / "intervention_predictions.jsonl").read_text().count("\n") == 3
    for name in expected:
        assert (first / name).read_bytes() == (second / name).read_bytes()
    with pytest.raises(InvalidMorpionRelationInterventionInputError, match="overwrite"):
        save_morpion_relation_interventions(report, first)
    save_morpion_relation_interventions(report, first, overwrite=True)


def test_cli_succeeds_and_rejects_malformed_seed_paths(
    tmp_path: Path,
    intervention_args: MorpionRelationInterventionArgs,
) -> None:
    """The CLI should run tiny real bundles and reject malformed repeatable args."""
    bundle = intervention_args.bundles[0]
    output = tmp_path / "cli"
    result = main([
        "--dataset-file",
        str(intervention_args.dataset_file),
        "--relational-bundle",
        f"0={bundle.relational_bundle}",
        "--ordinary-bundle",
        f"0={bundle.ordinary_bundle}",
        "--structural-analysis-dir",
        str(intervention_args.structural_analysis_dir),
        "--output-dir",
        str(output),
        "--max-rows",
        "6",
        "--validation-fraction",
        "0.5",
        "--batch-size",
        "2",
        "--device",
        "cpu",
    ])
    assert result == 0
    assert (output / "summary.json").is_file()
    assert (
        main([
            "--dataset-file",
            str(intervention_args.dataset_file),
            "--relational-bundle",
            "bad",
            "--output-dir",
            str(tmp_path / "bad"),
        ])
        == 2
    )
