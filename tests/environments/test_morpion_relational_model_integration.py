"""Integration tests for the relational Morpion entity-token model family."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, cast

import pytest
import torch
from coral.neural_networks.models.relation_biased_entity_token_transformer_value_net import (
    RelationBiasedEntityTokenTransformerValueNet,
)
from coral.neural_networks.nn_model_type import NNModelType
from valanga.evaluations import Certainty

from chipiron.environments.morpion.bootstrap.cycle_training import (
    morpion_training_args_from_evaluator_spec,
)
from chipiron.environments.morpion.bootstrap.evaluator_family import (
    entity_token_relational_transformer_small_morpion_evaluator_spec,
)
from chipiron.environments.morpion.bootstrap.runtime.runner import (
    MorpionRegressorMasterEvaluator,
    load_morpion_evaluator_from_model_bundle,
)
from chipiron.environments.morpion.learning import load_morpion_supervised_rows
from chipiron.environments.morpion.players.evaluators.datasets import (
    collate_morpion_relational_entity_token_supervised_samples,
    process_morpion_supervised_row_to_relational_entity_token_tensors,
)
from chipiron.environments.morpion.players.evaluators.neural_networks import (
    MORPION_ENTITY_RELATION_SCHEMA,
    MORPION_ENTITY_RELATION_TYPE_COUNT,
    MORPION_ENTITY_TOKEN_FEATURE_DIM,
    MORPION_ENTITY_TOKEN_INPUT_REPRESENTATION,
    MORPION_ENTITY_TOKEN_MODEL_KIND,
    MORPION_MANIFEST_FILE_NAME,
    MORPION_MODEL_ARGS_FILE_NAME,
    MORPION_RELATION_BIASED_ENTITY_TOKEN_MODEL_KIND,
    MorpionEntityTokenConverter,
    MorpionRegressorArgs,
    MorpionRelationalEntityTokenConverter,
    build_morpion_regressor,
    load_morpion_model_bundle,
    save_morpion_model_bundle,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training import (
    MorpionStreamingTrainingArgs,
    MorpionTrainingArgs,
    load_or_materialize_relational_entity_token_cache,
    relational_entity_token_cache_batch,
    train_morpion_regressor,
    train_morpion_regressor_streaming,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training.diagnostics import (
    diagnostic_adapter_kind,
    predict_morpion_rows_for_diagnostics,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training.service import (
    prediction_scale_stats_for_cached_batches,
)
from chipiron.environments.morpion.types import MorpionDynamics
from chipiron.learning.supervised import TensorSupervisedBatch, train_regression_batch
from tests.environments.test_morpion_entity_token_cache import (
    _build_jsonl_rows_file,
)
from tests.environments.test_morpion_entity_tokens import (
    _build_rows_file,
    _make_one_step_state,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_relational_model_kind_preset_and_parameter_contract() -> None:
    """The canonical preset should align with Coral and add one bias table."""
    assert (
        MORPION_RELATION_BIASED_ENTITY_TOKEN_MODEL_KIND
        == NNModelType.RELATION_BIASED_ENTITY_TOKEN_TRANSFORMER_VALUE_NET
    )
    preset = entity_token_relational_transformer_small_morpion_evaluator_spec()
    assert preset.name == "entity_token_relational_transformer_small"
    assert preset.model_type == MORPION_RELATION_BIASED_ENTITY_TOKEN_MODEL_KIND
    assert preset.entity_input_feature_dim == MORPION_ENTITY_TOKEN_FEATURE_DIM
    assert preset.entity_max_tokens == 1536
    assert preset.entity_d_model == 64
    assert preset.entity_n_head == 4
    assert preset.entity_n_layer == 2
    assert preset.entity_dim_feedforward == 256
    assert preset.entity_dropout_ratio == 0.0
    assert preset.entity_pooling == "value_token"
    assert preset.entity_output_tanh is False
    assert preset.entity_use_validity_feature is True
    assert preset.entity_relation_schema == MORPION_ENTITY_RELATION_SCHEMA
    assert preset.entity_relation_type_count == MORPION_ENTITY_RELATION_TYPE_COUNT
    training_args = morpion_training_args_from_evaluator_spec(
        spec=preset,
        dataset_file="rows.jsonl",
        output_dir="model",
        shuffle=False,
        validation_fraction=0.2,
        validation_seed=0,
        device="cpu",
    )
    assert training_args.model_kind == MORPION_RELATION_BIASED_ENTITY_TOKEN_MODEL_KIND
    assert training_args.entity_relation_schema == MORPION_ENTITY_RELATION_SCHEMA
    assert training_args.entity_relation_type_count == MORPION_ENTITY_RELATION_TYPE_COUNT

    ordinary = build_morpion_regressor(
        MorpionRegressorArgs(model_kind=MORPION_ENTITY_TOKEN_MODEL_KIND)
    )
    relational = build_morpion_regressor(_relational_model_args())
    ordinary_count = sum(parameter.numel() for parameter in ordinary.parameters())
    relational_count = sum(
        parameter.numel() for parameter in relational.parameters()
    )

    assert ordinary_count == 105_985
    assert relational_count == 106_049
    assert relational_count - ordinary_count == MORPION_ENTITY_RELATION_TYPE_COUNT * 4
    assert isinstance(relational.net, RelationBiasedEntityTokenTransformerValueNet)
    assert relational.net.args.num_relation_types == MORPION_ENTITY_RELATION_TYPE_COUNT


@pytest.mark.parametrize(
    "override",
    (
        {"entity_input_feature_dim": MORPION_ENTITY_TOKEN_FEATURE_DIM + 1},
        {"entity_use_validity_feature": False},
        {"entity_relation_schema": None},
        {"entity_relation_schema": "wrong_relations_v1"},
        {"entity_relation_type_count": None},
        {"entity_relation_type_count": MORPION_ENTITY_RELATION_TYPE_COUNT + 1},
    ),
)
def test_relational_model_args_reject_incompatible_schema(
    override: dict[str, object],
) -> None:
    """Relational construction should require the exact input/relation schema."""
    kwargs: dict[str, object] = {
        "model_kind": MORPION_RELATION_BIASED_ENTITY_TOKEN_MODEL_KIND,
        "entity_relation_schema": MORPION_ENTITY_RELATION_SCHEMA,
        "entity_relation_type_count": MORPION_ENTITY_RELATION_TYPE_COUNT,
    }
    kwargs.update(override)
    with pytest.raises(ValueError):
        MorpionRegressorArgs(**kwargs)  # type: ignore[arg-type]


def test_real_relational_forward_batch_empty_relations_and_truncation() -> None:
    """The real Coral model should support every clean converter shape."""
    torch.manual_seed(0)
    model = build_morpion_regressor(_relational_model_args()).eval()
    state = _make_one_step_state()
    relational = MorpionRelationalEntityTokenConverter(
        max_tokens=128
    ).state_to_tensors(state)

    with torch.no_grad():
        single_output = model(
            relational.token_tensor,
            relational.relation_triples,
        )
    sample = TensorSupervisedBatch(
        input_tensor=relational.token_tensor,
        auxiliary_input_tensors=(relational.relation_triples,),
        target_tensor=torch.tensor([0.0]),
        is_batch=False,
    )
    batch = collate_morpion_relational_entity_token_supervised_samples(
        (sample, sample)
    )
    with torch.no_grad():
        batch_output = model(*batch.get_model_input_tensors())

    global_only = MorpionRelationalEntityTokenConverter(max_tokens=1).state_to_tensors(
        state
    )
    with torch.no_grad():
        global_output = model(
            global_only.token_tensor,
            global_only.relation_triples,
        )
    truncated = MorpionRelationalEntityTokenConverter(max_tokens=64).state_to_tensors(
        state
    )
    with torch.no_grad():
        truncated_output = model(
            truncated.token_tensor,
            truncated.relation_triples,
        )

    assert single_output.shape == (1,)
    assert batch_output.shape == (2, 1)
    assert global_only.token_tensor.shape == (1, MORPION_ENTITY_TOKEN_FEATURE_DIM)
    assert global_only.relation_triples.shape == (0, 3)
    assert torch.isfinite(single_output).all()
    assert torch.isfinite(batch_output).all()
    assert torch.isfinite(global_output).all()
    assert torch.isfinite(truncated_output).all()


def test_real_relational_backward_reaches_active_bias_rows() -> None:
    """Two relation-biased layers should differentiate active relation rows."""
    torch.manual_seed(0)
    model = build_morpion_regressor(_relational_model_args())
    relational = MorpionRelationalEntityTokenConverter(
        max_tokens=128
    ).state_to_tensors(_make_one_step_state())
    prediction = model(relational.token_tensor, relational.relation_triples)

    prediction.square().mean().backward()

    gradient = cast(
        "RelationBiasedEntityTokenTransformerValueNet",
        model.net,
    ).relation_bias.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert float(gradient[1:].abs().sum().item()) > 0.0
    assert float(gradient[0].abs().sum().item()) == 0.0


def test_eager_and_cached_training_steps_reach_relation_bias(tmp_path: Path) -> None:
    """Direct and packed batches should both train the real relational model."""
    rows_path = _build_rows_file(tmp_path, target_values=(0.25, -0.5))
    rows = load_morpion_supervised_rows(rows_path).rows
    dynamics = MorpionDynamics()
    converter = MorpionRelationalEntityTokenConverter(
        dynamics=dynamics,
        max_tokens=128,
    )
    direct_batch = collate_morpion_relational_entity_token_supervised_samples(
        tuple(
            process_morpion_supervised_row_to_relational_entity_token_tensors(
                row,
                dynamics=dynamics,
                converter=converter,
            )
            for row in rows
        )
    )
    direct_model = build_morpion_regressor(_small_relational_model_args())
    direct_before = direct_model.net.relation_bias.weight.detach().clone()  # type: ignore[attr-defined]
    direct_stats = train_regression_batch(
        model=direct_model,
        optimizer=torch.optim.SGD(direct_model.parameters(), lr=0.05),
        criterion=torch.nn.MSELoss(),
        batch=direct_batch,
        device=torch.device("cpu"),
    )

    jsonl_path, _rows = _build_jsonl_rows_file(
        tmp_path,
        target_values=(0.25, -0.5),
    )
    cache = load_or_materialize_relational_entity_token_cache(
        rows_path=jsonl_path,
        row_chunk_size=1,
        max_rows=None,
        entity_max_tokens=128,
    )
    cached_batch = relational_entity_token_cache_batch(
        cache=cache,
        row_indices=(1, 0),
    )
    cached_model = build_morpion_regressor(_small_relational_model_args())
    cached_stats = train_regression_batch(
        model=cached_model,
        optimizer=torch.optim.SGD(cached_model.parameters(), lr=0.05),
        criterion=torch.nn.MSELoss(),
        batch=cached_batch,
        device=torch.device("cpu"),
    )

    assert math.isfinite(direct_stats.loss)
    assert direct_stats.sample_count == 2
    assert not torch.equal(
        direct_before,
        direct_model.net.relation_bias.weight,  # type: ignore[attr-defined]
    )
    assert direct_model.net.relation_bias.weight.grad is not None  # type: ignore[attr-defined]
    assert cached_batch.auxiliary_input_tensors[0].dtype == torch.long
    assert math.isfinite(cached_stats.loss)
    assert cached_stats.sample_count == 2
    assert cached_model.net.relation_bias.weight.grad is not None  # type: ignore[attr-defined]
    assert torch.isfinite(  # type: ignore[attr-defined]
        cached_model.net.relation_bias.weight.grad
    ).all()


def test_relational_eager_training_service_selects_relational_dataset(
    tmp_path: Path,
) -> None:
    """The eager service should train and persist the relational model family."""
    rows_path = _build_rows_file(tmp_path, target_values=(0.25, -0.5))
    output_dir = tmp_path / "relational_eager_bundle"

    model, metrics = train_morpion_regressor(
        _small_relational_training_args(
            dataset_file=rows_path,
            output_dir=output_dir,
            num_epochs=1,
        )
    )
    loaded, loaded_args, manifest = load_morpion_model_bundle(output_dir)

    assert isinstance(model.net, RelationBiasedEntityTokenTransformerValueNet)
    assert isinstance(loaded.net, RelationBiasedEntityTokenTransformerValueNet)
    assert loaded_args.model_kind == MORPION_RELATION_BIASED_ENTITY_TOKEN_MODEL_KIND
    assert manifest.entity_relation_schema == MORPION_ENTITY_RELATION_SCHEMA
    assert math.isfinite(cast("float", metrics["final_loss"]))


def test_relational_cached_streaming_diagnostics_and_dispatch(tmp_path: Path) -> None:
    """Streaming should select relational cache batches for all diagnostics."""
    rows_path, _rows = _build_jsonl_rows_file(
        tmp_path,
        target_values=(-1.0, -0.5, 0.5, 1.0),
    )
    output_dir = tmp_path / "relational_streaming_bundle"
    model, metrics = train_morpion_regressor_streaming(
        MorpionStreamingTrainingArgs(
            training_args=_small_relational_training_args(
                dataset_file=rows_path,
                output_dir=output_dir,
                num_epochs=1,
            ),
            row_chunk_size=2,
        )
    )
    cache = load_or_materialize_relational_entity_token_cache(
        rows_path=rows_path,
        row_chunk_size=2,
        max_rows=None,
        entity_max_tokens=128,
    )
    scale_stats = prediction_scale_stats_for_cached_batches(
        model=model,
        batch_builder=lambda indices: relational_entity_token_cache_batch(
            cache=cache,
            row_indices=indices,
        ),
        row_count=cache.manifest.row_count,
        batch_size=2,
        device=torch.device("cpu"),
    )

    assert metrics["flat_tensor_cache_used"] == "false"
    assert metrics["entity_token_cache_used"] == "false"
    assert metrics["relational_entity_token_cache_used"] == "true"
    assert metrics["num_samples"] == 4.0
    assert math.isfinite(cast("float", metrics["final_loss"]))
    assert math.isfinite(cast("float", metrics["train_quality_mse"]))
    assert scale_stats.count == 4
    assert scale_stats.mean is not None and math.isfinite(scale_stats.mean)
    assert scale_stats.std is not None and math.isfinite(scale_stats.std)


def test_relational_direct_row_diagnostics_use_two_input_adapter(tmp_path: Path) -> None:
    """Direct diagnostic prediction should select relational row conversion."""
    rows_path = _build_rows_file(tmp_path, target_values=(0.25, -0.5))
    rows = load_morpion_supervised_rows(rows_path).rows
    model = build_morpion_regressor(_small_relational_model_args())

    predictions = predict_morpion_rows_for_diagnostics(model, rows)

    assert (
        diagnostic_adapter_kind(MORPION_RELATION_BIASED_ENTITY_TOKEN_MODEL_KIND)
        == "relational_entity_tokens"
    )
    assert len(predictions) == 2
    assert all(math.isfinite(prediction) for prediction in predictions)


def test_relational_bundle_roundtrip_and_runtime_evaluation(tmp_path: Path) -> None:
    """Normal bundles should preserve relation metadata, weights, and inference."""
    torch.manual_seed(0)
    args = _small_relational_model_args()
    model = build_morpion_regressor(args).eval()
    state = _make_one_step_state()
    relational = MorpionRelationalEntityTokenConverter(
        max_tokens=128
    ).state_to_tensors(state)
    with torch.no_grad():
        before = model(relational.token_tensor, relational.relation_triples)
    bundle_dir = tmp_path / "relational_bundle"
    save_morpion_model_bundle(model, bundle_dir, model_args=args)

    loaded_model, loaded_args, manifest = load_morpion_model_bundle(bundle_dir)
    loaded_model.eval()
    with torch.no_grad():
        after = loaded_model(relational.token_tensor, relational.relation_triples)
    evaluator = cast(
        "MorpionRegressorMasterEvaluator",
        load_morpion_evaluator_from_model_bundle(bundle_dir),
    )
    model_inputs = evaluator.input_converter.state_to_model_input_tensors(state)
    value = evaluator.evaluate(state)

    assert isinstance(loaded_model.net, RelationBiasedEntityTokenTransformerValueNet)
    assert loaded_args.entity_relation_schema == MORPION_ENTITY_RELATION_SCHEMA
    assert loaded_args.entity_relation_type_count == MORPION_ENTITY_RELATION_TYPE_COUNT
    assert manifest.input_representation == MORPION_ENTITY_TOKEN_INPUT_REPRESENTATION
    assert manifest.entity_relation_schema == MORPION_ENTITY_RELATION_SCHEMA
    assert manifest.entity_relation_type_count == MORPION_ENTITY_RELATION_TYPE_COUNT
    assert torch.equal(
        loaded_model.net.relation_bias.weight,
        model.net.relation_bias.weight,  # type: ignore[attr-defined]
    )
    torch.testing.assert_close(after, before)
    assert len(model_inputs) == 2
    assert model_inputs[0].shape[1] == MORPION_ENTITY_TOKEN_FEATURE_DIM
    assert model_inputs[1].shape[1] == 3
    assert model_inputs[1].dtype == torch.long
    assert value.certainty is Certainty.ESTIMATE
    assert math.isfinite(value.score)


@pytest.mark.parametrize(
    ("file_name", "field_name", "invalid_value"),
    (
        (MORPION_MANIFEST_FILE_NAME, "entity_relation_schema", "wrong_v1"),
        (MORPION_MANIFEST_FILE_NAME, "entity_relation_schema", None),
        (MORPION_MANIFEST_FILE_NAME, "entity_relation_type_count", 17),
        (MORPION_MANIFEST_FILE_NAME, "entity_relation_type_count", None),
        (MORPION_MANIFEST_FILE_NAME, "input_representation", "handcrafted_features"),
        (MORPION_MANIFEST_FILE_NAME, "input_dim", 24),
        (MORPION_MANIFEST_FILE_NAME, "model_kind", MORPION_ENTITY_TOKEN_MODEL_KIND),
        (MORPION_MODEL_ARGS_FILE_NAME, "entity_relation_schema", "wrong_v1"),
        (MORPION_MODEL_ARGS_FILE_NAME, "entity_relation_schema", None),
        (MORPION_MODEL_ARGS_FILE_NAME, "entity_relation_type_count", 17),
        (MORPION_MODEL_ARGS_FILE_NAME, "entity_relation_type_count", None),
        (MORPION_MODEL_ARGS_FILE_NAME, "entity_use_validity_feature", False),
        (MORPION_MODEL_ARGS_FILE_NAME, "entity_input_feature_dim", 24),
        (MORPION_MODEL_ARGS_FILE_NAME, "model_kind", MORPION_ENTITY_TOKEN_MODEL_KIND),
    ),
)
def test_relational_bundle_rejects_incompatible_metadata(
    tmp_path: Path,
    file_name: str,
    field_name: str,
    invalid_value: object,
) -> None:
    """Explicit bundle metadata should reject every incompatible semantic."""
    bundle_dir = tmp_path / "corrupt_relational_bundle"
    args = _small_relational_model_args()
    save_morpion_model_bundle(
        build_morpion_regressor(args),
        bundle_dir,
        model_args=args,
    )
    path = bundle_dir / file_name
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field_name] = invalid_value
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        load_morpion_model_bundle(bundle_dir)


def test_ordinary_bundle_retains_one_input_and_null_relation_metadata(
    tmp_path: Path,
) -> None:
    """Ordinary token bundles should remain one-input and relation-free."""
    args = MorpionRegressorArgs(
        model_kind=MORPION_ENTITY_TOKEN_MODEL_KIND,
        entity_max_tokens=128,
        entity_d_model=16,
        entity_n_head=4,
        entity_n_layer=1,
        entity_dim_feedforward=32,
    )
    bundle_dir = tmp_path / "ordinary_entity_bundle"
    save_morpion_model_bundle(
        build_morpion_regressor(args),
        bundle_dir,
        model_args=args,
    )

    loaded, _loaded_args, manifest = load_morpion_model_bundle(bundle_dir)
    inputs = MorpionEntityTokenConverter(
        max_tokens=128
    ).state_to_model_input_tensors(_make_one_step_state())

    assert manifest.entity_relation_schema is None
    assert manifest.entity_relation_type_count is None
    assert len(inputs) == 1
    assert torch.isfinite(loaded(*inputs)).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_relational_model_inputs_move_to_cuda_without_dtype_change() -> None:
    """CUDA execution should keep relation triples integral and colocated."""
    model = build_morpion_regressor(_small_relational_model_args()).to("cuda")
    relational = MorpionRelationalEntityTokenConverter(
        max_tokens=128
    ).state_to_tensors(_make_one_step_state())
    tokens = relational.token_tensor.to("cuda")
    relations = relational.relation_triples.to("cuda")

    output = model(tokens, relations)

    assert tokens.device.type == "cuda"
    assert relations.device.type == "cuda"
    assert relations.dtype == torch.long
    assert torch.isfinite(output).all()


def _relational_model_args() -> MorpionRegressorArgs:
    """Return the canonical small relational model arguments."""
    return MorpionRegressorArgs(
        model_kind=MORPION_RELATION_BIASED_ENTITY_TOKEN_MODEL_KIND,
        entity_relation_schema=MORPION_ENTITY_RELATION_SCHEMA,
        entity_relation_type_count=MORPION_ENTITY_RELATION_TYPE_COUNT,
    )


def _small_relational_model_args() -> MorpionRegressorArgs:
    """Return a fast two-layer relational model for integration tests."""
    return MorpionRegressorArgs(
        model_kind=MORPION_RELATION_BIASED_ENTITY_TOKEN_MODEL_KIND,
        entity_max_tokens=128,
        entity_d_model=16,
        entity_n_head=4,
        entity_n_layer=2,
        entity_dim_feedforward=32,
        entity_dropout_ratio=0.0,
        entity_relation_schema=MORPION_ENTITY_RELATION_SCHEMA,
        entity_relation_type_count=MORPION_ENTITY_RELATION_TYPE_COUNT,
    )


def _small_relational_training_args(
    *,
    dataset_file: Path,
    output_dir: Path,
    num_epochs: int,
) -> MorpionTrainingArgs:
    """Return fast relational streaming arguments with exact schema metadata."""
    return MorpionTrainingArgs(
        dataset_file=dataset_file,
        output_dir=output_dir,
        batch_size=2,
        num_epochs=num_epochs,
        learning_rate=1e-3,
        shuffle=False,
        validation_fraction=0.25,
        model_kind=MORPION_RELATION_BIASED_ENTITY_TOKEN_MODEL_KIND,
        entity_max_tokens=128,
        entity_d_model=16,
        entity_n_head=4,
        entity_n_layer=2,
        entity_dim_feedforward=32,
        entity_dropout_ratio=0.0,
        entity_relation_schema=MORPION_ENTITY_RELATION_SCHEMA,
        entity_relation_type_count=MORPION_ENTITY_RELATION_TYPE_COUNT,
        device="cpu",
    )
