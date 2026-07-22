"""Cache-backed fixed-model inference for Morpion relation interventions."""
# ruff: noqa: TRY003

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from coral.neural_networks.models.relation_biased_entity_token_transformer_value_net import (
    RelationBiasedEntityTokenTransformerValueNet,
)

from chipiron.environments.morpion.learning import (
    iter_morpion_supervised_rows_from_path,
    morpion_supervised_rows_source_from_path,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.bundle import (
    load_morpion_model_bundle,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.entity_relations import (
    MORPION_ENTITY_RELATION_SCHEMA,
    MORPION_ENTITY_RELATION_TYPE_COUNT,
    MorpionEntityRelationType,
    is_relational_entity_token_model_kind,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.entity_tokens import (
    is_morpion_entity_token_model_kind,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training.cached_index_schedule import (
    cached_index_schedule,
    index_batches,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.training.relational_entity_token_cache import (
    MorpionRelationalEntityTokenCache,
    load_or_materialize_relational_entity_token_cache,
    relational_entity_token_cache_batch,
)
from chipiron.learning.supervised import move_supervised_batch_to_device
from chipiron.learning.torch_runtime import resolve_torch_device

from .args import (
    InvalidMorpionRelationInterventionInputError,
    MorpionRelationInterventionArgs,
    MorpionRelationInterventionInferenceError,
)
from .definitions import (
    RelationInterventionDefinition,
    all_relation_interventions,
    disable_relation_types,
)

if TYPE_CHECKING:
    from pathlib import Path

    from chipiron.environments.morpion.players.evaluators.neural_networks.bundle import (
        MorpionModelManifest,
    )
    from chipiron.environments.morpion.players.evaluators.neural_networks.model import (
        MorpionRegressor,
        MorpionRegressorArgs,
    )
    from chipiron.learning.supervised import SupervisedBatch

_CACHE_ROW_CHUNK_SIZE = 2_048
_BASELINE_METRIC_TOLERANCE = 1e-4


@dataclass(frozen=True, slots=True)
class SeedInterventionInference:
    """Compact CPU predictions and metadata for one trained seed."""

    seed: int
    relational_bundle: Path
    ordinary_bundle: Path | None
    row_indices: tuple[int, ...]
    targets: tuple[float, ...]
    baseline_predictions: tuple[float, ...]
    intervention_predictions: dict[str, tuple[float, ...]]
    ordinary_predictions: tuple[float, ...] | None
    saved_bundle_metrics: dict[str, object]
    relation_biases: dict[str, object]


@dataclass(frozen=True, slots=True)
class InterventionInferenceSet:
    """Shared validation/cache metadata and all completed seed results."""

    row_indices: tuple[int, ...]
    split_policy: str
    source_row_count: int
    effective_row_count: int
    cache: MorpionRelationalEntityTokenCache
    definitions: tuple[RelationInterventionDefinition, ...]
    requested_device: str
    resolved_device: str
    seeds: tuple[SeedInterventionInference, ...]


def run_relation_intervention_inference(
    args: MorpionRelationInterventionArgs,
) -> InterventionInferenceSet:
    """Load one shared cache and evaluate every fixed model and intervention."""
    _validate_args(args)
    source_row_count = _source_row_count(args.dataset_file)
    effective_row_count = (
        source_row_count
        if args.max_rows is None
        else min(source_row_count, args.max_rows)
    )
    if effective_row_count == 0:
        raise InvalidMorpionRelationInterventionInputError.invalid(
            "dataset is empty after applying max_rows"
        )
    schedule = cached_index_schedule(
        row_count=effective_row_count,
        validation_fraction=args.validation_fraction,
    )
    if not schedule.validation_indices:
        raise InvalidMorpionRelationInterventionInputError.invalid(
            "validation split is empty"
        )
    max_tokens = _validated_shared_max_tokens(args)
    cache = load_or_materialize_relational_entity_token_cache(
        rows_path=args.dataset_file,
        row_chunk_size=_CACHE_ROW_CHUNK_SIZE,
        max_rows=args.max_rows,
        entity_max_tokens=max_tokens,
    )
    if cache.manifest.row_count != effective_row_count:
        raise InvalidMorpionRelationInterventionInputError.invalid(
            "relational cache row count does not match the selected source rows"
        )
    device = resolve_torch_device(args.device)
    definitions = all_relation_interventions()
    seed_results = tuple(
        _run_one_seed(
            seed=bundle.seed,
            relational_bundle=bundle.relational_bundle,
            ordinary_bundle=bundle.ordinary_bundle,
            cache=cache,
            row_indices=schedule.validation_indices,
            definitions=definitions,
            batch_size=args.batch_size,
            device=device,
        )
        for bundle in sorted(args.bundles, key=lambda item: item.seed)
    )
    return InterventionInferenceSet(
        row_indices=schedule.validation_indices,
        split_policy=schedule.split_policy,
        source_row_count=source_row_count,
        effective_row_count=effective_row_count,
        cache=cache,
        definitions=definitions,
        requested_device=args.device,
        resolved_device=str(device),
        seeds=seed_results,
    )


def _run_one_seed(
    *,
    seed: int,
    relational_bundle: Path,
    ordinary_bundle: Path | None,
    cache: MorpionRelationalEntityTokenCache,
    row_indices: tuple[int, ...],
    definitions: tuple[RelationInterventionDefinition, ...],
    batch_size: int,
    device: torch.device,
) -> SeedInterventionInference:
    """Run baseline and all interventions for one seed with cleanup guarantees."""
    model, model_args, manifest = _load_relational_bundle(relational_bundle)
    ordinary_model = None
    if ordinary_bundle is not None:
        ordinary_model, ordinary_args, _ = _load_ordinary_bundle(ordinary_bundle)
        if ordinary_args.entity_max_tokens != model_args.entity_max_tokens:
            raise InvalidMorpionRelationInterventionInputError.invalid(
                f"seed {seed} ordinary and relational max-token settings differ"
            )
    baseline_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    ordinary_chunks: list[torch.Tensor] = []
    intervention_chunks: dict[str, list[torch.Tensor]] = {
        definition.name: [] for definition in definitions
    }
    model = model.to(device)
    if ordinary_model is not None:
        ordinary_model = ordinary_model.to(device)
    try:
        model.eval()
        if ordinary_model is not None:
            ordinary_model.eval()
        with torch.inference_mode():
            for batch_indices in index_batches(row_indices, batch_size=batch_size):
                sample_batch = relational_entity_token_cache_batch(
                    cache=cache,
                    row_indices=batch_indices,
                )
                device_batch = move_supervised_batch_to_device(
                    cast("SupervisedBatch", sample_batch), device
                )
                tokens, original_relations = device_batch.get_model_input_tensors()
                baseline = model(tokens, original_relations).detach().cpu().reshape(-1)
                targets = device_batch.get_target_value().detach().cpu().reshape(-1)
                _validate_batch_output(seed, batch_indices, baseline, targets)
                baseline_chunks.append(baseline)
                target_chunks.append(targets)
                if ordinary_model is not None:
                    ordinary = ordinary_model(tokens).detach().cpu().reshape(-1)
                    _validate_prediction_tensor(seed, "ordinary", ordinary)
                    ordinary_chunks.append(ordinary)
                for definition in definitions:
                    intervened_relations = disable_relation_types(
                        original_relations,
                        definition.relation_type_ids,
                    )
                    prediction = (
                        model(tokens, intervened_relations).detach().cpu().reshape(-1)
                    )
                    _validate_prediction_tensor(seed, definition.name, prediction)
                    intervention_chunks[definition.name].append(prediction)
                del device_batch, tokens, original_relations
    finally:
        model.to("cpu")
        if ordinary_model is not None:
            ordinary_model.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    baseline_values = _tensor_chunks_to_tuple(baseline_chunks)
    target_values = _tensor_chunks_to_tuple(target_chunks)
    saved_metrics = _saved_baseline_metrics(
        manifest,
        validation_count=len(row_indices),
        baseline=baseline_values,
        targets=target_values,
    )
    return SeedInterventionInference(
        seed=seed,
        relational_bundle=relational_bundle.resolve(),
        ordinary_bundle=None if ordinary_bundle is None else ordinary_bundle.resolve(),
        row_indices=row_indices,
        targets=target_values,
        baseline_predictions=baseline_values,
        intervention_predictions={
            name: _tensor_chunks_to_tuple(chunks)
            for name, chunks in intervention_chunks.items()
        },
        ordinary_predictions=(
            None if ordinary_model is None else _tensor_chunks_to_tuple(ordinary_chunks)
        ),
        saved_bundle_metrics=saved_metrics,
        relation_biases=_relation_bias_payload(model),
    )


def _validate_args(args: MorpionRelationInterventionArgs) -> None:
    """Validate paths, seeds, and numeric arguments before cache/model work."""
    if not args.dataset_file.is_file():
        raise InvalidMorpionRelationInterventionInputError.invalid(
            f"missing dataset file {args.dataset_file!s}"
        )
    if not args.bundles:
        raise InvalidMorpionRelationInterventionInputError.invalid(
            "at least one relational bundle is required"
        )
    seeds = tuple(bundle.seed for bundle in args.bundles)
    if len(set(seeds)) != len(seeds):
        raise InvalidMorpionRelationInterventionInputError.invalid(
            "duplicate relational seed IDs"
        )
    ordinary_presence = tuple(
        bundle.ordinary_bundle is not None for bundle in args.bundles
    )
    if any(ordinary_presence) and not all(ordinary_presence):
        raise InvalidMorpionRelationInterventionInputError.invalid(
            "ordinary bundle seed set must exactly match relational seeds"
        )
    for bundle in args.bundles:
        if isinstance(bundle.seed, bool) or bundle.seed < 0:
            raise InvalidMorpionRelationInterventionInputError.invalid(
                f"invalid seed {bundle.seed!r}"
            )
        if not bundle.relational_bundle.is_dir():
            raise InvalidMorpionRelationInterventionInputError.invalid(
                f"missing relational bundle {bundle.relational_bundle!s}"
            )
        if bundle.ordinary_bundle is not None and not bundle.ordinary_bundle.is_dir():
            raise InvalidMorpionRelationInterventionInputError.invalid(
                f"missing ordinary bundle {bundle.ordinary_bundle!s}"
            )
    if args.max_rows is not None and (
        isinstance(args.max_rows, bool) or args.max_rows <= 0
    ):
        raise InvalidMorpionRelationInterventionInputError.invalid(
            "max_rows must be a positive integer or null"
        )
    if not 0.0 < args.validation_fraction < 1.0:
        raise InvalidMorpionRelationInterventionInputError.invalid(
            "validation_fraction must be strictly between zero and one"
        )
    if isinstance(args.batch_size, bool) or args.batch_size <= 0:
        raise InvalidMorpionRelationInterventionInputError.invalid(
            "batch_size must be positive"
        )
    if (
        args.structural_analysis_dir is not None
        and not args.structural_analysis_dir.is_dir()
    ):
        raise InvalidMorpionRelationInterventionInputError.invalid(
            f"missing structural analysis directory {args.structural_analysis_dir!s}"
        )


def _validated_shared_max_tokens(args: MorpionRelationInterventionArgs) -> int:
    """Validate relational bundle kinds and return their shared cache width."""
    widths: set[int] = set()
    for bundle in args.bundles:
        _model, model_args, _manifest = _load_relational_bundle(
            bundle.relational_bundle
        )
        widths.add(model_args.entity_max_tokens)
    if len(widths) != 1:
        raise InvalidMorpionRelationInterventionInputError.invalid(
            "relational bundles require different entity_max_tokens caches"
        )
    return next(iter(widths))


def _load_relational_bundle(
    path: Path,
) -> tuple[MorpionRegressor, MorpionRegressorArgs, MorpionModelManifest]:
    """Load and validate one relational model bundle."""
    try:
        model, model_args, manifest = load_morpion_model_bundle(path)
    except Exception as exc:
        raise InvalidMorpionRelationInterventionInputError.invalid(
            f"could not load relational bundle {path!s}"
        ) from exc
    if (
        not is_relational_entity_token_model_kind(model_args.model_kind)
        or model_args.entity_relation_schema != MORPION_ENTITY_RELATION_SCHEMA
        or model_args.entity_relation_type_count != MORPION_ENTITY_RELATION_TYPE_COUNT
    ):
        raise InvalidMorpionRelationInterventionInputError.invalid(
            f"bundle is not a compatible relational entity-token model: {path!s}"
        )
    return model, model_args, manifest


def _load_ordinary_bundle(
    path: Path,
) -> tuple[MorpionRegressor, MorpionRegressorArgs, MorpionModelManifest]:
    """Load and validate one ordinary entity-token context model."""
    try:
        model, model_args, manifest = load_morpion_model_bundle(path)
    except Exception as exc:
        raise InvalidMorpionRelationInterventionInputError.invalid(
            f"could not load ordinary bundle {path!s}"
        ) from exc
    if not is_morpion_entity_token_model_kind(model_args.model_kind):
        raise InvalidMorpionRelationInterventionInputError.invalid(
            f"bundle is not an ordinary entity-token model: {path!s}"
        )
    return model, model_args, manifest


def _source_row_count(dataset_file: Path) -> int:
    """Count source rows without retaining decoded states."""
    source = morpion_supervised_rows_source_from_path(dataset_file)
    if source.row_count is not None:
        return source.row_count
    return sum(1 for _ in iter_morpion_supervised_rows_from_path(dataset_file))


def _validate_batch_output(
    seed: int,
    row_indices: tuple[int, ...],
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    """Validate baseline and target count/finiteness."""
    if predictions.numel() != len(row_indices) or targets.numel() != len(row_indices):
        raise MorpionRelationInterventionInferenceError.invalid(
            f"seed {seed} produced a batch count mismatch"
        )
    _validate_prediction_tensor(seed, "baseline", predictions)
    if not bool(torch.isfinite(targets).all()):
        raise MorpionRelationInterventionInferenceError.invalid(
            f"seed {seed} encountered non-finite targets"
        )


def _validate_prediction_tensor(
    seed: int, name: str, predictions: torch.Tensor
) -> None:
    """Reject non-finite fixed-model predictions."""
    if not bool(torch.isfinite(predictions).all()):
        raise MorpionRelationInterventionInferenceError.invalid(
            f"seed {seed} intervention {name!r} produced non-finite predictions"
        )


def _tensor_chunks_to_tuple(chunks: list[torch.Tensor]) -> tuple[float, ...]:
    """Flatten detached CPU chunks into stable Python floats."""
    return tuple(float(value) for chunk in chunks for value in chunk.tolist())


def _saved_baseline_metrics(
    manifest: MorpionModelManifest,
    *,
    validation_count: int,
    baseline: tuple[float, ...],
    targets: tuple[float, ...],
) -> dict[str, object]:
    """Compare only compatible full-validation saved metrics."""
    quality = manifest.metadata.get("regression_quality")
    validation: dict[str, object] | None = None
    if isinstance(quality, dict) and isinstance(quality.get("validation"), dict):
        validation = cast("dict[str, object]", quality["validation"])
    saved_count = None if validation is None else validation.get("count")
    saved_mse = None if validation is None else validation.get("mse")
    recomputed_mse = math.fsum(
        (prediction - target) ** 2
        for prediction, target in zip(baseline, targets, strict=True)
    ) / len(targets)
    compatible = (
        isinstance(saved_count, int | float)
        and int(saved_count) == validation_count
        and isinstance(saved_mse, int | float)
    )
    difference = None
    if compatible:
        difference = recomputed_mse - float(cast("int | float", saved_mse))
        if abs(difference) > _BASELINE_METRIC_TOLERANCE:
            raise MorpionRelationInterventionInferenceError.invalid(
                "recomputed baseline MSE differs materially from compatible saved metrics"
            )
    return {
        "source": "manifest.metadata.regression_quality.validation",
        "compatible_with_full_validation_split": compatible,
        "saved_count": saved_count,
        "saved_mse": saved_mse,
        "recomputed_baseline_mse": recomputed_mse,
        "baseline_mse_difference": difference,
        "comparison_tolerance": _BASELINE_METRIC_TOLERANCE,
        "incompatibility_reason": (
            None
            if compatible
            else "saved metric is absent or sampled on a different row count"
        ),
    }


def _relation_bias_payload(model: MorpionRegressor) -> dict[str, object]:
    """Copy the typed learned relation-bias table to JSON-safe CPU values."""
    if not isinstance(model.net, RelationBiasedEntityTokenTransformerValueNet):
        raise MorpionRelationInterventionInferenceError.invalid(
            "relational model has an unexpected typed network implementation"
        )
    weights = model.net.relation_bias.weight.detach().cpu()
    return {
        relation_type.name: [
            float(value) for value in weights[int(relation_type)].tolist()
        ]
        for relation_type in MorpionEntityRelationType
    }


__all__ = [
    "InterventionInferenceSet",
    "SeedInterventionInference",
    "run_relation_intervention_inference",
]
