"""Paired validation prediction diagnostics for saved Morpion evaluators.

Residuals in this module are always defined as ``prediction - target``.
The per-row oracle is an analysis upper bound: it uses the target to choose a
prediction and therefore cannot be used as an inference method.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Final, TextIO, cast

import torch

from chipiron.environments.morpion.learning import (
    iter_morpion_supervised_rows_from_path,
    morpion_supervised_rows_source_from_path,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.bundle import (
    load_morpion_model_bundle,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.entity_relations import (
    is_relational_entity_token_model_kind,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.entity_tokens import (
    is_morpion_entity_token_model_kind,
)
from chipiron.learning.supervised import (
    TensorSupervisedBatch,
    move_supervised_batch_to_device,
    regression_quality_stats,
)
from chipiron.learning.torch_runtime import resolve_torch_device

from .cached_index_schedule import cached_index_schedule, index_batches
from .entity_token_cache import (
    MorpionEntityTokenCache,
    entity_token_cache_batch,
    load_or_materialize_entity_token_cache,
)
from .flat_tensor_cache import (
    FlatTensorCache,
    flat_cache_batch,
    is_flat_morpion_training_model_kind,
    load_or_materialize_flat_tensor_cache,
)
from .relational_entity_token_cache import (
    MorpionRelationalEntityTokenCache,
    load_or_materialize_relational_entity_token_cache,
    relational_entity_token_cache_batch,
)

if TYPE_CHECKING:
    from chipiron.environments.morpion.players.evaluators.neural_networks.model import (
        MorpionRegressor,
        MorpionRegressorArgs,
    )
    from chipiron.learning.supervised import SupervisedBatch

MORPION_COMPARISON_SCHEMA: Final[str] = "morpion_evaluator_comparison_v1"
PAIRWISE_ERROR_TIE_TOLERANCE: Final[float] = 1e-12
WORST_ROW_LIMIT: Final[int] = 100
_CACHE_ROW_CHUNK_SIZE: Final[int] = 2_048
_OUTPUT_FILE_NAMES: Final[tuple[str, ...]] = (
    "summary.json",
    "predictions.jsonl",
    "predictions.csv",
    "worst_rows.json",
)


class MorpionComparisonDiagnosticsError(RuntimeError):
    """Base class for stable comparison-diagnostics failures."""


class InvalidMorpionComparisonInputError(MorpionComparisonDiagnosticsError):
    """Raised when requested comparison inputs are invalid or ambiguous."""

    @classmethod
    def missing_dataset(cls, path: Path) -> InvalidMorpionComparisonInputError:
        """Return an error for a missing dataset file."""
        return cls(f"Morpion comparison dataset file does not exist: {path!s}.")

    @classmethod
    def too_few_bundles(cls) -> InvalidMorpionComparisonInputError:
        """Return an error for fewer than two bundles."""
        return cls("Morpion comparison diagnostics require at least two bundles.")

    @classmethod
    def empty_evaluator_name(cls) -> InvalidMorpionComparisonInputError:
        """Return an error for an empty evaluator name."""
        return cls("Morpion comparison evaluator names must be non-empty.")

    @classmethod
    def duplicate_evaluator_name(
        cls,
        name: str,
    ) -> InvalidMorpionComparisonInputError:
        """Return an error for a duplicate evaluator name."""
        return cls(f"Duplicate Morpion comparison evaluator name: {name!r}.")

    @classmethod
    def missing_bundle(
        cls,
        name: str,
        path: Path,
    ) -> InvalidMorpionComparisonInputError:
        """Return an error for a missing bundle directory."""
        return cls(f"Bundle directory for evaluator {name!r} does not exist: {path!s}.")

    @classmethod
    def invalid_max_rows(cls) -> InvalidMorpionComparisonInputError:
        """Return an error for an invalid row limit."""
        return cls("max_rows must be a positive integer or None.")

    @classmethod
    def invalid_validation_fraction(cls) -> InvalidMorpionComparisonInputError:
        """Return an error for an invalid validation fraction."""
        return cls("validation_fraction must be greater than 0 and less than 1.")

    @classmethod
    def invalid_batch_size(cls) -> InvalidMorpionComparisonInputError:
        """Return an error for an invalid inference batch size."""
        return cls("batch_size must be a positive integer.")

    @classmethod
    def empty_dataset(cls) -> InvalidMorpionComparisonInputError:
        """Return an error for an empty effective dataset."""
        return cls("The effective Morpion comparison dataset contains no rows.")

    @classmethod
    def empty_validation_split(cls) -> InvalidMorpionComparisonInputError:
        """Return an error for a validation split with no rows."""
        return cls("The selected streaming validation split contains no rows.")

    @classmethod
    def csv_prefix_collision(
        cls,
        first_name: str,
        second_name: str,
        prefix: str,
    ) -> InvalidMorpionComparisonInputError:
        """Return an error for ambiguous sanitized CSV prefixes."""
        return cls(
            "Evaluator names produce the same CSV prefix "
            f"{prefix!r}: {first_name!r} and {second_name!r}."
        )

    @classmethod
    def invalid_output_directory(
        cls,
        path: Path,
    ) -> InvalidMorpionComparisonInputError:
        """Return an error when the requested output path is not a directory."""
        return cls(f"Morpion comparison output path is not a directory: {path!s}.")


class MorpionComparisonBundleLoadError(MorpionComparisonDiagnosticsError):
    """Raised when a requested evaluator bundle cannot be loaded."""

    def __init__(self, name: str, path: Path) -> None:
        """Initialize one canonical bundle-load wrapper error."""
        super().__init__(f"Failed to load Morpion bundle {name!r} from {path!s}.")


class MorpionComparisonEvaluationError(MorpionComparisonDiagnosticsError):
    """Raised when evaluator outputs cannot form a paired comparison."""

    @classmethod
    def cache_row_count_mismatch(
        cls,
        *,
        representation: str,
        expected: int,
        actual: int,
    ) -> MorpionComparisonEvaluationError:
        """Return an error for a cache with a mismatched effective row count."""
        return cls(
            f"{representation} cache contains {actual} rows; expected {expected}."
        )

    @classmethod
    def prediction_count_mismatch(
        cls,
        *,
        evaluator_name: str,
        expected: int,
        actual: int,
    ) -> MorpionComparisonEvaluationError:
        """Return an error for a non-scalar-per-row model output."""
        return cls(
            f"Evaluator {evaluator_name!r} returned {actual} prediction values "
            f"for a batch of {expected} rows; expected exactly one scalar per row."
        )

    @classmethod
    def target_count_mismatch(
        cls,
        *,
        evaluator_name: str,
        expected: int,
        actual: int,
    ) -> MorpionComparisonEvaluationError:
        """Return an error for a malformed target batch."""
        return cls(
            f"Evaluator {evaluator_name!r} received {actual} targets for a batch "
            f"of {expected} rows."
        )

    @classmethod
    def non_finite_prediction(
        cls,
        *,
        evaluator_name: str,
        row_index: int,
    ) -> MorpionComparisonEvaluationError:
        """Return an error for a NaN or infinite prediction."""
        return cls(
            f"Evaluator {evaluator_name!r} produced a non-finite prediction at "
            f"dataset row {row_index}."
        )

    @classmethod
    def non_finite_target(
        cls,
        *,
        evaluator_name: str,
        row_index: int,
    ) -> MorpionComparisonEvaluationError:
        """Return an error for a NaN or infinite target."""
        return cls(
            f"Evaluator {evaluator_name!r} received a non-finite target at "
            f"dataset row {row_index}."
        )

    @classmethod
    def target_mismatch(
        cls,
        *,
        evaluator_name: str,
        row_index: int,
        expected: float,
        actual: float,
    ) -> MorpionComparisonEvaluationError:
        """Return an error when adapters disagree about one target."""
        return cls(
            f"Target mismatch for evaluator {evaluator_name!r} at dataset row "
            f"{row_index}: expected {expected!r}, got {actual!r}."
        )

    @classmethod
    def unsupported_model_kind(
        cls,
        model_kind: str,
    ) -> MorpionComparisonEvaluationError:
        """Return an error for a model kind without a diagnostics adapter."""
        return cls(f"Unsupported Morpion comparison model kind: {model_kind!r}.")


class _MalformedBundleArgumentError(argparse.ArgumentTypeError):
    """Raised when one repeatable CLI bundle value is malformed."""

    @classmethod
    def invalid_separator(cls) -> _MalformedBundleArgumentError:
        """Return the exact-separator parse error."""
        return cls("--bundle must use exactly NAME=PATH")

    @classmethod
    def empty_component(cls) -> _MalformedBundleArgumentError:
        """Return the non-empty-component parse error."""
        return cls("--bundle must use non-empty NAME=PATH")


@dataclass(frozen=True, slots=True)
class MorpionComparisonBundle:
    """A stable evaluator name paired with one saved bundle directory."""

    evaluator_name: str
    bundle_path: Path


@dataclass(frozen=True, slots=True)
class MorpionComparisonDiagnosticsArgs:
    """Inputs controlling one paired validation diagnostic run."""

    dataset_file: Path
    bundles: tuple[MorpionComparisonBundle, ...]
    output_dir: Path
    max_rows: int | None = None
    validation_fraction: float = 0.2
    batch_size: int = 64
    device: str = "auto"


@dataclass(frozen=True, slots=True)
class MorpionComparisonDiagnostics:
    """In-memory paired predictions and aggregate diagnostic artifacts."""

    schema: str
    dataset_file: Path
    source_row_count: int
    effective_row_count: int
    validation_fraction: float
    split_policy: str
    validation_row_indices: tuple[int, ...]
    requested_device: str
    resolved_device: str
    targets: tuple[float, ...]
    evaluator_bundle_paths: dict[str, Path]
    evaluator_model_kinds: dict[str, str]
    csv_prefixes: dict[str, str]
    predictions: dict[str, tuple[float, ...]]
    summary: dict[str, object]
    worst_rows: dict[str, object]


@dataclass(frozen=True, slots=True)
class _LoadedEvaluator:
    name: str
    bundle_path: Path
    model: MorpionRegressor
    model_args: MorpionRegressorArgs


@dataclass(frozen=True, slots=True)
class _EvaluationResult:
    predictions: tuple[float, ...]
    targets: tuple[float, ...]


type _BatchBuilder = Callable[[tuple[int, ...]], TensorSupervisedBatch]


def build_morpion_comparison_diagnostics(
    args: MorpionComparisonDiagnosticsArgs,
) -> MorpionComparisonDiagnostics:
    """Evaluate saved bundles on exactly the same cached validation rows."""
    _validate_args(args)
    dataset_file = args.dataset_file.resolve()
    loaded_evaluators = _load_evaluators(args.bundles)
    csv_prefixes = _csv_prefixes(tuple(item.name for item in loaded_evaluators))
    source_row_count = _source_row_count(dataset_file)
    effective_row_count = (
        source_row_count
        if args.max_rows is None
        else min(source_row_count, args.max_rows)
    )
    if effective_row_count == 0:
        raise InvalidMorpionComparisonInputError.empty_dataset()
    schedule = cached_index_schedule(
        row_count=effective_row_count,
        validation_fraction=args.validation_fraction,
    )
    validation_indices = schedule.validation_indices
    if not validation_indices:
        raise InvalidMorpionComparisonInputError.empty_validation_split()
    device = resolve_torch_device(args.device)

    prediction_map: dict[str, tuple[float, ...]] = {}
    common_targets: tuple[float, ...] | None = None
    grouped_evaluators = _group_evaluators(loaded_evaluators)
    for representation in ("flat", "entity_token", "relational_entity_token"):
        for evaluator, batch_builder in _representation_batch_builders(
            representation=representation,
            evaluators=grouped_evaluators[representation],
            dataset_file=dataset_file,
            max_rows=args.max_rows,
            expected_row_count=effective_row_count,
        ):
            result = _evaluate_one_evaluator(
                evaluator=evaluator,
                batch_builder=batch_builder,
                row_indices=validation_indices,
                batch_size=args.batch_size,
                device=device,
            )
            common_targets = _merge_targets(
                expected=common_targets,
                actual=result.targets,
                evaluator_name=evaluator.name,
                row_indices=validation_indices,
            )
            prediction_map[evaluator.name] = result.predictions
    if common_targets is None:
        raise InvalidMorpionComparisonInputError.too_few_bundles()

    ordered_names = tuple(sorted(prediction_map))
    prediction_map = {name: prediction_map[name] for name in ordered_names}
    bundle_paths = {
        item.name: item.bundle_path.resolve()
        for item in sorted(loaded_evaluators, key=lambda evaluator: evaluator.name)
    }
    model_kinds = {
        item.name: item.model_args.model_kind
        for item in sorted(loaded_evaluators, key=lambda evaluator: evaluator.name)
    }
    evaluator_metrics = {
        name: _regression_metrics(prediction_map[name], common_targets)
        for name in ordered_names
    }
    pairwise = _pairwise_metrics(
        evaluator_names=ordered_names,
        predictions=prediction_map,
        targets=common_targets,
    )
    ensembles = _ensemble_metrics(
        evaluator_names=ordered_names,
        predictions=prediction_map,
        targets=common_targets,
    )
    worst_rows = _worst_rows_artifact(
        evaluator_names=ordered_names,
        row_indices=validation_indices,
        targets=common_targets,
        predictions=prediction_map,
    )
    error_tail = _error_tail_summary(
        evaluator_names=ordered_names,
        evaluator_metrics=evaluator_metrics,
        worst_rows=worst_rows,
    )
    dataset_summary: dict[str, object] = {
        "source_dataset_path": os.fspath(dataset_file),
        "source_dataset_row_count": source_row_count,
        "effective_row_count": effective_row_count,
        "validation_fraction": args.validation_fraction,
        "split_policy_name": schedule.split_policy,
        "validation_row_count": len(validation_indices),
        "validation_row_indices": list(validation_indices),
        "requested_device": args.device,
        "resolved_device": str(device),
    }
    evaluator_summary: dict[str, object] = {}
    for name in ordered_names:
        evaluator_summary[name] = {
            "bundle_path": os.fspath(bundle_paths[name]),
            "model_kind": model_kinds[name],
            "csv_prefix": csv_prefixes[name],
            **evaluator_metrics[name],
        }
    summary: dict[str, object] = {
        "schema": MORPION_COMPARISON_SCHEMA,
        "residual_convention": "prediction_minus_target",
        "dataset": dataset_summary,
        "evaluators": evaluator_summary,
        "pairwise": pairwise,
        "ensembles": ensembles,
        "error_tail": error_tail,
    }
    return MorpionComparisonDiagnostics(
        schema=MORPION_COMPARISON_SCHEMA,
        dataset_file=dataset_file,
        source_row_count=source_row_count,
        effective_row_count=effective_row_count,
        validation_fraction=args.validation_fraction,
        split_policy=schedule.split_policy,
        validation_row_indices=validation_indices,
        requested_device=args.device,
        resolved_device=str(device),
        targets=common_targets,
        evaluator_bundle_paths=bundle_paths,
        evaluator_model_kinds=model_kinds,
        csv_prefixes=csv_prefixes,
        predictions=prediction_map,
        summary=summary,
        worst_rows=worst_rows,
    )


def save_morpion_comparison_diagnostics(
    diagnostics: MorpionComparisonDiagnostics,
    output_dir: Path,
) -> None:
    """Write all comparison artifacts from a complete in-memory result."""
    target_dir = output_dir.resolve()
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{target_dir.name}.staging.",
            dir=target_dir.parent,
        )
    )
    try:
        _write_json(staging_dir / "summary.json", diagnostics.summary)
        _write_predictions_jsonl(staging_dir / "predictions.jsonl", diagnostics)
        _write_predictions_csv(staging_dir / "predictions.csv", diagnostics)
        _write_json(staging_dir / "worst_rows.json", diagnostics.worst_rows)
        if not target_dir.exists():
            staging_dir.replace(target_dir)
            return
        if not target_dir.is_dir():
            raise InvalidMorpionComparisonInputError.invalid_output_directory(
                target_dir
            )
        for file_name in _OUTPUT_FILE_NAMES:
            (staging_dir / file_name).replace(target_dir / file_name)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


def _validate_args(args: MorpionComparisonDiagnosticsArgs) -> None:
    """Validate comparison inputs before loading caches or models."""
    if not args.dataset_file.is_file():
        raise InvalidMorpionComparisonInputError.missing_dataset(args.dataset_file)
    if len(args.bundles) < 2:
        raise InvalidMorpionComparisonInputError.too_few_bundles()
    seen_names: set[str] = set()
    for bundle in args.bundles:
        if not bundle.evaluator_name.strip():
            raise InvalidMorpionComparisonInputError.empty_evaluator_name()
        if bundle.evaluator_name in seen_names:
            raise InvalidMorpionComparisonInputError.duplicate_evaluator_name(
                bundle.evaluator_name
            )
        seen_names.add(bundle.evaluator_name)
        if not bundle.bundle_path.is_dir():
            raise InvalidMorpionComparisonInputError.missing_bundle(
                bundle.evaluator_name,
                bundle.bundle_path,
            )
    if args.max_rows is not None and (
        isinstance(args.max_rows, bool) or args.max_rows <= 0
    ):
        raise InvalidMorpionComparisonInputError.invalid_max_rows()
    if not 0.0 < args.validation_fraction < 1.0:
        raise InvalidMorpionComparisonInputError.invalid_validation_fraction()
    if isinstance(args.batch_size, bool) or args.batch_size <= 0:
        raise InvalidMorpionComparisonInputError.invalid_batch_size()


def _load_evaluators(
    bundles: tuple[MorpionComparisonBundle, ...],
) -> tuple[_LoadedEvaluator, ...]:
    """Load and validate every requested bundle through the canonical loader."""
    loaded: list[_LoadedEvaluator] = []
    for bundle in sorted(bundles, key=lambda item: item.evaluator_name):
        try:
            model, model_args, _ = load_morpion_model_bundle(bundle.bundle_path)
        except Exception as exc:
            raise MorpionComparisonBundleLoadError(
                bundle.evaluator_name,
                bundle.bundle_path,
            ) from exc
        loaded.append(
            _LoadedEvaluator(
                name=bundle.evaluator_name,
                bundle_path=bundle.bundle_path,
                model=model,
                model_args=model_args,
            )
        )
    return tuple(loaded)


def _source_row_count(dataset_file: Path) -> int:
    """Return the full source row count without retaining decoded rows."""
    source = morpion_supervised_rows_source_from_path(dataset_file)
    if source.row_count is not None:
        return source.row_count
    return sum(1 for _ in iter_morpion_supervised_rows_from_path(dataset_file))


def _group_evaluators(
    evaluators: tuple[_LoadedEvaluator, ...],
) -> dict[str, tuple[_LoadedEvaluator, ...]]:
    """Group evaluators by the one supported cached input representation."""
    grouped: dict[str, list[_LoadedEvaluator]] = {
        "flat": [],
        "entity_token": [],
        "relational_entity_token": [],
    }
    for evaluator in evaluators:
        model_kind = evaluator.model_args.model_kind
        if is_relational_entity_token_model_kind(model_kind):
            grouped["relational_entity_token"].append(evaluator)
        elif is_morpion_entity_token_model_kind(model_kind):
            grouped["entity_token"].append(evaluator)
        elif is_flat_morpion_training_model_kind(model_kind):
            grouped["flat"].append(evaluator)
        else:
            raise MorpionComparisonEvaluationError.unsupported_model_kind(model_kind)
    return {key: tuple(value) for key, value in grouped.items()}


def _representation_batch_builders(
    *,
    representation: str,
    evaluators: tuple[_LoadedEvaluator, ...],
    dataset_file: Path,
    max_rows: int | None,
    expected_row_count: int,
) -> tuple[tuple[_LoadedEvaluator, _BatchBuilder], ...]:
    """Return one adapter-specific batch builder for each evaluator."""
    if not evaluators:
        return ()
    if representation == "flat":
        cache = load_or_materialize_flat_tensor_cache(
            rows_path=dataset_file,
            row_chunk_size=_CACHE_ROW_CHUNK_SIZE,
            max_rows=max_rows,
        )
        _validate_cache_row_count(
            representation=representation,
            expected=expected_row_count,
            actual=cache.manifest.row_count,
        )
        return tuple(
            (
                evaluator,
                _flat_batch_builder(cache, evaluator.model_args.feature_names),
            )
            for evaluator in evaluators
        )
    builders: list[tuple[_LoadedEvaluator, _BatchBuilder]] = []
    caches: dict[int, MorpionEntityTokenCache | MorpionRelationalEntityTokenCache] = {}
    for evaluator in evaluators:
        max_tokens = evaluator.model_args.entity_max_tokens
        cache = caches.get(max_tokens)
        if cache is None:
            if representation == "entity_token":
                cache = load_or_materialize_entity_token_cache(
                    rows_path=dataset_file,
                    row_chunk_size=_CACHE_ROW_CHUNK_SIZE,
                    max_rows=max_rows,
                    entity_max_tokens=max_tokens,
                )
            else:
                cache = load_or_materialize_relational_entity_token_cache(
                    rows_path=dataset_file,
                    row_chunk_size=_CACHE_ROW_CHUNK_SIZE,
                    max_rows=max_rows,
                    entity_max_tokens=max_tokens,
                )
            caches[max_tokens] = cache
        _validate_cache_row_count(
            representation=representation,
            expected=expected_row_count,
            actual=cache.manifest.row_count,
        )
        if representation == "entity_token":
            builders.append((
                evaluator,
                _entity_batch_builder(cast("MorpionEntityTokenCache", cache)),
            ))
        else:
            builders.append((
                evaluator,
                _relational_batch_builder(
                    cast("MorpionRelationalEntityTokenCache", cache)
                ),
            ))
    return tuple(builders)


def _validate_cache_row_count(
    *,
    representation: str,
    expected: int,
    actual: int,
) -> None:
    """Reject a stale or inconsistent cache row count."""
    if actual != expected:
        raise MorpionComparisonEvaluationError.cache_row_count_mismatch(
            representation=representation,
            expected=expected,
            actual=actual,
        )


def _flat_batch_builder(
    cache: FlatTensorCache,
    feature_names: tuple[str, ...],
) -> _BatchBuilder:
    """Build one flat-feature adapter closure."""

    def build(row_indices: tuple[int, ...]) -> TensorSupervisedBatch:
        return flat_cache_batch(
            cache=cache,
            row_indices=row_indices,
            requested_feature_names=feature_names,
        )

    return build


def _entity_batch_builder(cache: MorpionEntityTokenCache) -> _BatchBuilder:
    """Build one ordinary entity-token adapter closure."""

    def build(row_indices: tuple[int, ...]) -> TensorSupervisedBatch:
        return entity_token_cache_batch(cache=cache, row_indices=row_indices)

    return build


def _relational_batch_builder(
    cache: MorpionRelationalEntityTokenCache,
) -> _BatchBuilder:
    """Build one relational entity-token adapter closure."""

    def build(row_indices: tuple[int, ...]) -> TensorSupervisedBatch:
        return relational_entity_token_cache_batch(
            cache=cache,
            row_indices=row_indices,
        )

    return build


def _evaluate_one_evaluator(
    *,
    evaluator: _LoadedEvaluator,
    batch_builder: _BatchBuilder,
    row_indices: tuple[int, ...],
    batch_size: int,
    device: torch.device,
) -> _EvaluationResult:
    """Run one model without retaining input batches on the accelerator."""
    predictions: list[float] = []
    targets: list[float] = []
    model = evaluator.model.to(device)
    model.eval()
    with torch.no_grad():
        for batch_indices in index_batches(row_indices, batch_size=batch_size):
            sample_batch = batch_builder(batch_indices)
            device_batch = move_supervised_batch_to_device(
                cast("SupervisedBatch", sample_batch),
                device,
            )
            batch_output = model(*device_batch.get_model_input_tensors())
            flat_output = batch_output.detach().cpu().reshape(-1)
            flat_targets = device_batch.get_target_value().detach().cpu().reshape(-1)
            if flat_output.numel() != len(batch_indices):
                raise MorpionComparisonEvaluationError.prediction_count_mismatch(
                    evaluator_name=evaluator.name,
                    expected=len(batch_indices),
                    actual=int(flat_output.numel()),
                )
            if flat_targets.numel() != len(batch_indices):
                raise MorpionComparisonEvaluationError.target_count_mismatch(
                    evaluator_name=evaluator.name,
                    expected=len(batch_indices),
                    actual=int(flat_targets.numel()),
                )
            for position, row_index in enumerate(batch_indices):
                prediction = float(flat_output[position].item())
                target = float(flat_targets[position].item())
                if not math.isfinite(prediction):
                    raise MorpionComparisonEvaluationError.non_finite_prediction(
                        evaluator_name=evaluator.name,
                        row_index=row_index,
                    )
                if not math.isfinite(target):
                    raise MorpionComparisonEvaluationError.non_finite_target(
                        evaluator_name=evaluator.name,
                        row_index=row_index,
                    )
                predictions.append(prediction)
                targets.append(target)
    model.to("cpu")
    return _EvaluationResult(
        predictions=tuple(predictions),
        targets=tuple(targets),
    )


def _merge_targets(
    *,
    expected: tuple[float, ...] | None,
    actual: tuple[float, ...],
    evaluator_name: str,
    row_indices: tuple[int, ...],
) -> tuple[float, ...]:
    """Require exact ordered target agreement across every input adapter."""
    if expected is None:
        return actual
    if len(actual) != len(expected):
        raise MorpionComparisonEvaluationError.target_count_mismatch(
            evaluator_name=evaluator_name,
            expected=len(expected),
            actual=len(actual),
        )
    for position, (expected_value, actual_value) in enumerate(
        zip(expected, actual, strict=True)
    ):
        if expected_value != actual_value:
            raise MorpionComparisonEvaluationError.target_mismatch(
                evaluator_name=evaluator_name,
                row_index=row_indices[position],
                expected=expected_value,
                actual=actual_value,
            )
    return expected


def _regression_metrics(
    predictions: tuple[float, ...],
    targets: tuple[float, ...],
) -> dict[str, object]:
    """Return the required finite scalar-regression metrics and error tails."""
    prediction_tensor = torch.tensor(predictions, dtype=torch.float64)
    target_tensor = torch.tensor(targets, dtype=torch.float64)
    quality = regression_quality_stats(
        predictions=prediction_tensor,
        targets=target_tensor,
    )
    residuals = prediction_tensor - target_tensor
    absolute_errors = torch.abs(residuals)
    squared_errors = residuals.square()
    mse = cast("float", quality.mse)
    return {
        "count": quality.count,
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": quality.mae,
        "mean_error": quality.residual_mean,
        "residual_standard_deviation": quality.residual_std,
        "target_mean": quality.target_mean,
        "target_standard_deviation": quality.target_std,
        "prediction_mean": quality.prediction_mean,
        "prediction_standard_deviation": quality.prediction_std,
        "prediction_standard_deviation_over_target_standard_deviation": (
            quality.prediction_std_over_target_std
        ),
        "pearson_correlation": quality.pearson_correlation,
        "r2_vs_mean_baseline": quality.r2_vs_mean_baseline,
        "mean_baseline_mse": quality.mean_baseline_mse,
        "maximum_absolute_error": float(torch.max(absolute_errors).item()),
        "absolute_error_percentiles": _percentiles(
            absolute_errors,
            (50, 75, 90, 95, 99),
        ),
        "squared_error_percentiles": _percentiles(
            squared_errors,
            (50, 90, 95, 99),
        ),
    }


def _percentiles(
    values: torch.Tensor,
    percentile_values: tuple[int, ...],
) -> dict[str, float]:
    """Return linearly interpolated percentiles with stable names."""
    quantiles = torch.tensor(
        tuple(value / 100.0 for value in percentile_values),
        dtype=torch.float64,
    )
    results = torch.quantile(values.to(dtype=torch.float64), quantiles)
    return {
        f"p{percentile}": float(result.item())
        for percentile, result in zip(percentile_values, results, strict=True)
    }


def _pair_key(first_name: str, second_name: str) -> str:
    """Return one mapping-order-independent unordered pair key."""
    first, second = sorted((first_name, second_name))
    return f"{first}__vs__{second}"


def _pairwise_metrics(
    *,
    evaluator_names: tuple[str, ...],
    predictions: dict[str, tuple[float, ...]],
    targets: tuple[float, ...],
) -> dict[str, object]:
    """Compute prediction agreement, residual agreement, and paired wins."""
    pairwise: dict[str, object] = {}
    for first_name, second_name in combinations(evaluator_names, 2):
        first_predictions = predictions[first_name]
        second_predictions = predictions[second_name]
        first_residuals = tuple(
            prediction - target
            for prediction, target in zip(first_predictions, targets, strict=True)
        )
        second_residuals = tuple(
            prediction - target
            for prediction, target in zip(second_predictions, targets, strict=True)
        )
        disagreements = tuple(
            first - second
            for first, second in zip(
                first_predictions,
                second_predictions,
                strict=True,
            )
        )
        first_wins: list[int] = []
        second_wins: list[int] = []
        tied: list[int] = []
        for index, (first_residual, second_residual) in enumerate(
            zip(first_residuals, second_residuals, strict=True)
        ):
            error_delta = abs(first_residual) - abs(second_residual)
            if abs(error_delta) <= PAIRWISE_ERROR_TIE_TOLERANCE:
                tied.append(index)
            elif error_delta < 0.0:
                first_wins.append(index)
            else:
                second_wins.append(index)
        count = len(targets)
        pairwise[_pair_key(first_name, second_name)] = {
            "evaluator_a": first_name,
            "evaluator_b": second_name,
            "tie_tolerance": PAIRWISE_ERROR_TIE_TOLERANCE,
            "prediction_pearson_correlation": _pearson(
                first_predictions,
                second_predictions,
            ),
            "residual_pearson_correlation": _pearson(
                first_residuals,
                second_residuals,
            ),
            "mean_absolute_prediction_disagreement": _mean(
                tuple(abs(value) for value in disagreements)
            ),
            "root_mean_squared_prediction_disagreement": math.sqrt(
                _mean(tuple(value * value for value in disagreements))
            ),
            "fraction_a_lower_absolute_error": len(first_wins) / count,
            "fraction_b_lower_absolute_error": len(second_wins) / count,
            "fraction_tied": len(tied) / count,
            "mean_target_on_rows_won_by_a": _mean_selected(targets, first_wins),
            "mean_target_on_rows_won_by_b": _mean_selected(targets, second_wins),
        }
    return pairwise


def _ensemble_metrics(
    *,
    evaluator_names: tuple[str, ...],
    predictions: dict[str, tuple[float, ...]],
    targets: tuple[float, ...],
) -> dict[str, object]:
    """Compute equal-weight pair/all ensembles and the non-deployable oracle."""
    ensembles: dict[str, object] = {}
    individual_mse = {
        name: cast("float", _regression_metrics(predictions[name], targets)["mse"])
        for name in evaluator_names
    }
    for first_name, second_name in combinations(evaluator_names, 2):
        pair_prediction = tuple(
            0.5 * (first + second)
            for first, second in zip(
                predictions[first_name],
                predictions[second_name],
                strict=True,
            )
        )
        pair_metrics = _ensemble_regression_metrics(pair_prediction, targets)
        pair_mse = cast("float", pair_metrics["mse"])
        better_mse = min(individual_mse[first_name], individual_mse[second_name])
        ensembles[_pair_key(first_name, second_name)] = {
            "evaluator_a": first_name,
            "evaluator_b": second_name,
            **pair_metrics,
            "mse_improvement_vs_a": individual_mse[first_name] - pair_mse,
            "mse_improvement_vs_b": individual_mse[second_name] - pair_mse,
            "relative_mse_improvement_vs_better_individual": (
                None if better_mse <= 0.0 else (better_mse - pair_mse) / better_mse
            ),
        }
    if len(evaluator_names) >= 3:
        all_prediction = tuple(
            math.fsum(predictions[name][index] for name in evaluator_names)
            / len(evaluator_names)
            for index in range(len(targets))
        )
        ensembles["all_evaluators_equal_weight"] = {
            "evaluator_names": list(evaluator_names),
            **_ensemble_regression_metrics(all_prediction, targets),
        }
    oracle_predictions: list[float] = []
    winning_counts = {name: 0 for name in evaluator_names}
    for index, target in enumerate(targets):
        winner = min(
            evaluator_names,
            key=lambda name: (abs(predictions[name][index] - target), name),
        )
        winning_counts[winner] += 1
        oracle_predictions.append(predictions[winner][index])
    oracle_metrics = _ensemble_regression_metrics(tuple(oracle_predictions), targets)
    ensembles["per_row_oracle"] = {
        "deployable": False,
        "description": (
            "Analysis upper bound that selects the lowest-error evaluator using "
            "each row's target; it is not a usable inference method."
        ),
        "mse": oracle_metrics["mse"],
        "mae": oracle_metrics["mae"],
        "winning_row_count_per_evaluator": winning_counts,
    }
    return ensembles


def _ensemble_regression_metrics(
    predictions: tuple[float, ...],
    targets: tuple[float, ...],
) -> dict[str, object]:
    """Return the required scalar metrics for one fixed ensemble prediction."""
    metrics = _regression_metrics(predictions, targets)
    return {
        key: metrics[key]
        for key in (
            "mse",
            "rmse",
            "mae",
            "pearson_correlation",
            "r2_vs_mean_baseline",
        )
    }


def _pearson(first: tuple[float, ...], second: tuple[float, ...]) -> float | None:
    """Return population Pearson correlation, or None for zero variance."""
    stats = regression_quality_stats(
        predictions=torch.tensor(first, dtype=torch.float64),
        targets=torch.tensor(second, dtype=torch.float64),
    )
    return stats.pearson_correlation


def _mean(values: tuple[float, ...]) -> float:
    """Return a stable arithmetic mean for one non-empty tuple."""
    return math.fsum(values) / len(values)


def _mean_selected(
    values: tuple[float, ...],
    selected_indices: list[int],
) -> float | None:
    """Return the mean of selected values, or None when no row was selected."""
    if not selected_indices:
        return None
    return math.fsum(values[index] for index in selected_indices) / len(
        selected_indices
    )


def _worst_rows_artifact(
    *,
    evaluator_names: tuple[str, ...],
    row_indices: tuple[int, ...],
    targets: tuple[float, ...],
    predictions: dict[str, tuple[float, ...]],
) -> dict[str, object]:
    """Build deterministic worst-error and largest-disagreement row records."""
    evaluator_rows: dict[str, object] = {}
    for evaluator_name in evaluator_names:
        ordered_positions = sorted(
            range(len(targets)),
            key=lambda position: (
                -((predictions[evaluator_name][position] - targets[position]) ** 2),
                row_indices[position],
            ),
        )[:WORST_ROW_LIMIT]
        evaluator_rows[evaluator_name] = [
            _worst_evaluator_row(
                evaluator_name=evaluator_name,
                position=position,
                evaluator_names=evaluator_names,
                row_indices=row_indices,
                targets=targets,
                predictions=predictions,
            )
            for position in ordered_positions
        ]
    disagreement_positions = sorted(
        range(len(targets)),
        key=lambda position: (
            -_prediction_spread(evaluator_names, predictions, position),
            row_indices[position],
        ),
    )[:WORST_ROW_LIMIT]
    largest_disagreements = [
        {
            "row_index": row_indices[position],
            "target": targets[position],
            "prediction_spread": _prediction_spread(
                evaluator_names,
                predictions,
                position,
            ),
            "predictions": {
                name: predictions[name][position] for name in evaluator_names
            },
            "residuals": {
                name: predictions[name][position] - targets[position]
                for name in evaluator_names
            },
        }
        for position in disagreement_positions
    ]
    return {
        "schema": MORPION_COMPARISON_SCHEMA,
        "residual_convention": "prediction_minus_target",
        "worst_rows_by_evaluator": evaluator_rows,
        "largest_disagreements": largest_disagreements,
    }


def _worst_evaluator_row(
    *,
    evaluator_name: str,
    position: int,
    evaluator_names: tuple[str, ...],
    row_indices: tuple[int, ...],
    targets: tuple[float, ...],
    predictions: dict[str, tuple[float, ...]],
) -> dict[str, object]:
    """Return one evaluator-specific worst-row record."""
    target = targets[position]
    prediction = predictions[evaluator_name][position]
    residual = prediction - target
    other_names = tuple(name for name in evaluator_names if name != evaluator_name)
    return {
        "row_index": row_indices[position],
        "target": target,
        "prediction": prediction,
        "residual": residual,
        "absolute_error": abs(residual),
        "squared_error": residual * residual,
        "other_predictions": {
            name: predictions[name][position] for name in other_names
        },
        "other_residuals": {
            name: predictions[name][position] - target for name in other_names
        },
    }


def _prediction_spread(
    evaluator_names: tuple[str, ...],
    predictions: dict[str, tuple[float, ...]],
    position: int,
) -> float:
    """Return max-minus-min prediction spread for one paired row."""
    values = tuple(predictions[name][position] for name in evaluator_names)
    return max(values) - min(values)


def _error_tail_summary(
    *,
    evaluator_names: tuple[str, ...],
    evaluator_metrics: dict[str, dict[str, object]],
    worst_rows: dict[str, object],
) -> dict[str, object]:
    """Return compact tail metrics and deterministic row-index lookups."""
    by_evaluator = cast(
        "dict[str, list[dict[str, object]]]",
        worst_rows["worst_rows_by_evaluator"],
    )
    largest_disagreements = cast(
        "list[dict[str, object]]",
        worst_rows["largest_disagreements"],
    )
    return {
        "evaluators": {
            name: {
                "maximum_absolute_error": evaluator_metrics[name][
                    "maximum_absolute_error"
                ],
                "absolute_error_percentiles": evaluator_metrics[name][
                    "absolute_error_percentiles"
                ],
                "squared_error_percentiles": evaluator_metrics[name][
                    "squared_error_percentiles"
                ],
                "worst_row_indices": [row["row_index"] for row in by_evaluator[name]],
            }
            for name in evaluator_names
        },
        "largest_disagreement_row_indices": [
            row["row_index"] for row in largest_disagreements
        ],
    }


def _csv_prefixes(evaluator_names: tuple[str, ...]) -> dict[str, str]:
    """Sanitize names into stable, unambiguous CSV column prefixes."""
    prefixes: dict[str, str] = {}
    owners: dict[str, str] = {}
    for name in sorted(evaluator_names):
        prefix = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_") or "evaluator"
        owner = owners.get(prefix)
        if owner is not None:
            raise InvalidMorpionComparisonInputError.csv_prefix_collision(
                owner,
                name,
                prefix,
            )
        owners[prefix] = name
        prefixes[name] = prefix
    return prefixes


def _prediction_row(
    diagnostics: MorpionComparisonDiagnostics,
    position: int,
) -> dict[str, object]:
    """Return one JSON-friendly paired prediction record."""
    target = diagnostics.targets[position]
    predictions = {
        name: values[position] for name, values in diagnostics.predictions.items()
    }
    residuals = {name: prediction - target for name, prediction in predictions.items()}
    return {
        "row_index": diagnostics.validation_row_indices[position],
        "target": target,
        "predictions": predictions,
        "residuals": residuals,
        "absolute_errors": {
            name: abs(residual) for name, residual in residuals.items()
        },
        "squared_errors": {
            name: residual * residual for name, residual in residuals.items()
        },
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Write one deterministic finite-number JSON document."""
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _write_predictions_jsonl(
    path: Path,
    diagnostics: MorpionComparisonDiagnostics,
) -> None:
    """Stream one finite JSON object per paired validation row."""
    with path.open("w", encoding="utf-8") as stream:
        for position in range(len(diagnostics.targets)):
            json.dump(
                _prediction_row(diagnostics, position),
                stream,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            stream.write("\n")


def _write_predictions_csv(
    path: Path,
    diagnostics: MorpionComparisonDiagnostics,
) -> None:
    """Stream one flat CSV record per paired validation row."""
    evaluator_names = tuple(diagnostics.predictions)
    field_names = ["row_index", "target"]
    metric_suffixes = (
        "prediction",
        "residual",
        "absolute_error",
        "squared_error",
    )
    for name in evaluator_names:
        prefix = diagnostics.csv_prefixes[name]
        field_names.extend(f"{prefix}_{suffix}" for suffix in metric_suffixes)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=field_names, lineterminator="\n")
        writer.writeheader()
        for position, target in enumerate(diagnostics.targets):
            row: dict[str, object] = {
                "row_index": diagnostics.validation_row_indices[position],
                "target": target,
            }
            for name in evaluator_names:
                prefix = diagnostics.csv_prefixes[name]
                prediction = diagnostics.predictions[name][position]
                residual = prediction - target
                row[f"{prefix}_prediction"] = prediction
                row[f"{prefix}_residual"] = residual
                row[f"{prefix}_absolute_error"] = abs(residual)
                row[f"{prefix}_squared_error"] = residual * residual
            writer.writerow(row)


def _parse_bundle_argument(value: str) -> MorpionComparisonBundle:
    """Parse one exact repeatable ``NAME=PATH`` CLI argument."""
    if value.count("=") != 1:
        raise _MalformedBundleArgumentError.invalid_separator()
    name, raw_path = value.split("=", maxsplit=1)
    if not name.strip() or not raw_path.strip():
        raise _MalformedBundleArgumentError.empty_component()
    return MorpionComparisonBundle(
        evaluator_name=name,
        bundle_path=Path(raw_path),
    )


def _argument_parser() -> argparse.ArgumentParser:
    """Build the small standalone comparison CLI parser."""
    parser = argparse.ArgumentParser(
        description="Compare saved Morpion evaluators on paired validation rows."
    )
    parser.add_argument("--dataset-file", type=Path, required=True)
    parser.add_argument(
        "--bundle",
        type=_parse_bundle_argument,
        action="append",
        required=True,
        help="Repeatable evaluator bundle in exact NAME=PATH form.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    return parser


def _print_summary(diagnostics: MorpionComparisonDiagnostics, stream: TextIO) -> None:
    """Print a concise ranking and the most useful paired diagnostics."""
    evaluators = cast("dict[str, dict[str, object]]", diagnostics.summary["evaluators"])
    ranking = sorted(
        evaluators,
        key=lambda name: (cast("float", evaluators[name]["mse"]), name),
    )
    print("Validation ranking by MSE:", file=stream)
    for rank, name in enumerate(ranking, start=1):
        print(
            f"  {rank}. {name}: MSE={evaluators[name]['mse']:.8g} "
            f"MAE={evaluators[name]['mae']:.8g}",
            file=stream,
        )
    pairwise = cast("dict[str, dict[str, object]]", diagnostics.summary["pairwise"])
    print("Pairwise residual correlations:", file=stream)
    for key, metrics in pairwise.items():
        print(f"  {key}: {metrics['residual_pearson_correlation']}", file=stream)
    ensembles = cast("dict[str, dict[str, object]]", diagnostics.summary["ensembles"])
    print("Equal-weight ensemble MSE:", file=stream)
    for key, metrics in ensembles.items():
        if key == "per_row_oracle":
            continue
        print(f"  {key}: {metrics['mse']:.8g}", file=stream)
    oracle = ensembles["per_row_oracle"]
    print(f"Per-row oracle MSE: {oracle['mse']:.8g}", file=stream)
    error_tail = cast("dict[str, object]", diagnostics.summary["error_tail"])
    tails = cast("dict[str, dict[str, object]]", error_tail["evaluators"])
    print("Worst-error row indices:", file=stream)
    for name in sorted(tails):
        indices = cast("list[int]", tails[name]["worst_row_indices"])
        print(f"  {name}: {indices[:10]}", file=stream)
    disagreement_indices = cast(
        "list[int]",
        error_tail["largest_disagreement_row_indices"],
    )
    print(f"Largest-disagreement row indices: {disagreement_indices[:10]}", file=stream)


def main(argv: Sequence[str] | None = None) -> int:
    """Run paired Morpion comparison diagnostics from the command line."""
    parser = _argument_parser()
    namespace = parser.parse_args(argv)
    comparison_args = MorpionComparisonDiagnosticsArgs(
        dataset_file=namespace.dataset_file,
        bundles=tuple(namespace.bundle),
        output_dir=namespace.output_dir,
        max_rows=namespace.max_rows,
        validation_fraction=namespace.validation_fraction,
        batch_size=namespace.batch_size,
        device=namespace.device,
    )
    try:
        diagnostics = build_morpion_comparison_diagnostics(comparison_args)
        save_morpion_comparison_diagnostics(diagnostics, comparison_args.output_dir)
    except MorpionComparisonDiagnosticsError as exc:
        print(f"Morpion comparison diagnostics failed: {exc}", file=sys.stderr)
        return 2
    _print_summary(diagnostics, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MORPION_COMPARISON_SCHEMA",
    "PAIRWISE_ERROR_TIE_TOLERANCE",
    "InvalidMorpionComparisonInputError",
    "MorpionComparisonBundle",
    "MorpionComparisonBundleLoadError",
    "MorpionComparisonDiagnostics",
    "MorpionComparisonDiagnosticsArgs",
    "MorpionComparisonDiagnosticsError",
    "MorpionComparisonEvaluationError",
    "build_morpion_comparison_diagnostics",
    "main",
    "save_morpion_comparison_diagnostics",
]
