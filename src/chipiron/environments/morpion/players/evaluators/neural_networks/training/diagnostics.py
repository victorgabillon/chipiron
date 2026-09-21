"""Diagnostic prediction adapters for Morpion neural-network evaluators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import torch

from chipiron.environments.morpion.players.evaluators.neural_networks.entity_relations import (
    is_relational_entity_token_model_kind,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.entity_tokens import (
    MORPION_ENTITY_TOKEN_FEATURE_DIM,
    is_morpion_entity_token_model_kind,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.feature_schema import (
    DEFAULT_MORPION_FEATURE_SUBSET_NAME,
)
from chipiron.learning.supervised import move_supervised_batch_to_device
from chipiron.learning.torch_runtime import module_device

from .args import MorpionTrainingArgs
from .row_batches import rows_to_sample_batch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import nn

    from chipiron.environments.morpion.learning import MorpionSupervisedRow
    from chipiron.learning.supervised import TensorSupervisedBatch


@dataclass(frozen=True, slots=True)
class MorpionDiagnosticPredictionResult:
    """Predictions or a clean diagnostics skip reason."""

    predictions: list[float]
    skipped: bool = False
    reason: str | None = None
    detail: str | None = None


class UnsupportedMorpionDiagnosticInputFormatError(ValueError):
    """Raised when diagnostics cannot infer a model's supervised-row input path."""

    reason = "unsupported_model_input_format"

    def __init__(self, model_kind: object, detail: str | None = None) -> None:
        """Initialize the unsupported diagnostic input-format error."""
        self.model_kind = model_kind
        self.detail = detail
        message = f"Unsupported Morpion diagnostics model input format: {model_kind!r}."
        if detail:
            message = f"{message} {detail}"
        super().__init__(message)


def try_predict_morpion_rows_for_diagnostics(
    model: nn.Module,
    row_examples: Sequence[MorpionSupervisedRow],
    *,
    feature_subset_name: str = DEFAULT_MORPION_FEATURE_SUBSET_NAME,
    feature_names: tuple[str, ...] = (),
) -> MorpionDiagnosticPredictionResult:
    """Predict diagnostics rows or return a clean skip reason."""
    if not row_examples:
        return MorpionDiagnosticPredictionResult(predictions=[])
    try:
        predictions = predict_morpion_rows_for_diagnostics(
            model,
            row_examples,
            feature_subset_name=feature_subset_name,
            feature_names=feature_names,
        )
    except UnsupportedMorpionDiagnosticInputFormatError as exc:
        return MorpionDiagnosticPredictionResult(
            predictions=[],
            skipped=True,
            reason=UnsupportedMorpionDiagnosticInputFormatError.reason,
            detail=str(exc),
        )
    except ValueError as exc:
        if not _is_diagnostics_input_format_value_error(exc):
            raise
        return MorpionDiagnosticPredictionResult(
            predictions=[],
            skipped=True,
            reason=UnsupportedMorpionDiagnosticInputFormatError.reason,
            detail=str(exc),
        )
    return MorpionDiagnosticPredictionResult(predictions=predictions)


def predict_morpion_rows_for_diagnostics(
    model: nn.Module,
    row_examples: Sequence[MorpionSupervisedRow],
    *,
    feature_subset_name: str = DEFAULT_MORPION_FEATURE_SUBSET_NAME,
    feature_names: tuple[str, ...] = (),
) -> list[float]:
    """Predict raw Morpion rows using the model family's training input adapter."""
    if not row_examples:
        return []

    model_args = getattr(model, "args", None)
    model_kind = getattr(model_args, "model_kind", None)
    if model_kind is None:
        raise UnsupportedMorpionDiagnosticInputFormatError(model_kind)
    diagnostic_adapter_kind(model_kind)

    diagnostics_args = diagnostic_training_args(
        model_args=model_args,
        feature_subset_name=feature_subset_name,
        feature_names=feature_names,
    )
    sample_batch = rows_to_sample_batch(tuple(row_examples), args=diagnostics_args)
    model.eval()
    with torch.no_grad():
        try:
            return _predict_from_sample_batch(model, sample_batch)
        except ValueError as exc:
            if not _is_diagnostics_input_format_value_error(exc):
                raise
            raise UnsupportedMorpionDiagnosticInputFormatError(
                model_kind,
                detail=str(exc),
            ) from exc


def diagnostic_adapter_kind(
    model_kind: object,
) -> Literal["flat", "entity_tokens", "relational_entity_tokens"]:
    """Return the Morpion diagnostics adapter family for one model kind."""
    if model_kind in {"linear", "mlp"}:
        return "flat"
    if isinstance(model_kind, str) and is_relational_entity_token_model_kind(
        model_kind
    ):
        return "relational_entity_tokens"
    if isinstance(model_kind, str) and is_morpion_entity_token_model_kind(model_kind):
        return "entity_tokens"
    raise UnsupportedMorpionDiagnosticInputFormatError(model_kind)


def diagnostic_training_args(
    *,
    model_args: object,
    feature_subset_name: str,
    feature_names: tuple[str, ...],
) -> MorpionTrainingArgs:
    """Build Morpion training args for diagnostics-only row adaptation."""
    model_kind = getattr(model_args, "model_kind", None)
    adapter_kind = diagnostic_adapter_kind(model_kind)
    resolved_feature_subset_name = feature_subset_name
    resolved_feature_names = feature_names
    if adapter_kind == "flat":
        resolved_feature_subset_name = str(
            getattr(model_args, "feature_subset_name", feature_subset_name)
        )
        resolved_feature_names = tuple(
            getattr(model_args, "feature_names", feature_names)
        )
    return MorpionTrainingArgs(
        dataset_file="",
        output_dir="",
        model_kind=model_kind,
        feature_subset_name=resolved_feature_subset_name,
        feature_names=resolved_feature_names,
        hidden_sizes=cast(
            "tuple[int, ...] | None", getattr(model_args, "hidden_sizes", None)
        ),
        entity_max_tokens=int(getattr(model_args, "entity_max_tokens", 1536)),
        entity_input_feature_dim=int(
            getattr(
                model_args,
                "entity_input_feature_dim",
                MORPION_ENTITY_TOKEN_FEATURE_DIM,
            )
        ),
        entity_d_model=int(getattr(model_args, "entity_d_model", 64)),
        entity_n_head=int(getattr(model_args, "entity_n_head", 4)),
        entity_n_layer=int(getattr(model_args, "entity_n_layer", 2)),
        entity_dim_feedforward=int(getattr(model_args, "entity_dim_feedforward", 256)),
        entity_dropout_ratio=float(getattr(model_args, "entity_dropout_ratio", 0.0)),
        entity_pooling=str(getattr(model_args, "entity_pooling", "value_token")),
        entity_output_tanh=bool(getattr(model_args, "entity_output_tanh", False)),
        entity_use_validity_feature=bool(
            getattr(model_args, "entity_use_validity_feature", True)
        ),
        entity_relation_schema=cast(
            "str | None", getattr(model_args, "entity_relation_schema", None)
        ),
        entity_relation_type_count=cast(
            "int | None", getattr(model_args, "entity_relation_type_count", None)
        ),
        device="auto",
    )


def move_tensor_to_model_device(tensor: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """Move one tensor to the model's current device."""
    return tensor.to(module_device(model))


def _predict_from_sample_batch(
    model: nn.Module,
    sample_batch: TensorSupervisedBatch,
) -> list[float]:
    """Run one diagnostics batch through the model and return float predictions."""
    device_batch = move_supervised_batch_to_device(sample_batch, module_device(model))
    predictions = model(*device_batch.get_model_input_tensors())
    prediction_values = predictions.squeeze(-1).detach().cpu().tolist()
    if isinstance(prediction_values, float):
        return [float(prediction_values)]
    return [float(prediction) for prediction in prediction_values]


def _is_diagnostics_input_format_value_error(exc: ValueError) -> bool:
    """Return whether a ValueError clearly describes a diagnostics input mismatch."""
    message = str(exc).lower()
    if not message:
        return False
    return any(
        marker in message
        for marker in (
            "input",
            "shape",
            "dimension",
            "dim",
            "ndim",
            "expected",
            "normalize",
            "token",
        )
    )
