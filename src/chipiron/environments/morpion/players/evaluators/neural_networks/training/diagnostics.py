"""Diagnostic prediction adapters for Morpion neural-network evaluators."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from chipiron.environments.morpion.players.evaluators.neural_networks.feature_schema import (
    DEFAULT_MORPION_FEATURE_SUBSET_NAME,
)
from chipiron.environments.morpion.players.evaluators.neural_networks.graph_tokens import (
    MORPION_GRAPH_TOKEN_FEATURE_DIM,
    is_morpion_entity_token_transformer_model_kind,
)
from chipiron.learning.torch_runtime import module_device

from .args import MorpionTrainingArgs
from .row_batches import rows_to_sample_batch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import nn

    from chipiron.environments.morpion.learning import MorpionSupervisedRow


class UnsupportedMorpionDiagnosticInputFormatError(ValueError):
    """Raised when diagnostics cannot infer a model's supervised-row input path."""

    reason = "unsupported_model_input_format"

    def __init__(self, model_kind: object) -> None:
        """Initialize the unsupported diagnostic input-format error."""
        super().__init__(
            f"Unsupported Morpion diagnostics model input format: {model_kind!r}."
        )


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

    diagnostics_args = diagnostic_training_args(
        model_args=model_args,
        feature_subset_name=feature_subset_name,
        feature_names=feature_names,
    )
    sample_batch = rows_to_sample_batch(tuple(row_examples), args=diagnostics_args)
    model.eval()
    with torch.no_grad():
        predictions = model(
            move_tensor_to_model_device(sample_batch.get_input_layer(), model)
        )
    prediction_values = predictions.squeeze(-1).detach().cpu().tolist()
    if isinstance(prediction_values, float):
        return [float(prediction_values)]
    return [float(prediction) for prediction in prediction_values]


def diagnostic_training_args(
    *,
    model_args: object,
    feature_subset_name: str,
    feature_names: tuple[str, ...],
) -> MorpionTrainingArgs:
    """Build Morpion training args for diagnostics-only row adaptation."""
    model_kind = getattr(model_args, "model_kind", None)
    if model_kind not in {"linear", "mlp"} and not (
        isinstance(model_kind, str)
        and is_morpion_entity_token_transformer_model_kind(model_kind)
    ):
        raise UnsupportedMorpionDiagnosticInputFormatError(model_kind)
    return MorpionTrainingArgs(
        dataset_file="",
        output_dir="",
        model_kind=model_kind,
        feature_subset_name=str(
            getattr(model_args, "feature_subset_name", feature_subset_name)
        ),
        feature_names=tuple(getattr(model_args, "feature_names", feature_names)),
        hidden_sizes=cast(
            "tuple[int, ...] | None", getattr(model_args, "hidden_sizes", None)
        ),
        graph_max_tokens=int(getattr(model_args, "graph_max_tokens", 1536)),
        graph_input_feature_dim=int(
            getattr(
                model_args,
                "graph_input_feature_dim",
                MORPION_GRAPH_TOKEN_FEATURE_DIM,
            )
        ),
        graph_d_model=int(getattr(model_args, "graph_d_model", 64)),
        graph_n_head=int(getattr(model_args, "graph_n_head", 4)),
        graph_n_layer=int(getattr(model_args, "graph_n_layer", 2)),
        graph_dim_feedforward=int(getattr(model_args, "graph_dim_feedforward", 256)),
        graph_dropout_ratio=float(getattr(model_args, "graph_dropout_ratio", 0.0)),
        graph_pooling=str(getattr(model_args, "graph_pooling", "value_token")),
        graph_output_tanh=bool(getattr(model_args, "graph_output_tanh", True)),
        device="auto",
    )


def move_tensor_to_model_device(tensor: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """Move one tensor to the model's current device."""
    return tensor.to(module_device(model))
