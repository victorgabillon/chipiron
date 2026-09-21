"""Normal Morpion neural value evaluation from portable model bundles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

import torch
from valanga.evaluations import Certainty, Value

from chipiron.environments.morpion.types import MorpionDynamics, MorpionState
from chipiron.learning.torch_runtime import module_device

from .morpion_state_evaluator import (
    MorpionMasterEvaluator,
    MorpionOverEventDetector,
    MorpionStateEvaluator,
)
from .neural_networks.bundle import load_morpion_model_bundle
from .neural_networks.entity_relations import (
    MorpionRelationalEntityTokenConverter,
    is_relational_entity_token_model_kind,
)
from .neural_networks.entity_tokens import (
    MorpionEntityTokenConverter,
    is_morpion_entity_token_model_kind,
)
from .neural_networks.state_to_tensor import MorpionFeatureTensorConverter

if TYPE_CHECKING:
    from pathlib import Path


class MorpionStateToTensorConverter(Protocol):
    """Minimal interface shared by Morpion neural input converters."""

    def state_to_model_input_tensors(
        self,
        state: MorpionState,
    ) -> tuple[torch.Tensor, ...]:
        """Convert one Morpion state to positional model input tensors."""
        ...


class _MorpionRegressor(Protocol):
    """Callable neural regressor loaded from a Morpion model bundle."""

    def __call__(self, *model_inputs: torch.Tensor) -> Any:
        """Return the raw model output for one converted state."""
        ...


@dataclass(frozen=True, slots=True)
class MorpionRegressorMasterEvaluator(MorpionMasterEvaluator):
    """Anemone-compatible Morpion evaluator backed by a saved regressor bundle."""

    input_converter: MorpionStateToTensorConverter
    regressor: object

    @property
    def feature_converter(self) -> MorpionStateToTensorConverter:
        """Return the input converter under the legacy attribute name."""
        return self.input_converter

    def evaluate(self, state: object) -> Value:
        """Evaluate a Morpion state through the loaded regressor bundle."""
        over_event, terminal_value = self.over_detector.check_obvious_over_events(
            cast("Any", state)
        )
        if terminal_value is not None:
            return Value(
                score=terminal_value,
                certainty=Certainty.TERMINAL,
                over_event=over_event,
            )

        morpion_state = cast("MorpionState", state)
        model_inputs = self.input_converter.state_to_model_input_tensors(morpion_state)
        regressor = cast(_MorpionRegressor, self.regressor)  # noqa: TC006
        device_inputs = tuple(
            tensor.to(module_device(cast("torch.nn.Module", regressor)))
            for tensor in model_inputs
        )
        with torch.no_grad():
            raw_output = regressor(*device_inputs)  # pylint: disable=not-callable
        score = float(raw_output.detach().cpu().reshape(-1)[0].item())
        return Value(
            score=score,
            certainty=Certainty.ESTIMATE,
            over_event=None,
        )


def load_morpion_evaluator_from_model_bundle(
    model_bundle_path: str | Path,
    *,
    device: str = "cpu",
) -> MorpionMasterEvaluator:
    """Load one saved Morpion bundle into the Anemone evaluator protocol."""
    model, model_args, _ = load_morpion_model_bundle(model_bundle_path)
    model.to(torch.device(device)).eval()
    over_detector = MorpionOverEventDetector()
    input_converter: MorpionStateToTensorConverter
    if is_relational_entity_token_model_kind(model_args.model_kind):
        input_converter = MorpionRelationalEntityTokenConverter(
            dynamics=MorpionDynamics(),
            max_tokens=model_args.entity_max_tokens,
            global_geometry_features=model_args.global_geometry_features,
            edge_token_mode=model_args.edge_token_mode,
            latent_window_move_features=model_args.latent_window_move_features,
        )
    elif is_morpion_entity_token_model_kind(model_args.model_kind):
        input_converter = MorpionEntityTokenConverter(
            dynamics=MorpionDynamics(),
            max_tokens=model_args.entity_max_tokens,
            global_geometry_features=model_args.global_geometry_features,
            edge_token_mode=model_args.edge_token_mode,
            latent_window_move_features=model_args.latent_window_move_features,
        )
    else:
        input_converter = MorpionFeatureTensorConverter(
            dynamics=MorpionDynamics(),
            feature_subset=model_args.feature_subset,
        )
    return MorpionRegressorMasterEvaluator(
        evaluator=MorpionStateEvaluator(),
        over=over_detector,
        over_detector=over_detector,
        input_converter=input_converter,
        regressor=model,
    )
