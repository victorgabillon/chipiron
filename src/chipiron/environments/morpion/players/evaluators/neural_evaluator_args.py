"""Serializable configuration for normal Morpion neural evaluation."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class MorpionNeuralEvaluatorArgs:
    """Locate a portable bundle; its manifest defines the model and representation."""

    model_bundle: str
    device: str = "cpu"
    type: Literal["morpion_neural"] = "morpion_neural"
