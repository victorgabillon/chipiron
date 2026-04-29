"""Pipeline-mode and evaluator-update policy types for Morpion bootstrap."""

from __future__ import annotations

from typing import Literal

MorpionEvaluatorUpdatePolicy = Literal[
    "future_only",
    "reevaluate_all",
    "reevaluate_frontier",
]

MorpionPipelineMode = Literal[
    "single_process",
    "artifact_pipeline",
]

MorpionPipelineStage = Literal[
    "loop",
    "growth",
    "dataset",
    "training",
    "reevaluation",
]

DEFAULT_MORPION_EVALUATOR_UPDATE_POLICY: MorpionEvaluatorUpdatePolicy = "future_only"
DEFAULT_MORPION_PIPELINE_MODE: MorpionPipelineMode = "single_process"

__all__ = [
    "DEFAULT_MORPION_EVALUATOR_UPDATE_POLICY",
    "DEFAULT_MORPION_PIPELINE_MODE",
    "MorpionEvaluatorUpdatePolicy",
    "MorpionPipelineMode",
    "MorpionPipelineStage",
]
