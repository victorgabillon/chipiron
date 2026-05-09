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

MorpionTrainingExportMode = Literal[
    "flat",
    "sharded",
    "both",
]

MorpionPipelineStage = Literal[
    "loop",
    "growth",
    "dataset",
    "dataset_worker",
    "training",
    "training_worker",
    "reevaluation",
]

DEFAULT_MORPION_EVALUATOR_UPDATE_POLICY: MorpionEvaluatorUpdatePolicy = "future_only"
DEFAULT_MORPION_PIPELINE_MODE: MorpionPipelineMode = "single_process"
DEFAULT_MORPION_TRAINING_EXPORT_MODE: MorpionTrainingExportMode = "flat"

__all__ = [
    "DEFAULT_MORPION_EVALUATOR_UPDATE_POLICY",
    "DEFAULT_MORPION_PIPELINE_MODE",
    "DEFAULT_MORPION_TRAINING_EXPORT_MODE",
    "MorpionEvaluatorUpdatePolicy",
    "MorpionPipelineMode",
    "MorpionPipelineStage",
    "MorpionTrainingExportMode",
]
