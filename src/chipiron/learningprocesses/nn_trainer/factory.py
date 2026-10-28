"""Compatibility facade for legacy supervised chess training imports.

New code should import training config from:
    chipiron.scripts.learn_nn_supervised.training_args

New code should import checkpoint helpers from:
    chipiron.scripts.learn_nn_supervised.checkpoint_helpers

This module remains only to support older imports during the migration away from
learningprocesses.nn_trainer.
"""

from chipiron.scripts.learn_nn_supervised.checkpoint_helpers import (
    safe_nn_architecture_save,
    safe_nn_param_save,
    safe_nn_trainer_save,
)
from chipiron.scripts.learn_nn_supervised.training_args import (
    GameInputArgs,
    NNTrainerArgs,
    NNTrainerConfigError,
    OptimizerType,
    SupervisedTrainingArgs,
    SupervisedTrainingConfigError,
)

__all__ = [
    "GameInputArgs",
    "NNTrainerArgs",
    "NNTrainerConfigError",
    "OptimizerType",
    "SupervisedTrainingArgs",
    "SupervisedTrainingConfigError",
    "safe_nn_architecture_save",
    "safe_nn_param_save",
    "safe_nn_trainer_save",
]
