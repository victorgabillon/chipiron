"""Legacy compatibility facade for supervised chess training config.

New code should import config classes from
``chipiron.scripts.learn_nn_supervised.training_args`` and checkpoint helpers
from ``chipiron.scripts.learn_nn_supervised.checkpoint_helpers``.
"""

from __future__ import annotations

from chipiron.scripts.learn_nn_supervised.checkpoint_helpers import (
    safe_nn_architecture_save as safe_nn_architecture_save,
)
from chipiron.scripts.learn_nn_supervised.checkpoint_helpers import (
    safe_nn_param_save as safe_nn_param_save,
)
from chipiron.scripts.learn_nn_supervised.checkpoint_helpers import (
    safe_nn_trainer_save as safe_nn_trainer_save,
)
from chipiron.scripts.learn_nn_supervised.training_args import (
    GameInputArgs as GameInputArgs,
)
from chipiron.scripts.learn_nn_supervised.training_args import (
    NNTrainerArgs as NNTrainerArgs,
)
from chipiron.scripts.learn_nn_supervised.training_args import (
    NNTrainerConfigError as NNTrainerConfigError,
)
from chipiron.scripts.learn_nn_supervised.training_args import (
    OptimizerType as OptimizerType,
)
from chipiron.scripts.learn_nn_supervised.training_args import (
    SupervisedTrainingArgs as SupervisedTrainingArgs,
)
from chipiron.scripts.learn_nn_supervised.training_args import (
    SupervisedTrainingConfigError as SupervisedTrainingConfigError,
)
