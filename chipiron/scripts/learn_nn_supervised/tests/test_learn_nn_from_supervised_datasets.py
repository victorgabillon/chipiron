"""Module for test learn nn from supervised datasets."""

from typing import Any

from parsley_coco import make_partial_dataclass_with_optional_paths

import chipiron.scripts as scripts
from chipiron.environments.types import GameKind
from chipiron.learningprocesses.nn_trainer.factory import GameInputArgs, NNTrainerArgs
from chipiron.players.boardevaluators.datasets.datasets import DataSetArgs
from chipiron.players.boardevaluators.neural_networks.input_converters.model_input_representation_type import (
    ModelInputRepresentationType,
)
from chipiron.scripts.factory import create_script
from chipiron.scripts.learn_nn_supervised.learn_nn_from_supervised_datasets import (
    LearnNNScriptArgs,
)
from chipiron.scripts.script_args import BaseScriptArgs

PartialOpLearnNNScriptArgs = make_partial_dataclass_with_optional_paths(
    cls=LearnNNScriptArgs
)
PartialOpNNTrainerArgs = make_partial_dataclass_with_optional_paths(cls=NNTrainerArgs)
PartialOpDataSetArgs = make_partial_dataclass_with_optional_paths(cls=DataSetArgs)
PartialOpBaseScriptArgs = make_partial_dataclass_with_optional_paths(cls=BaseScriptArgs)


ARCH_TO_REP = {
    "architecture_p1.yaml": ModelInputRepresentationType.PIECE_DIFFERENCE,
    "architecture_prelu_nobug.yaml": ModelInputRepresentationType.NOBUG364,
    "architecture_transformerone.yaml": ModelInputRepresentationType.PIECE_MAP,
}


PartialOpGameInputArgs = make_partial_dataclass_with_optional_paths(cls=GameInputArgs)


configs_dataclasses: list[Any] = [
    PartialOpLearnNNScriptArgs(
        nn_trainer_args=PartialOpNNTrainerArgs(
            reuse_existing_model=False,
            specific_saving_folder="chipiron/scripts/learn_nn_supervised/tests/tests_outputs",
            neural_network_architecture_args_path_to_yaml_file="chipiron/scripts/learn_nn_supervised/"
            "board_evaluators_common_training_data/nn_pytorch/architectures/"
            + architecture_file,
            game_input=PartialOpGameInputArgs(
                game_kind=GameKind.CHESS,
                representation=ARCH_TO_REP[architecture_file],
            ),
        ),
        dataset_args=PartialOpDataSetArgs(
            train_file_name="chipiron/scripts/learn_nn_supervised/tests/small_dataset.pi",
            test_file_name="chipiron/scripts/learn_nn_supervised/tests/small_dataset.pi",
        ),
        base_script_args=PartialOpBaseScriptArgs(testing=True),
    )
    for architecture_file in [
        "architecture_p1.yaml",
        "architecture_prelu_nobug.yaml",
        "architecture_transformerone.yaml",
    ]
]


def test_learn_nn() -> None:
    """Test learn nn."""
    for config in configs_dataclasses:
        script_object: scripts.IScript = create_script(
            script_type=scripts.ScriptType.LEARN_NN,
            extra_args=config,
            should_parse_command_line_arguments=False,
        )

        # run the script
        script_object.run()

        # terminate the script
        script_object.terminate()


if __name__ == "__main__":
    test_learn_nn()
