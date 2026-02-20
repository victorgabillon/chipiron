from __future__ import annotations

from pathlib import Path
from typing import Any

from parsley import make_partial_dataclass_with_optional_paths

from chipiron import scripts
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
PartialOpGameInputArgs = make_partial_dataclass_with_optional_paths(cls=GameInputArgs)

ARCH_TO_REP = {
    "architecture_p1.yaml": ModelInputRepresentationType.PIECE_DIFFERENCE,
    "architecture_prelu_nobug.yaml": ModelInputRepresentationType.NOBUG364,
    "architecture_transformerone.yaml": ModelInputRepresentationType.PIECE_MAP,
}


def _make_configs(*, saving_root: Path) -> list[Any]:
    """Build configs, writing outputs under saving_root."""
    dataset_file = str((Path(__file__).parent / "small_dataset.pi").resolve())
    return [
        PartialOpLearnNNScriptArgs(
            nn_trainer_args=PartialOpNNTrainerArgs(
                reuse_existing_model=False,
                # ✅ writable output per test run
                specific_saving_folder=str(
                    saving_root / architecture_file.removesuffix(".yaml")
                ),
                # ✅ read-only packaged resources are fine
                neural_network_architecture_args_path_to_yaml_file=(
                    "package://scripts/learn_nn_supervised/"
                    "board_evaluators_common_training_data/nn_pytorch/architectures/"
                    + architecture_file
                ),
                game_input=PartialOpGameInputArgs(
                    game_kind=GameKind.CHESS,
                    representation=ARCH_TO_REP[architecture_file],
                ),
            ),
            dataset_args=PartialOpDataSetArgs(
                train_file_name=dataset_file,
                test_file_name=dataset_file,
            ),
            base_script_args=PartialOpBaseScriptArgs(testing=True),
        )
        for architecture_file in [
            "architecture_p1.yaml",
            "architecture_prelu_nobug.yaml",
            "architecture_transformerone.yaml",
        ]
    ]


def test_learn_nn(tmp_path: Path) -> None:
    """Test learn nn."""
    saving_root = tmp_path / "learn_nn_supervised_outputs"
    saving_root.mkdir(parents=True, exist_ok=True)

    for config in _make_configs(saving_root=saving_root):
        script_object: scripts.IScript = create_script(
            script_type=scripts.ScriptType.LEARN_NN,
            extra_args=config,
            should_parse_command_line_arguments=False,
        )
        script_object.run()
        script_object.terminate()
