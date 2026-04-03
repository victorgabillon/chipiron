"""Module for test learn nn from scratch and fixed boards."""
# pylint: disable=duplicate-code

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("parsley")
pytest.importorskip("torch")
pytest.importorskip("PySide6")

from parsley import make_partial_dataclass_with_optional_paths

from chipiron import scripts
from chipiron.environments.chess.players.evaluators.boardevaluators.datasets.datasets import (
    DataSetArgs,
)
from chipiron.learningprocesses.nn_trainer.factory import NNTrainerArgs
from chipiron.players.player_ids import PlayerConfigTag
from chipiron.scripts.factory import create_script
from chipiron.scripts.learn_from_scratch_value_and_fixed_boards.learn_from_scratch_value_and_fixed_boards import (
    LearnNNFromScratchScriptArgs,
)
from chipiron.scripts.script_args import BaseScriptArgs
from tests.model_bundle_test_utils import create_tiny_model_bundle

PartialOpLearnNNFromScratchScriptArgs = make_partial_dataclass_with_optional_paths(
    cls=LearnNNFromScratchScriptArgs
)
PartialOpNNTrainerArgs = make_partial_dataclass_with_optional_paths(cls=NNTrainerArgs)
PartialOpDataSetArgs = make_partial_dataclass_with_optional_paths(cls=DataSetArgs)
PartialOpBaseScriptArgs = make_partial_dataclass_with_optional_paths(cls=BaseScriptArgs)


def _make_config(*, tmp_path: Path) -> Any:
    """Build a local-only config for the scratch learning test."""
    bundle_dir = create_tiny_model_bundle(
        tmp_path,
        bundle_name="scratch_model_bundle",
        input_representation="piece_difference",
    )
    out_dir = tmp_path / "learn_from_scratch_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_file = str((Path(__file__).parent / "small_dataset.pi").resolve())

    return PartialOpLearnNNFromScratchScriptArgs(
        nn_trainer_args=PartialOpNNTrainerArgs(
            reuse_existing_model=False,
            specific_saving_folder=str(out_dir),
            neural_network_architecture_args_path_to_yaml_file=str(
                bundle_dir / "architecture.yaml"
            ),
        ),
        dataset_args=PartialOpDataSetArgs(
            train_file_name=dataset_file,
            test_file_name=dataset_file,
        ),
        evaluating_player_args=PlayerConfigTag.RANDOM,
        base_script_args=PartialOpBaseScriptArgs(testing=True),
    )


def test_learn_nn_from_scratch_and_fixed_boards(tmp_path: Path) -> None:
    """Test learn nn from scratch and fixed boards."""
    config = _make_config(tmp_path=tmp_path)

    script_object: scripts.IScript = create_script(
        script_type=scripts.ScriptType.LEARN_NN_FROM_SCRATCH,
        extra_args=config,
        should_parse_command_line_arguments=False,
    )

    script_object.run()
    script_object.terminate()
