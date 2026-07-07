"""Compatibility tests for supervised chess training config names."""

from __future__ import annotations

from dataclasses import fields
from importlib.resources import files
from pathlib import Path
from typing import get_type_hints

import pytest
import yaml


def test_canonical_and_legacy_training_args_imports_match() -> None:
    """Canonical imports and legacy aliases should resolve to the same objects."""
    from chipiron.learningprocesses.nn_trainer import factory
    from chipiron.scripts.learn_nn_supervised import training_args

    assert training_args.SupervisedTrainingArgs is not None
    assert training_args.NNTrainerArgs is training_args.SupervisedTrainingArgs
    assert (
        training_args.NNTrainerConfigError
        is training_args.SupervisedTrainingConfigError
    )
    assert factory.SupervisedTrainingArgs is training_args.SupervisedTrainingArgs
    assert factory.NNTrainerArgs is training_args.SupervisedTrainingArgs
    assert (
        factory.SupervisedTrainingConfigError
        is training_args.SupervisedTrainingConfigError
    )
    assert factory.NNTrainerConfigError is training_args.SupervisedTrainingConfigError


def test_script_args_keep_public_nn_trainer_args_field_name() -> None:
    """The public script dataclass field should remain YAML-compatible."""
    from chipiron.scripts.learn_from_scratch_value_and_fixed_boards.learn_from_scratch_value_and_fixed_boards import (
        LearnNNFromScratchScriptArgs,
    )
    from chipiron.scripts.learn_nn_supervised.learn_nn_from_supervised_datasets import (
        LearnNNScriptArgs,
    )
    from chipiron.scripts.learn_nn_supervised.training_args import (
        SupervisedTrainingArgs,
    )

    supervised_hints = get_type_hints(LearnNNScriptArgs)
    scratch_hints = get_type_hints(LearnNNFromScratchScriptArgs)
    supervised_fields = {field.name: field for field in fields(LearnNNScriptArgs)}
    scratch_fields = {
        field.name: field for field in fields(LearnNNFromScratchScriptArgs)
    }

    assert "nn_trainer_args" in supervised_fields
    assert "nn_trainer_args" in scratch_fields
    assert supervised_hints["nn_trainer_args"] is SupervisedTrainingArgs
    assert scratch_hints["nn_trainer_args"] is SupervisedTrainingArgs


def test_supervised_test_yaml_key_still_loads_with_parser() -> None:
    """Parser-backed supervised options should still use nn_trainer_args."""
    pytest.importorskip("parsley")
    from parsley import create_parsley

    from chipiron.scripts.learn_nn_supervised.learn_nn_from_supervised_datasets import (
        LearnNNScriptArgs,
    )
    from chipiron.scripts.learn_nn_supervised.training_args import (
        SupervisedTrainingArgs,
    )

    package_root = files("chipiron")
    yaml_path = Path("tests/scripts/learn_nn_supervised/test_exp_options.yaml")

    parser = create_parsley(
        args_dataclass_name=LearnNNScriptArgs,
        should_parse_command_line_arguments=False,
        package_name=str(package_root),
    )
    args: LearnNNScriptArgs = parser.parse_arguments(config_file_path=str(yaml_path))

    assert isinstance(args.nn_trainer_args, SupervisedTrainingArgs)
    assert args.nn_trainer_args.reuse_existing_model is False


@pytest.mark.parametrize(
    "yaml_path",
    [
        Path("src/chipiron/scripts/learn_nn_supervised/exp_options.yaml"),
        Path("tests/scripts/learn_nn_supervised/test_exp_options.yaml"),
        Path(
            "src/chipiron/scripts/learn_from_scratch_value_and_fixed_boards/"
            "exp_options.yaml"
        ),
        Path(
            "tests/scripts/learn_from_scratch_value_and_fixed_boards/"
            "test_exp_options.yaml"
        ),
    ],
)
def test_existing_training_option_files_keep_nn_trainer_args_key(
    yaml_path: Path,
) -> None:
    """Current option files should not migrate the public YAML key in PR22."""
    with yaml_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    assert "nn_trainer_args" in config
    assert "supervised_training_args" not in config
