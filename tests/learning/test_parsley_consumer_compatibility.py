"""Consumer contracts for the published Parsley 0.3 construction semantics."""

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import pytest
import yaml
from dacite import MissingValueError
from parsley import create_parsley, make_partial_dataclass_with_optional_paths

from chipiron.environments.chess.players.evaluators.boardevaluators.datasets.datasets import (
    DataSetArgs,
)
from chipiron.games.domain.game.game_tag import GameConfigTag
from chipiron.games.domain.match.match_args import MatchArgs
from chipiron.games.domain.match.match_settings_args import MatchSettingsArgs
from chipiron.players.player_args import PlayerArgs
from chipiron.players.player_ids import PlayerConfigTag
from chipiron.scripts.gui_launcher.builders import generate_inputs
from chipiron.scripts.gui_launcher.models import ArgsChosenByUser, ParticipantSelection
from chipiron.scripts.learn_nn_supervised.learn_nn_from_supervised_datasets import (
    LearnNNScriptArgs,
)
from chipiron.scripts.learn_nn_supervised.training_args import (
    SupervisedTrainingArgs,
    SupervisedTrainingConfigError,
)
from chipiron.scripts.one_match.one_match import MatchScriptArgs
from chipiron.scripts.script_args import BaseScriptArgs
from chipiron.utils.small_tools import get_package_root_path, resolve_package_path

if TYPE_CHECKING:
    from pathlib import Path


def _parser(schema: Any) -> Any:
    return create_parsley(
        schema, package_name=get_package_root_path("chipiron"), verbosity=50
    )


@pytest.mark.parametrize("tag", list(GameConfigTag))
def test_game_presets_preserve_nullable_move_limit(tag: GameConfigTag) -> None:
    """A declared None default is valid, including direct preset loading."""
    args = tag.get_match_settings_args()
    assert args.max_half_moves is None
    assert args.starting_position.type == "from_file"
    assert args.each_player_has_its_own_thread is (tag is GameConfigTag.NAVO)


def test_chess_launcher_extra_args_resolve_presets() -> None:
    """Keep the established launcher/YAML/extra_args integration intact."""
    _, extra, config = generate_inputs(
        ArgsChosenByUser(
            participants=[
                ParticipantSelection(PlayerConfigTag.RANDOM),
                ParticipantSelection(PlayerConfigTag.GUI_HUMAN),
            ]
        )
    )
    args = _parser(MatchScriptArgs).parse_arguments(
        config_file_path=resolve_package_path(config), extra_args=extra, argv=[]
    )
    assert args.gui is True
    assert isinstance(args.match_args.player_one, PlayerArgs)
    assert isinstance(args.match_args.player_two, PlayerArgs)
    assert isinstance(args.match_args.match_setting, MatchSettingsArgs)
    assert args.base_script_args.seed == 0


def test_match_type_factory_resolves_nested_default_presets() -> None:
    """The dataclass type factory defers construction until presets resolve."""
    args = _parser(MatchScriptArgs).parse_arguments(argv=[])
    assert isinstance(args.match_args.player_one, PlayerArgs)
    assert isinstance(args.match_args.match_setting, MatchSettingsArgs)


@dataclass
class _ConstructedMatch:
    """Model the unsupported custom-factory form for the real match schema."""

    match: MatchArgs = field(default_factory=lambda: MatchArgs())


def test_custom_factory_must_not_return_unresolved_match_presets() -> None:
    """Reject unresolved constructed values rather than reconstructing them."""
    with pytest.raises(ValueError, match="unresolved YAML preset"):
        _parser(_ConstructedMatch).parse_arguments(argv=[])


def test_scientific_factory_defaults_are_unchanged_and_isolated() -> None:
    """Omitting an entire field retains the declared chess scientific recipe."""
    parser = _parser(LearnNNScriptArgs)
    first = parser.parse_arguments(argv=[])
    second = parser.parse_arguments(argv=[])
    assert asdict(first.nn_trainer_args) == asdict(SupervisedTrainingArgs())
    assert first.dataset_args == LearnNNScriptArgs().dataset_args
    assert first.nn_trainer_args is not second.nn_trainer_args
    assert (
        first.nn_trainer_args.neural_network_architecture_args
        is not second.nn_trainer_args.neural_network_architecture_args
    )


@pytest.mark.parametrize("source", ["cli", "yaml", "extra"])
def test_explicit_dataset_preserves_scientific_selection(
    source: str, tmp_path: Path
) -> None:
    """Complete input preserves both dataset choices through every public route."""
    dataset = DataSetArgs("train.pkl", "validation.pkl", True)
    parser = _parser(LearnNNScriptArgs)
    if source == "cli":
        args = parser.parse_arguments(
            argv=[
                "--dataset_args.train_file_name",
                "train.pkl",
                "--dataset_args.test_file_name",
                "validation.pkl",
                "--dataset_args.preprocessing_data_set",
                "true",
            ],
            strict=True,
        )
    elif source == "yaml":
        config = tmp_path / "options.yaml"
        config.write_text(yaml.safe_dump({"dataset_args": asdict(dataset)}))
        args = parser.parse_arguments(config_file_path=str(config), argv=[])
    else:
        partial = make_partial_dataclass_with_optional_paths(LearnNNScriptArgs)
        args = parser.parse_arguments(extra_args=partial(dataset_args=dataset), argv=[])
    assert args.dataset_args == dataset


def test_partial_dataset_requires_its_own_required_filename() -> None:
    """A supplied nested mapping cannot silently borrow a parent factory path."""
    with pytest.raises(MissingValueError, match=r"dataset_args\.train_file_name"):
        _parser(LearnNNScriptArgs).parse_arguments(
            argv=["--dataset_args.preprocessing_data_set", "true"]
        )


def test_explicit_null_dataset_is_preserved(tmp_path: Path) -> None:
    """An intentional absent validation dataset remains None, not a factory path."""
    config = tmp_path / "options.yaml"
    config.write_text(
        "dataset_args:\n  train_file_name: train.pkl\n  test_file_name: null\n"
    )
    args = _parser(LearnNNScriptArgs).parse_arguments(
        config_file_path=str(config), argv=[]
    )
    assert args.dataset_args == DataSetArgs("train.pkl", None, False)


def test_general_script_source_precedence(tmp_path: Path) -> None:
    """Programmatic overrides win over CLI, YAML and declared defaults."""
    config = tmp_path / "options.yaml"
    config.write_text("seed: 11\nprofiling: true\n")
    partial = make_partial_dataclass_with_optional_paths(BaseScriptArgs)
    args = _parser(BaseScriptArgs).parse_arguments(
        config_file_path=str(config),
        extra_args=partial(seed=23),
        argv=["--seed", "19", "--profiling", "false"],
        strict=True,
    )
    assert args.seed == 23
    assert args.profiling is False
    assert args.logging_levels == BaseScriptArgs().logging_levels


def test_training_validation_runs_after_sources_are_composed(tmp_path: Path) -> None:
    """Reuse validation sees the final parameter path, retaining its error type."""
    parser = _parser(LearnNNScriptArgs)
    with pytest.raises(SupervisedTrainingConfigError):
        parser.parse_arguments(argv=["--nn_trainer_args.reuse_existing_model", "true"])
    config = tmp_path / "options.yaml"
    config.write_text("nn_trainer_args:\n  reuse_existing_model: true\n")
    args = parser.parse_arguments(
        config_file_path=str(config),
        argv=[
            "--nn_trainer_args.nn_parameters_file_if_reusing_existing_one",
            "weights.pt",
        ],
    )
    assert args.nn_trainer_args.reuse_existing_model is True
    assert (
        args.nn_trainer_args.nn_parameters_file_if_reusing_existing_one == "weights.pt"
    )
