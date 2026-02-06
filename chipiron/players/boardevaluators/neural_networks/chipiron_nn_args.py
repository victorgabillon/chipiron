"""Module for chipiron nn args."""

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeGuard, cast

import yaml
from coral.neural_networks.input_converters.content_to_input import (
    ContentToInputFunction,
)
from dacite import Config, from_dict

from chipiron.environments.chess.types import ChessState
from chipiron.environments.types import GameKind
from chipiron.players.boardevaluators.neural_networks.input_converters.board_to_input import (
    create_chess_state_to_input,
)
from chipiron.players.boardevaluators.neural_networks.input_converters.model_input_representation_type import (
    ModelInputRepresentationType,
)
from chipiron.utils import path
from chipiron.utils.small_tools import resolve_package_path

CHIPIRON_NN_ARGS_FILENAME = "chipiron_nn.yaml"


class ChipironNNArgsError(ValueError):
    """Base error for chipiron neural network arguments."""


class InvalidChipironNNArgsError(ChipironNNArgsError):
    """Raised when NN args are not structured as expected."""

    def __init__(self, file_path: str) -> None:
        """Initialize the error with the invalid file path."""
        super().__init__(
            f"Invalid chipiron NN args in {file_path!r}: expected mapping with string keys"
        )


class UnsupportedChipironNNArgsVersionError(ChipironNNArgsError):
    """Raised when the NN args version is unsupported."""

    def __init__(self, version: int) -> None:
        """Initialize the error with the unsupported version."""
        super().__init__(f"Unsupported chipiron NN args version: {version}.")


class UnsupportedChipironNNGameKindError(ChipironNNArgsError):
    """Raised when the NN args specify an unsupported game kind."""

    def __init__(self, game_kind: GameKind) -> None:
        """Initialize the error with the unsupported game kind."""
        super().__init__(f"Unsupported game_kind for now: {game_kind!r}.")


@dataclass(frozen=True, slots=True)
class ChipironNNArgs:
    """Chipironnnargs implementation."""

    version: int = 1
    game_kind: GameKind = GameKind.CHESS
    input_representation: str = "piece_difference"


ContentToInputBuilder = Callable[[str], ContentToInputFunction[ChessState]]


def get_chipiron_nn_args_file_path_from(folder_path: path) -> str:
    """Return chipiron nn args file path from."""
    return os.path.join(folder_path, CHIPIRON_NN_ARGS_FILENAME)


def _serialize_chipiron_nn_args(args: ChipironNNArgs) -> dict[str, int | str]:
    """Serialize NN args into primitive values for YAML output."""
    return {
        "version": args.version,
        "game_kind": args.game_kind.value,
        "input_representation": args.input_representation,
    }


def save_chipiron_nn_args(args: ChipironNNArgs, folder_path: path) -> None:
    """Save chipiron nn args."""
    file_path = get_chipiron_nn_args_file_path_from(folder_path)
    with open(file_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(_serialize_chipiron_nn_args(args), handle)


def _is_str_key_mapping(obj: object) -> TypeGuard[Mapping[str, Any]]:
    """Return whether the object is a mapping with string keys."""
    if not isinstance(obj, Mapping):
        return False

    m = cast(
        "Mapping[object, object]", obj
    )  # key/value types become object (not Unknown)
    return all(isinstance(k, str) for k in m)


def _as_mapping(obj: object, *, file_path: str) -> Mapping[str, Any]:
    """Validate and return a string-keyed mapping for YAML data."""
    if not _is_str_key_mapping(obj):
        raise InvalidChipironNNArgsError(file_path)
    return obj


def load_chipiron_nn_args(folder_path: path) -> ChipironNNArgs:
    """Load chipiron nn args."""
    file_path = get_chipiron_nn_args_file_path_from(folder_path)
    with open(file_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    data = _as_mapping(raw, file_path=file_path)

    return from_dict(
        data_class=ChipironNNArgs,
        data=data,
        config=Config(
            cast=[int],
            type_hooks={
                GameKind: lambda x: GameKind(str(x)),
            },
            strict=False,
        ),
    )


def _create_chess_content_to_input(
    input_representation: str,
) -> ContentToInputFunction[ChessState]:
    """Build a chess content-to-input converter for the representation."""
    representation = ModelInputRepresentationType(input_representation)
    return create_chess_state_to_input(representation)


_CONTENT_TO_INPUT_BUILDERS: dict[GameKind, ContentToInputBuilder] = {
    GameKind.CHESS: _create_chess_content_to_input,
}


def create_content_to_input_convert(
    chipiron_nn_args: ChipironNNArgs,
) -> ContentToInputFunction[ChessState]:
    """Create content to input convert."""
    if chipiron_nn_args.version != 1:
        raise UnsupportedChipironNNArgsVersionError(chipiron_nn_args.version)
    if chipiron_nn_args.game_kind is not GameKind.CHESS:
        raise UnsupportedChipironNNGameKindError(chipiron_nn_args.game_kind)
    builder = _CONTENT_TO_INPUT_BUILDERS[chipiron_nn_args.game_kind]
    return builder(chipiron_nn_args.input_representation)


def create_content_to_input_from_folder(
    folder_path: path,
) -> ContentToInputFunction[ChessState]:
    """Create content to input from folder."""
    chipiron_nn_args = load_chipiron_nn_args(folder_path)
    return create_content_to_input_convert(chipiron_nn_args)


def create_content_to_input_from_model_weights(
    model_weights_file_name: path,
) -> ContentToInputFunction[ChessState]:
    """Create content to input from model weights."""
    model_weights_file_name = resolve_package_path(str(model_weights_file_name))
    folder_path = os.path.dirname(model_weights_file_name)
    return create_content_to_input_from_folder(folder_path)
