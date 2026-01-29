from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, TypeGuard, cast

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
from chipiron.players.boardevaluators.neural_networks.input_converters.ModelInputRepresentationType import (
    ModelInputRepresentationType,
)
from chipiron.utils import path
from chipiron.utils.small_tools import resolve_package_path

CHIPIRON_NN_ARGS_FILENAME = "chipiron_nn.yaml"


@dataclass(frozen=True, slots=True)
class ChipironNNArgs:
    version: int = 1
    game_kind: GameKind = GameKind.CHESS
    input_representation: str = "piece_difference"


ContentToInputBuilder = Callable[[str], ContentToInputFunction[ChessState]]


def get_chipiron_nn_args_file_path_from(folder_path: path) -> str:
    return os.path.join(folder_path, CHIPIRON_NN_ARGS_FILENAME)


def _serialize_chipiron_nn_args(args: ChipironNNArgs) -> dict[str, int | str]:
    return {
        "version": args.version,
        "game_kind": args.game_kind.value,
        "input_representation": args.input_representation,
    }


def save_chipiron_nn_args(args: ChipironNNArgs, folder_path: path) -> None:
    file_path = get_chipiron_nn_args_file_path_from(folder_path)
    with open(file_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(_serialize_chipiron_nn_args(args), handle)


def _is_str_key_mapping(obj: object) -> TypeGuard[Mapping[str, Any]]:
    if not isinstance(obj, Mapping):
        return False

    m = cast(
        "Mapping[object, object]", obj
    )  # key/value types become object (not Unknown)
    return all(isinstance(k, str) for k in m.keys())


def _as_mapping(obj: object, *, file_path: str) -> Mapping[str, Any]:
    if not _is_str_key_mapping(obj):
        raise ValueError(
            f"Invalid chipiron NN args in {file_path!r}: expected mapping with string keys"
        )
    return obj


def load_chipiron_nn_args(folder_path: path) -> ChipironNNArgs:
    file_path = get_chipiron_nn_args_file_path_from(folder_path)
    with open(file_path, "r", encoding="utf-8") as f:
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
    representation = ModelInputRepresentationType(input_representation)
    return create_chess_state_to_input(representation)


_CONTENT_TO_INPUT_BUILDERS: dict[GameKind, ContentToInputBuilder] = {
    GameKind.CHESS: _create_chess_content_to_input,
}


def create_content_to_input_convert(
    chipiron_nn_args: ChipironNNArgs,
) -> ContentToInputFunction[ChessState]:
    if chipiron_nn_args.version != 1:
        raise ValueError(
            f"Unsupported chipiron NN args version: {chipiron_nn_args.version}."
        )
    if chipiron_nn_args.game_kind is not GameKind.CHESS:
        raise ValueError(
            f"Unsupported game_kind for now: {chipiron_nn_args.game_kind!r}."
        )
    builder = _CONTENT_TO_INPUT_BUILDERS[chipiron_nn_args.game_kind]
    return builder(chipiron_nn_args.input_representation)


def create_content_to_input_from_folder(
    folder_path: path,
) -> ContentToInputFunction[ChessState]:
    chipiron_nn_args = load_chipiron_nn_args(folder_path)
    return create_content_to_input_convert(chipiron_nn_args)


def create_content_to_input_from_model_weights(
    model_weights_file_name: path,
) -> ContentToInputFunction[ChessState]:
    model_weights_file_name = resolve_package_path(str(model_weights_file_name))
    folder_path = os.path.dirname(model_weights_file_name)
    return create_content_to_input_from_folder(folder_path)
