"""Chipiron-specific neural network wiring arguments."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import yaml
from coral.neural_networks.input_converters.content_to_input import (
    ContentToInputFunction,
)

from chipiron.environments.types import GameKind
from chipiron.players.boardevaluators.neural_networks.input_converters.ModelInputRepresentationType import (
    ModelInputRepresentationType,
)
from chipiron.players.boardevaluators.neural_networks.input_converters.board_to_input import (
    create_board_to_input,
)
from chipiron.utils import path


CHIPIRON_NN_ARGS_FILENAME = "chipiron_nn.yaml"


@dataclass(frozen=True, slots=True)
class ChipironNNArgs:
    version: int
    game_kind: GameKind
    input_representation: str


ContentToInputBuilder = Callable[[str], ContentToInputFunction]


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


def load_chipiron_nn_args(folder_path: path) -> ChipironNNArgs:
    file_path = get_chipiron_nn_args_file_path_from(folder_path)
    with open(file_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid chipiron NN args in {file_path!r}.")
    try:
        version_raw = payload.get("version", 1)
        version = int(version_raw)
        game_kind = GameKind(str(payload["game_kind"]))
        input_representation = str(payload["input_representation"])
    except KeyError as exc:
        raise ValueError(
            f"Missing {exc.args[0]!r} in chipiron NN args at {file_path!r}."
        ) from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid chipiron NN args values in {file_path!r}."
        ) from exc
    return ChipironNNArgs(
        version=version,
        game_kind=game_kind,
        input_representation=input_representation,
    )


def _create_chess_content_to_input(
    input_representation: str,
) -> ContentToInputFunction:
    representation = ModelInputRepresentationType(input_representation)
    return create_board_to_input(representation)


def _create_checkers_content_to_input(
    input_representation: str,
) -> ContentToInputFunction:
    raise ValueError(
        f"Checkers input conversion is not implemented: {input_representation}."
    )


_CONTENT_TO_INPUT_BUILDERS: dict[GameKind, ContentToInputBuilder] = {
    GameKind.CHESS: _create_chess_content_to_input,
    GameKind.CHECKERS: _create_checkers_content_to_input,
}


def create_content_to_input_convert(
    chipiron_nn_args: ChipironNNArgs,
) -> ContentToInputFunction:
    if chipiron_nn_args.version != 1:
        raise ValueError(
            f"Unsupported chipiron NN args version: {chipiron_nn_args.version}."
        )
    try:
        builder = _CONTENT_TO_INPUT_BUILDERS[chipiron_nn_args.game_kind]
    except KeyError as exc:
        raise ValueError(
            f"No content-to-input builder for {chipiron_nn_args.game_kind!r}."
        ) from exc
    return builder(chipiron_nn_args.input_representation)


def create_content_to_input_from_folder(folder_path: path) -> ContentToInputFunction:
    chipiron_nn_args = load_chipiron_nn_args(folder_path)
    return create_content_to_input_convert(chipiron_nn_args)


def create_content_to_input_from_model_weights(
    model_weights_file_name: path,
) -> ContentToInputFunction:
    folder_path = os.path.dirname(model_weights_file_name)
    return create_content_to_input_from_folder(folder_path)
