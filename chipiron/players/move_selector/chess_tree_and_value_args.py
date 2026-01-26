"""Chess-specific tree-and-value args alias."""

from __future__ import annotations

from typing import TypeAlias

from chipiron.environments.chess.types import ChessState
from chipiron.players.boardevaluators.master_board_evaluator import (
    MasterBoardEvaluatorArgs,
)

from .tree_and_value_args import TreeAndValueAppArgs

TreeAndValueChipironArgs: TypeAlias = TreeAndValueAppArgs[
    ChessState, MasterBoardEvaluatorArgs
]
