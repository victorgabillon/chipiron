"""
This module contains the implementation of the CommandLineHumanMoveSelector class, which allows a human player
 to select moves through the command line interface.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import chipiron.environments.chess_env.board as boards
from chipiron.utils import seed
from chipiron.utils.logger import chipiron_logger

from .move_selector import MoveRecommendation
from .move_selector_types import MoveSelectorTypes

if TYPE_CHECKING:
    from chipiron.environments.chess_env.move import moveUci
    from chipiron.environments.chess_env.move.imove import moveKey


@dataclass
class CommandLineHumanPlayerArgs:
    """
    Represents the arguments for a human player that selects moves through the command line interface.
    """

    type: Literal[MoveSelectorTypes.CommandLineHuman]  # for serialization


@dataclass
class GuiHumanPlayerArgs:
    """
    Represents the arguments for a human player that selects moves through the GUI.
    """

    type: Literal[MoveSelectorTypes.GuiHuman]  # for serialization


class CommandLineHumanMoveSelector:
    """
    A move selector that allows a human player to select moves through the command line interface.
    """

    def select_move(self, board: boards.IBoard, move_seed: seed) -> MoveRecommendation:
        """
        Selects a move based on user input through the command line interface.

        Args:
            board (boards.BoardChi): The current state of the chess board.
            move_seed (seed): The seed used for move selection.

        Returns:
            MoveRecommendation: The selected move recommendation.

        Raises:
            AssertionError: If the selected move is not a legal move.
        """
        chipiron_logger.info("Legal Moves %s", board.legal_moves)
        legal_moves_uci: list[moveUci] = []
        move: moveKey
        for move in board.legal_moves:
            legal_moves_uci.append(board.get_uci_from_move_key(move))
        chipiron_logger.info("Legal Moves %s", legal_moves_uci)

        good_move: bool = False
        move_uci: str = ""
        while not good_move:
            move_uci = input("Input your move")
            print(f"choice of move {move_uci}")
            if move_uci in legal_moves_uci:
                good_move = True
            else:
                print("Bad move, Not legal")

        assert move_uci in legal_moves_uci

        return MoveRecommendation(move=move_uci, evaluation=None)
