"""
This module contains the implementation of the CommandLineHumanMoveSelector class, which allows a human player
 to select moves through the command line interface.
"""

from dataclasses import dataclass
from typing import Literal

import chipiron.environments.chess.board as boards
from chipiron.environments.chess.move import moveUci
from chipiron.environments.chess.move.imove import moveKey
from chipiron.utils import seed

from .move_selector import MoveRecommendation
from .move_selector_types import MoveSelectorTypes


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
        print(f"Legal Moves {board.legal_moves}")
        legal_moves_uci: list[moveUci] = []
        move: moveKey
        for move in board.legal_moves:
            legal_moves_uci.append(board.get_uci_from_move_key(move))
        print(f"Legal Moves {legal_moves_uci}")

        good_move: bool = False
        while not good_move:
            move_uci: str = input("Input your move")
            print(f"choice of move {move_uci}")
            if move_uci in legal_moves_uci:
                good_move = True
            else:
                print("Bad move, Not legal")

        assert move_uci in legal_moves_uci

        return MoveRecommendation(move=move_uci, evaluation=None)
