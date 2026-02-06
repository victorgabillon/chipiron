"""Document the module contains the implementation of the CommandLineHumanMoveSelector class, which allows a human player.

to select moves through the command line interface.
"""

from dataclasses import dataclass
from typing import Literal

import atomheart.board as boards
from valanga.game import BranchName, Seed
from valanga.policy import NotifyProgressCallable, Recommendation

from chipiron.environments.chess.types import ChessState
from chipiron.utils.logger import chipiron_logger

from .move_selector_types import MoveSelectorTypes


@dataclass
class CommandLineHumanPlayerArgs:
    """Represents the arguments for a human player that selects moves through the command line interface."""

    type: Literal[MoveSelectorTypes.CommandLineHuman]  # for serialization


@dataclass
class GuiHumanPlayerArgs:
    """Represents the arguments for a human player that selects moves through the GUI."""

    type: Literal[MoveSelectorTypes.GuiHuman]  # for serialization


class CommandLineHumanMoveSelector:
    """A move selector that allows a human player to select moves through the command line interface."""

    def recommend(
        self,
        state: ChessState,
        seed: Seed,
        notify_progress: NotifyProgressCallable | None = None,
    ) -> Recommendation:
        # seed can be ignored (stockfish is deterministic unless you randomize)
        """Recommend."""
        best: BranchName = self.select_move(
            state.board, move_seed=seed, notify_progress=notify_progress
        ).recommended_name
        return Recommendation(recommended_name=best)

    def select_move(
        self,
        board: boards.IBoard,
        move_seed: Seed,
        notify_progress: NotifyProgressCallable | None = None,
    ) -> Recommendation:
        """Select a move based on user input through the command line interface.

        Args:
            board (boards.BoardChi): The current state of the chess board.
            move_seed (seed): The seed used for move selection.

        Returns:
            Recommendation: The selected move recommendation.

        Raises:
            AssertionError: If the selected move is not a legal move.

        """
        chipiron_logger.info("Legal Moves %s", board.legal_moves)
        legal_moves_uci = [
            board.get_uci_from_move_key(move) for move in board.legal_moves
        ]
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

        return Recommendation(recommended_name=move_uci, evaluation=None)
