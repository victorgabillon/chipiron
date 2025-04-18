"""
This module contains the implementation of the StockfishPlayer class, which is a move selector that uses the Stockfish chess engine to recommend moves.

The StockfishPlayer class is a dataclass that represents a player that selects moves using the Stockfish engine. It has the following attributes:
- type: A literal value representing the type of move selector (in this case, MoveSelectorTypes.Stockfish).
- depth: An integer representing the depth to which Stockfish should search for moves (default is 20).
- time_limit: A float representing the time limit (in seconds) for Stockfish to search for moves (default is 0.1).
- engine: An instance of the Stockfish engine.

The StockfishPlayer class has the following methods:
- select_move: Selects a move based on the given board state and move seed. Returns a MoveRecommendation object.
- print_info: Prints the type of move selector.

Note: The Stockfish engine is initialized lazily when the first move is selected, and it is automatically closed after each move is selected.
"""

from dataclasses import dataclass
from typing import Any, Literal

import chess.engine

import chipiron.environments.chess.board as boards

from ...environments.chess import BoardChi
from ...environments.chess.board import create_board_chi
from ...environments.chess.board.utils import FenPlusHistory
from .move_selector import MoveRecommendation
from .move_selector_types import MoveSelectorTypes


@dataclass
class StockfishPlayer:
    """
    A player that selects moves using the Stockfish chess engine.

    Attributes:
        type (Literal[MoveSelectorTypes.Stockfish]): The type of move selector (for serialization).
        depth (int): The depth to which Stockfish should search for moves.
        time_limit (float): The time limit (in seconds) for Stockfish to search for moves.
        engine (Any): The Stockfish chess engine instance.

    Methods:
        select_move(board: boards.BoardChi, move_seed: int) -> MoveRecommendation:
            Selects a move based on the given board state and move seed.

        print_info() -> None:
            Prints the type of move selector.
    """

    type: Literal[MoveSelectorTypes.Stockfish]  # for serialization
    depth: int = 20
    time_limit: float = 0.1
    engine: Any = None

    def select_move(self, board: boards.IBoard, move_seed: int) -> MoveRecommendation:
        """
        Selects a move based on the given board state and move seed.

        Args:
            board (boards.BoardChi): The current board state.
            move_seed (int): The seed for move selection.

        Returns:
            MoveRecommendation: A MoveRecommendation object representing the selected move.
        """

        if self.engine is None:
            # if this object is created in the init then sending the object
            # self.engine = chess.engine.SimpleEngine.popen_uci(
            #    # TODO: should we remove the hardcoding
            #    r"stockfish/stockfish_14.1_linux_x64/stockfish_14.1_linux_x64")
            self.engine = chess.engine.SimpleEngine.popen_uci(
                # TODO: should we remove the hardcoding
                r"stockfish/stockfish/stockfish-ubuntu-x86-64-avx2"
            )

        # transform the board
        board_chi: BoardChi = create_board_chi(
            fen_with_history=FenPlusHistory(
                current_fen=board.fen,
                historical_moves=board.move_history_stack,
                # note that we do not give here historical_boards, hope this does not create but related to 3 fold repetition computation
            )
        )
        result = self.engine.play(
            board_chi.chess_board, chess.engine.Limit(self.time_limit)
        )

        self.engine.quit()
        self.engine = None
        return MoveRecommendation(move=result.move.uci())

    def print_info(self) -> None:
        """
        Prints the type of move selector.
        """
        print(self.type)
