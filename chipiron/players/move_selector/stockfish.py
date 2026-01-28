"""
This module contains the implementation of the StockfishPlayer class, which i        if self.engine is None:
            # Check if Stockfish binary exists
            if not STOCKFISH_BINARY_PATH.exists():
                raise FileNotFoundError(
                    f"Stockfish binary not found at {STOCKFISH_BINARY_PATH}.\n"
                    f"Please install Stockfish by running:\n"
                    f"    make stockfish\n"
                    f"This will download and install Stockfish to the appropriate location."
                )lector that uses the Stockfish chess engine to recommend moves.

The StockfishPlayer class is a dataclass that represents a player that selects moves using the Stockfish engine. It has the following attributes:
- type: A literal value representing the type of move selector (in this case, MoveSelectorTypes.Stockfish).
- depth: An integer representing the depth to which Stockfish should search for moves (default is 20).
- time_limit: A float representing the time limit (in seconds) for Stockfish to search for moves (default is 0.1).
- engine: An instance of the Stockfish engine.

The StockfishPlayer class has the following methods:
- select_move: Selects a move based on the given board state and move seed. Returns a Recommendation object.
- print_info: Prints the type of move selector.

Note: The Stockfish engine is initialized lazily when the first move is selected, and it is automatically closed after each move is selected.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import atomheart.board as boards
import chess.engine
from atomheart.board import create_board_chi
from atomheart.board.utils import FenPlusHistory
from valanga.policy import Recommendation

from chipiron.environments.chess.types import ChessState
from chipiron.utils.path_variables import STOCKFISH_BINARY_PATH

from .move_selector_types import MoveSelectorTypes

if TYPE_CHECKING:
    from atomheart import BoardChi
from valanga.game import BranchName, Seed


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
        select_move(board: boards.BoardChi, move_seed: int) -> Recommendation:
            Selects a move based on the given board state and move seed.

        print_info() -> None:
            Prints the type of move selector.
    """

    type: Literal[MoveSelectorTypes.Stockfish]  # for serialization
    depth: int = 20
    time_limit: float = 0.1
    engine: Any = None

    @staticmethod
    def is_stockfish_available() -> bool:
        """
        Check if Stockfish is properly installed and available.

        Returns:
            bool: True if Stockfish binary exists and appears to be executable.
        """
        return STOCKFISH_BINARY_PATH.exists() and STOCKFISH_BINARY_PATH.is_file()

    def recommend(self, state: ChessState, seed: Seed) -> Recommendation:
        # seed can be ignored (stockfish is deterministic unless you randomize)
        best: BranchName = self.select_move(
            state.board, move_seed=seed
        ).recommended_name
        return Recommendation(recommended_name=best)

    def select_move(self, board: boards.IBoard, move_seed: int) -> Recommendation:
        """
        Selects a move based on the given board state and move seed.

        Args:
            board (boards.BoardChi): The current board state.
            move_seed (int): The seed for move selection.

        Returns:
            Recommendation: A Recommendation object representing the selected move.

        Raises:
            FileNotFoundError: If Stockfish binary is not found, with instructions to install it.
        """

        if self.engine is None:
            # Check if Stockfish binary exists
            if not STOCKFISH_BINARY_PATH.exists():
                raise FileNotFoundError(
                    f"Stockfish binary not found at {STOCKFISH_BINARY_PATH}.\n"
                    f"Please install Stockfish by running:\n"
                    f"    make stockfish\n"
                    f"This will download and install Stockfish 16 (~40MB) to the correct location."
                )

            try:
                self.engine = chess.engine.SimpleEngine.popen_uci(
                    str(STOCKFISH_BINARY_PATH)
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to start Stockfish engine at {STOCKFISH_BINARY_PATH}.\n"
                    f"The binary may be corrupted. Try reinstalling with:\n"
                    f"    rm -rf {STOCKFISH_BINARY_PATH.parent}\n"
                    f"    make stockfish\n"
                    f"Original error: {e}"
                ) from e

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
        return Recommendation(recommended_name=result.move.uci(), evaluation=None)

    def print_info(self) -> None:
        """
        Prints the type of move selector.
        """
        print(self.type)
