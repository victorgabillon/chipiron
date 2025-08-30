"""
Module where we define the Stockfish Board Evaluator
"""

from dataclasses import dataclass, field
from typing import Literal

import chess.engine

import chipiron.environments.chess_env.board as boards
from chipiron.environments.chess_env.board.board_chi import BoardChi
from chipiron.environments.chess_env.board.factory import create_board_chi
from chipiron.environments.chess_env.board.utils import FenPlusHistory
from chipiron.players.boardevaluators.board_evaluator_type import BoardEvalTypes
from chipiron.utils.logger import chipiron_logger


@dataclass
class StockfishBoardEvalArgs:
    """Represents the arguments for the Stockfish board evaluator.

    Attributes:
        depth (int): The depth of the search algorithm.
        time_limit (float): The time limit for the search algorithm.
    """

    # for deserialization
    type: Literal[BoardEvalTypes.STOCKFISH_BOARD_EVAL] = field(init=False)

    depth: int
    time_limit: float

    def __post_init__(self) -> None:
        """
        Post-initialization method for the dataclass.
        Automatically sets the 'type' attribute to BoardEvalTypes.STOCKFISH_BOARD_EVAL
        to avoid manual assignment and to facilitate discrimination during deserialization.
        """

        # to avoid having to set the type mannually in the code as it is obvious and only used for discrimanating in deserialization
        object.__setattr__(self, "type", BoardEvalTypes.STOCKFISH_BOARD_EVAL)


class StockfishBoardEvaluator:
    """
    A board evaluator powered by stockfish
    """

    engine: chess.engine.SimpleEngine | None

    def __init__(self, args: StockfishBoardEvalArgs) -> None:
        """
        Initializes a StockfishBoardEvaluator object.

        Args:
            args (StockfishBoardEvalArgs): The arguments for the StockfishBoardEvaluator.

        Returns:
            None
        """
        self.engine = None

    def value_white(self, board: boards.IBoard) -> float:
        """
        Computes the value of the board for the white player.

        Args:
            board (boards.BoardChi): The board object representing the current state of the game.

        Returns:
            float: The value of the board for the white player.
        """
        try:
            if self.engine is None:
                # Try multiple possible Stockfish paths
                stockfish_paths = [
                    "/usr/games/stockfish",
                    "stockfish",
                    r"stockfish/stockfish/stockfish-ubuntu-x86-64-avx2",
                ]

                for path in stockfish_paths:
                    try:
                        self.engine = chess.engine.SimpleEngine.popen_uci(path)
                        break
                    except Exception:
                        continue

                if self.engine is None:
                    chipiron_logger.error(
                        "Stockfish binary not found in any of the expected locations."
                    )
                    return 0.0  # Fallback if no Stockfish found

            # Use the board's chess_board directly to avoid state inconsistencies
            # transform the board
            board_chi: BoardChi = create_board_chi(
                fen_with_history=FenPlusHistory(
                    current_fen=board.fen,
                    # historical_moves=board.move_history_stack,
                    # historical_boards=board.board_history_stack,  # type: ignore
                    # note that we do not give here historical_boards, hope this does not create but related to 3 fold repetition computation
                )
            )

            # Test both approaches to show the difference
            print(f"DEBUG: Original board FEN: {board.fen}")
            print(f"DEBUG: Original board move history: {board.move_history_stack}")

            print(f"DEBUG: tyoee of board_chi.chess_board: {type(board)}")

            # The issue is deeper - even with same FENs, there can be engine state issues
            # Use ONLY the direct approach to avoid any state confusion
            # info = self.engine.analyse(board.chess_board, chess.engine.Limit(time=0.01))
            # print(f"DEBUG: Direct board analysis: {info}")
            # print(f"DEBUG: Direct board FEN: {board.chess_board.fen()}")

            print(f"DEBUG: tyoee of board_chi.chess_board: {type(board)}")

            info = self.engine.analyse(
                board_chi.chess_board, chess.engine.Limit(time=0.01)
            )

            score = info.get("score")

            if score is None:
                return 0.0

            # Convert score to float
            white_score = score.white()

            if score.is_mate():
                # For mate scores, return a large value based on mate distance
                mate_moves = white_score.mate()
                if mate_moves is not None:
                    if mate_moves > 0:
                        return 1000.0 - mate_moves  # White wins
                    else:
                        return -1000.0 - mate_moves  # Black wins
                return 0.0
            else:
                # Convert centipawns to pawns (divide by 100)
                cp_value = white_score.cp if hasattr(white_score, "cp") else 0
                return float(cp_value) / 100.0

        except Exception:
            return 0.0  # Fallback on any error
