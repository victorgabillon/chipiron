"""
Module where we define the Stockfish Board Evaluator
"""

from dataclasses import dataclass

import chess.engine

import chipiron.environments.chess.board as boards


# TODO are we calling this?
@dataclass
class StockfishBoardEvalArgs:
    """Represents the arguments for the Stockfish board evaluator.

    Attributes:
        depth (int): The depth of the search algorithm.
        time_limit (float): The time limit for the search algorithm.
    """

    depth: int = 20
    time_limit: float = 0.1


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
        # todo make a large reformat so that the players are created after the launch of process
        return 0.0  # to win time atm, please make it work fast when in testing mode and 0.1s in normal mode

        # if self.engine is None:
        #     # if this object is created in the init then seending the object
        #     self.engine = chess.engine.SimpleEngine.popen_uci(
        #         # TODO: should we remove the hardcoding
        #         r"stockfish/stockfish/stockfish-ubuntu-x86-64-avx2")
        # info = self.engine.analyse(board.board, chess.engine.Limit(time=0.01))
        # self.engine.quit()
        # self.engine = None
        # return info["score"]
