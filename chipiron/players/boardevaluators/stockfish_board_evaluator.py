"""
Module where we define the Stockfish Board Evaluator
"""
from dataclasses import dataclass

import chess.engine

import chipiron.environments.chess.board as boards


# TODO are we calling this?
@dataclass
class StockfishBoardEvalArgs:
    depth: int = 20
    time_limit: float = 0.1


class StockfishBoardEvaluator:
    """
    A board evaluator powered by stockfish
    """
    engine: chess.engine.SimpleEngine | None

    def __init__(
            self,
            args: StockfishBoardEvalArgs
    ) -> None:
        self.engine = None

    def value_white(
            self,
            board: boards.BoardChi
    ) -> float:
        print('infos')
        # todo make a large reformat so that the players are created after the launch of process
        """ computes the value white of the board"""
        return 0.  # to win time atm, please make it work fast when in testing mode and 0.1s in normal mode

        # if self.engine is None:
        #     # if this object is created in the init then seending the object
        #     self.engine = chess.engine.SimpleEngine.popen_uci(
        #         # TODO: should we remove the hardcoding
        #         r"stockfish/stockfish/stockfish-ubuntu-x86-64-avx2")
        # info = self.engine.analyse(board.board, chess.engine.Limit(time=0.01))
        # self.engine.quit()
        # self.engine = None
        # return info["score"]
