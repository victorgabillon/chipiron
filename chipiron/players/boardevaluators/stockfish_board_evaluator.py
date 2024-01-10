"""
Module where we define the Stockfish Board Evaluator
"""
import chess.engine
import chipiron.environments.chess.board as boards
from dataclasses import dataclass


# TODO are we calling this?
@dataclass
class StockfishBoardEvalArgs:
    depth: int = 20
    time_limit: float = 0.1


engine = chess.engine.SimpleEngine.popen_uci(
    "/home/victor_old/.pycharm/chipiron/stockfish/stockfish_14.1_linux_x64/stockfish_14.1_linux_x64")


class StockfishBoardEvaluator:
    """
    A board evaluator powered by stockfish
    """

    def __init__(self,
                 args: StockfishBoardEvalArgs):
        ...
        # is this doing anything???
        # self.depth = args.depth
        # board = chess.Board()
        # a = self.engine.analyse(board, chess.engine.Limit(time=args.time_limit))

    def value_white(
            self,
            board: boards.BoardChi
    ):
        print('infos')

        """ computes the value white of the board"""
        # self.engine = chess.engine.SimpleEngine.popen_uci("/home/victor_old/.pycharm/chipiron/stockfish/stockfish_14.1_linux_x64/stockfish_14.1_linux_x64")
        # looks like the engine dos not work when the gui is on ???!!!???
        # info = engine.analyse(board, chess.engine.Limit(time=0.1))
        return 0  # info["score"]
