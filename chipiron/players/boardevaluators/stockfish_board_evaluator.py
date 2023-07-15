"""
Module where we define the Stockfish Board Evaluator
"""
import chess.engine
import chessenvironment.board as boards


class StockfishBoardEvaluator:
    """
    A board evaluator powered by stockfish
    """

    def __init__(self):
        self.engine = chess.engine.SimpleEngine.popen_uci(
            "/home/victor_old/.pycharm/chipiron/stockfish/stockfish_14.1_linux_x64/stockfish_14.1_linux_x64")
        self.depth = 20
        board = chess.Board()
        a = self.engine.analyse(board, chess.engine.Limit(time=0.1))
        print('rrrfff', a)

    def value_white(
            self,
            board: boards.BoardChi
    ):
        """ computes the value white of the board"""
        self.engine = chess.engine.SimpleEngine.popen_uci(
            "/home/victor_old/.pycharm/chipiron/stockfish/stockfish_14.1_linux_x64/stockfish_14.1_linux_x64")
        info = self.engine.analyse(board, chess.engine.Limit(time=0.1))


        return info["score"]
