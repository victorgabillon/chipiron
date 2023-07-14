import chess.engine


class StockfishBoardEvaluator:
    def __init__(self):
        self.engine = chess.engine.SimpleEngine.popen_uci(
            "stockfish/stockfish_14.1_linux_x64/stockfish_14.1_linux_x64")
        self.depth = 20

    def value_white(self, board):
        info = self.engine.analyse(board, chess.engine.Limit(time=0.1))
        return info["score"]
