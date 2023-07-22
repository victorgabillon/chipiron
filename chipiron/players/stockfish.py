import chess.engine
import chipiron as ch
import asyncio
import environments.chess.board as boards


class Stockfish:

    def __init__(self, options):
        self.depth = options['depth']

    def select_move(self, board: boards.BoardChi):
        # TODO: should we remove the hardcoding
        engine = chess.engine.SimpleEngine.popen_uci(r"stockfish/stockfish_14.1_linux_x64/stockfish_14.1_linux_x64")
        result = engine.play(board, chess.engine.Limit(time=0.1))
        return result.move

    def print_info(self):
        super().print_info()
        print('type: Stockfish')


async def pl(board, engine):
    result = await engine.play(board, chess.engine.Limit(time=0.1))
    return result
