import chess.engine
import chipiron.environments.chess.board as boards
from dataclasses import dataclass
from typing import Literal, Any

Human_Name_Literal = 'Stockfish'


@dataclass
class StockfishPlayer:
    type: Literal[Human_Name_Literal]  # for serialization
    depth: int = 20
    time_limit: float = 0.1
    engine: Any = None

    def select_move(self, board: boards.BoardChi):
        if self.engine is None:
            # if this object is created in the init then seending the object
            self.engine = chess.engine.SimpleEngine.popen_uci(
                # TODO: should we remove the hardcoding
                r"stockfish/stockfish_14.1_linux_x64/stockfish_14.1_linux_x64")
        result = self.engine.play(board, chess.engine.Limit(self.time_limit))
        return result.move

    def print_info(self):
        print(self.type)


async def pl(board, engine):
    result = await engine.play(board, chess.engine.Limit(time=0.1))
    return result
