import chess.engine
import chipiron.environments.chess.board as boards
from dataclasses import dataclass
from typing import Literal, Any
from .move_selector import MoveRecommendation
from .move_selector_types import MoveSelectorTypes


@dataclass
class StockfishPlayer:
    type: Literal[MoveSelectorTypes.Stockfish]  # for serialization
    depth: int = 20
    time_limit: float = 0.1
    engine: Any = None

    def select_move(
            self,
            board: boards.BoardChi,
            move_seed
    ):
        if self.engine is None:
            # if this object is created in the init then seending the object
            # self.engine = chess.engine.SimpleEngine.popen_uci(
            #    # TODO: should we remove the hardcoding
            #    r"stockfish/stockfish_14.1_linux_x64/stockfish_14.1_linux_x64")
            self.engine = chess.engine.SimpleEngine.popen_uci(
                # TODO: should we remove the hardcoding
                r"stockfish/stockfish/stockfish-ubuntu-x86-64-avx2")

        result = self.engine.play(board.board, chess.engine.Limit(self.time_limit))
        self.engine.quit()
        self.engine = None
        return MoveRecommendation(move=result.move)

    def print_info(self):
        print(self.type)
