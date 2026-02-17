"""Runtime Stockfish move selector implementation."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import atomheart.board as boards
import chess.engine
from atomheart.board import create_board_chi
from atomheart.board.utils import FenPlusHistory
from valanga.game import BranchName, Seed
from valanga.policy import NotifyProgressCallable, Recommendation

from chipiron.environments.chess.types import ChessState
from chipiron.utils.path_variables import STOCKFISH_BINARY_PATH

if TYPE_CHECKING:
    from atomheart import BoardChi


class StockfishError(RuntimeError):
    """Base error for Stockfish-related failures."""


class StockfishBinaryNotFoundError(StockfishError):
    """Raised when the Stockfish binary cannot be found."""

    def __init__(self, path: Path) -> None:
        msg = (
            f"Stockfish binary not found at {path}.\n"
            "Please install Stockfish by running:\n"
            "    make stockfish\n"
            "This will download and install Stockfish 16 (~40MB) "
            "to the correct location."
        )
        super().__init__(msg)


class StockfishStartupError(StockfishError):
    """Raised when the Stockfish engine fails to start."""

    def __init__(self, path: Path, original_error: OSError) -> None:
        msg = (
            f"Failed to start Stockfish engine at {path}.\n"
            "The binary may be corrupted. Try reinstalling with:\n"
            f"    rm -rf {path.parent}\n"
            "    make stockfish\n"
            f"Original error: {original_error}"
        )
        super().__init__(msg)


@dataclass
class StockfishSelector:
    """Runtime selector backed by a Stockfish process."""

    depth: int = 20
    time_limit: float = 0.1
    engine: Any = None

    @staticmethod
    def is_stockfish_available() -> bool:
        return STOCKFISH_BINARY_PATH.exists() and STOCKFISH_BINARY_PATH.is_file()

    def recommend(
        self,
        state: ChessState,
        seed: Seed,
        notify_progress: NotifyProgressCallable | None = None,
    ) -> Recommendation:
        _ = seed
        _ = notify_progress
        best: BranchName = self._select_move(state.board).recommended_name
        return Recommendation(recommended_name=best)

    def _select_move(self, board: boards.IBoard) -> Recommendation:
        if self.engine is None:
            if not STOCKFISH_BINARY_PATH.exists():
                raise StockfishBinaryNotFoundError(STOCKFISH_BINARY_PATH)

            try:
                self.engine = chess.engine.SimpleEngine.popen_uci(
                    str(STOCKFISH_BINARY_PATH)
                )
            except OSError as e:
                raise StockfishStartupError(STOCKFISH_BINARY_PATH, e) from e

        board_chi: BoardChi = create_board_chi(
            fen_with_history=FenPlusHistory(
                current_fen=board.fen,
                historical_moves=board.move_history_stack,
            )
        )
        result = self.engine.play(
            board_chi.chess_board,
            chess.engine.Limit(time=self.time_limit, depth=self.depth),
        )

        self.engine.quit()
        self.engine = None
        return Recommendation(recommended_name=result.move.uci(), evaluation=None)


def create_stockfish_selector(
    *, depth: int = 20, time_limit: float = 0.1
) -> StockfishSelector:
    """Build a runtime Stockfish selector from serializable args."""
    return StockfishSelector(depth=depth, time_limit=time_limit)
