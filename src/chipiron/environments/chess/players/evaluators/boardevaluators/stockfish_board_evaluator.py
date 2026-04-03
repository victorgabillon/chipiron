"""Module where we define the Stockfish Board Evaluator."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import chess.engine
from atomheart.games.chess.board.factory import create_board_chi
from atomheart.games.chess.board.utils import FenPlusHistory
from valanga import Color, OverEvent
from valanga.evaluations import Certainty, Value
from valanga.over_event import Outcome

from chipiron.environments.chess.players.evaluators.boardevaluators.board_evaluator_type import (
    BoardEvalTypes,
)
from chipiron.environments.chess.types import ChessState
from chipiron.utils.logger import chipiron_logger
from chipiron.utils.path_variables import STOCKFISH_BINARY_PATH

if TYPE_CHECKING:
    from collections.abc import Hashable

    from atomheart.games.chess.board.board_chi import BoardChi


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
        """Post-initialization method for the dataclass.

        Automatically sets the 'type' attribute to BoardEvalTypes.STOCKFISH_BOARD_EVAL
        to avoid manual assignment and to facilitate discrimination during deserialization.
        """
        # to avoid having to set the type mannually in the code as it is obvious and only used for discrimanating in deserialization
        object.__setattr__(self, "type", BoardEvalTypes.STOCKFISH_BOARD_EVAL)


class StockfishBoardEvaluator:
    """A board evaluator powered by stockfish."""

    engine: chess.engine.SimpleEngine | None
    args: StockfishBoardEvalArgs

    def __init__(self, args: StockfishBoardEvalArgs) -> None:
        """Initialize a StockfishBoardEvaluator object.

        Args:
            args (StockfishBoardEvalArgs): The arguments for the StockfishBoardEvaluator.

        Returns:
            None

        """
        self.args = args
        self.engine = None

    @staticmethod
    def _candidate_paths() -> list[str]:
        """Return candidate filesystem paths for the Stockfish binary."""
        return [
            str(STOCKFISH_BINARY_PATH),
            "/usr/games/stockfish",
            "stockfish",
            r"stockfish/stockfish/stockfish-ubuntu-x86-64-avx2",
        ]

    @staticmethod
    def is_available() -> bool:
        """Return whether available."""
        return any(
            Path(path).exists() for path in StockfishBoardEvaluator._candidate_paths()
        )

    def evaluate(self, state: ChessState) -> Value:
        """Compute the value of the board for the white player.

        Args:
            state (ChessState): The state object representing the current state of the game.

        Returns:
            float: The value of the board for the white player.

        """
        board = state.board
        try:
            if self.engine is None:
                # Try multiple possible Stockfish paths
                for path in self._candidate_paths():
                    try:
                        self.engine = chess.engine.SimpleEngine.popen_uci(path)
                        break
                    except (FileNotFoundError, OSError):
                        continue

                if self.engine is None:
                    chipiron_logger.warning(
                        "Stockfish binary not found in any of the expected locations."
                    )
                    return Value(
                        score=0.0, certainty=Certainty.ESTIMATE
                    )  # Fallback if no Stockfish found

            # Use the board's chess_board directly to avoid state inconsistencies
            # transform the board
            board_chi: BoardChi = create_board_chi(
                fen_with_history=FenPlusHistory(
                    current_fen=board.fen,
                )
            )

            info = self.engine.analyse(
                board_chi.chess_board, chess.engine.Limit(time=0.01)
            )

            score = info.get("score")

            if score is None:
                return Value(
                    score=0.0, certainty=Certainty.ESTIMATE
                )  # Fallback if no score

            # Convert score to float
            white_score = score.white()

            if score.is_mate():
                # For mate scores, return a large value based on mate distance
                mate_moves = white_score.mate()
                if mate_moves is not None:
                    if mate_moves > 0:
                        return Value(
                            score=1000.0 - mate_moves,
                            certainty=Certainty.FORCED,
                            over_event=cast(
                                "OverEvent[Hashable]",
                                OverEvent[Color](
                                    outcome=Outcome.WIN,
                                    winner=Color.WHITE,
                                ),
                            ),
                        )  # White wins
                    return Value(
                        score=-1000.0 - mate_moves,
                        certainty=Certainty.FORCED,
                        over_event=cast(
                            "OverEvent[Hashable]",
                            OverEvent[Color](
                                outcome=Outcome.WIN,
                                winner=Color.BLACK,
                            ),
                        ),
                    )  # Black wins
                return Value(score=0.0, certainty=Certainty.ESTIMATE)  # ???
            # Convert centipawns to pawns (divide by 100)
            cp_value = getattr(white_score, "cp", 0)
            return Value(score=float(cp_value) / 100.0, certainty=Certainty.ESTIMATE)

        except (chess.engine.EngineError, FileNotFoundError, OSError):
            return Value(
                score=0.0, certainty=Certainty.ESTIMATE
            )  # Fallback on any error
