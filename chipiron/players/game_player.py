"""
Module for the GamePlayer class.
"""

import copy
import queue

import chess

from chipiron.players.move_selector.move_selector import MoveRecommendation
from chipiron.utils import seed
from chipiron.utils.communication.player_game_messages import MoveMessage
from chipiron.utils.dataclass import IsDataclass
from chipiron.utils.logger import chipiron_logger

from ..environments.chess.board.utils import FenPlusHistory
from .player import Player


class GamePlayer:
    """A class that wraps a player for a game purposes
    it adds the color information and probably stuff to continue the computation when the opponent is computing
    """

    _player: Player
    color: chess.Color

    def __init__(self, player: Player, color: chess.Color) -> None:
        self.color = color
        self._player = player

    @property
    def player(self) -> Player:
        """Return the player object.

        Returns:
            Player: The player object.
        """
        return self._player

    def select_move(
        self, fen_plus_history: FenPlusHistory, seed_int: seed | None = None
    ) -> MoveRecommendation:
        """Selects the best move to play based on the current board position.

        Args:
            seed_int (seed | None, optional): The seed value for randomization. Defaults to None.

        Raises:
            Exception: If there are no legal moves in the current position.

        Returns:
            MoveRecommendation: The recommended move to play.
        """

        assert seed_int is not None
        assert self.color == fen_plus_history.current_turn()
        best_move: MoveRecommendation = self._player.select_move(
            fen_plus_history=fen_plus_history, seed_int=seed_int
        )
        return best_move


def game_player_computes_move_on_board_and_send_move_in_queue(
    fen_plus_history: FenPlusHistory,
    game_player: GamePlayer,
    queue_move: queue.Queue[IsDataclass],
    seed_int: seed,
) -> None:
    """Computes the move for the game player on the given board and sends the move in the queue.

    Args:
        game_player (GamePlayer): The game player.
        queue_move (queue.Queue[IsDataclass]): The queue to send the move.
        seed_int (seed): The seed for move selection.

    Returns:
        None
    """

    if fen_plus_history.current_turn() == game_player.color:
        move_recommendation: MoveRecommendation = game_player.select_move(
            fen_plus_history=fen_plus_history, seed_int=seed_int
        )
        message: MoveMessage = MoveMessage(
            move=move_recommendation.move,
            corresponding_board=fen_plus_history.current_fen,
            player_name=game_player.player.id,
            evaluation=move_recommendation.evaluation,
            color_to_play=game_player.color,
        )
        chipiron_logger.info("Sending %s", message)

        deep_copy_message = copy.deepcopy(message)
        queue_move.put(deep_copy_message)
    else:
        chipiron_logger.warning(
            "Game player rejects move query %s",
            fen_plus_history.current_fen,
        )
