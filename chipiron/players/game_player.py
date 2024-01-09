import chess
from .player import Player
from chipiron.environments.chess.board import BoardChi
import queue
import copy


class GamePlayer:
    """A class that wraps a player for a game purposes
    it adds the color information and probably stuff to continue the computation when the opponent is computing"""

    def __init__(self,
                 player: Player,
                 color):
        self.color = color
        self._player = player

    @property
    def player(self):
        return self._player

    def select_move(self,
                    board: BoardChi):
        best_move = self._player.select_move(board)
        return best_move


def send_board_to_game_player(
        board: BoardChi,
        game_player: GamePlayer,
        queue_move: queue.Queue) -> None:

    if board.turn == game_player.color:
        move: chess.Move = game_player.select_move(board=board)
        message = {'type': 'move',
                   'move': move,
                   'corresponding_board': board.fen(),
                   'player': game_player.player.id
                   }
        deep_copy_message = copy.deepcopy(message)
        print('sending ', message)
        queue_move.put(deep_copy_message)
