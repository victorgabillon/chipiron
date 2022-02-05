from src.players.player import Player


class GamePlayer(Player):
    """A class that wraps a player for a game purposes
    it adds the coor information and probably stuff so as to continue the computation when the opponent is computing"""

    def __init__(self, player, color):
        self.color = color
        self._player = player

    def select_move(self,board):
        best_move = self._player.select_move(board)
        return best_move


