from dataclasses import dataclass
import chess
from chipiron.environments.chess.board import fen
from chipiron.games.game.game_playing_status import PlayingStatus


@dataclass
class GameStatusMessage:
    status: PlayingStatus


@dataclass
class BackMessage:
    ...
