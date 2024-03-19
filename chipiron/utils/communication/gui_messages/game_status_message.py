from dataclasses import dataclass

from chipiron.games.game.game_playing_status import PlayingStatus


@dataclass
class BackMessage:
    ...


@dataclass
class GameStatusMessage:
    status: PlayingStatus
