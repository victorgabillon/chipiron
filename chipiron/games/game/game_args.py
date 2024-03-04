from enum import Enum
from dataclasses import dataclass
from chipiron.environments.chess.board.starting_position import AllStartingPositionArgs


@dataclass
class GameArgs:
    starting_position: AllStartingPositionArgs
    max_half_moves: int | None = None
    each_player_has_its_own_thread: bool = False
