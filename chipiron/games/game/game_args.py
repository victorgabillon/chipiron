from enum import Enum
from dataclasses import dataclass
from chipiron.environments.chess.board.starting_position import FenStaringPositionArgs, FileStaringPositionArgs


@dataclass
class GameArgs:
    starting_position: FenStaringPositionArgs | FileStaringPositionArgs
    max_half_moves: int | None = None
