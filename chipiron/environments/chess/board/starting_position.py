"""
Module defining the starting position arguments for the chess board.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Protocol

from chipiron.environments.chess.board.board_tools import convert_to_fen


class StartingPositionArgsType(str, Enum):
    """
    Enum class representing the type of starting position arguments.
    """

    fromFile = "from_file"
    fen = "fen"


@dataclass
class StartingPositionArgs(Protocol):
    """
    Dataclass representing the base class for starting position arguments.
    """

    type: StartingPositionArgsType

    def get_fen(self) -> str: ...


@dataclass
class FenStartingPositionArgs:
    """
    Dataclass representing the starting position arguments specified by FEN.
    """

    type: Literal[StartingPositionArgsType.fen]
    fen: str

    def get_fen(self) -> str:
        return self.fen


@dataclass
class FileStartingPositionArgs(StartingPositionArgs):
    """
    Dataclass representing the starting position arguments specified by a file.
    """

    type: Literal[StartingPositionArgsType.fromFile]
    file_name: str

    def get_fen(self) -> str:
        with open("data/starting_boards/" + self.file_name, "r") as f:
            ascii_board: str = str(f.read())
            fen: str = convert_to_fen(ascii_board)
        return fen


AllStartingPositionArgs = FenStartingPositionArgs | FileStartingPositionArgs
