"""
Module defining the starting position arguments for the chess board.
"""

from dataclasses import dataclass
from enum import Enum
from importlib.resources import as_file, files
from typing import Literal, Protocol

from chipiron.environments.chess_env.board.board_tools import convert_to_fen


class StartingPositionArgsType(str, Enum):
    """
    Enum class representing the type of starting position arguments.
    """

    FROMFILE = "from_file"
    FEN = "fen"


@dataclass
class StartingPositionArgs(Protocol):
    """
    Dataclass representing the base class for starting position arguments.
    """

    type: StartingPositionArgsType

    def get_fen(self) -> str:
        """
        Returns the FEN (Forsyth-Edwards Notation) string representing the current state of the chess board.
        Returns:
            str: The FEN string of the current board position.
        """


@dataclass
class FenStartingPositionArgs:
    """
    Dataclass representing the starting position arguments specified by FEN.
    """

    type: Literal[StartingPositionArgsType.FEN]
    fen: str

    def get_fen(self) -> str:
        """
        Returns the FEN (Forsyth-Edwards Notation) string representing the current state of the chess board.
        Returns:
            str: The FEN string of the current board position.
        """
        return self.fen


@dataclass
class FileStartingPositionArgs(StartingPositionArgs):
    """
    Dataclass representing the starting position arguments specified by a file.
    """

    type: Literal[StartingPositionArgsType.FROMFILE]
    file_name: str

    def get_fen(self) -> str:
        """
        Returns the FEN (Forsyth-Edwards Notation) string representing the current state of the chess board.
        Returns:
            str: The FEN string of the current board position.
        """
        resource = files("chipiron").joinpath("data/starting_boards/" + self.file_name)
        with as_file(resource) as real_path:
            with open(real_path, "r", encoding="utf-8") as f:
                ascii_board: str = str(f.read())
                fen: str = convert_to_fen(ascii_board)
        return fen


AllStartingPositionArgs = FenStartingPositionArgs | FileStartingPositionArgs
