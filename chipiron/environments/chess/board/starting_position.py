"""
Module defining the starting position arguments for the chess board.
"""
from dataclasses import dataclass
from enum import Enum


class StartingPositionArgsType(str, Enum):
    """
    Enum class representing the type of starting position arguments.
    """
    fromFile = 'from_file'
    fen = 'fen'


@dataclass
class StartingPositionArgs:
    """
    Dataclass representing the base class for starting position arguments.
    """
    type: StartingPositionArgsType


@dataclass
class FenStartingPositionArgs(StartingPositionArgs):
    """
    Dataclass representing the starting position arguments specified by FEN.
    """
    fen: str


@dataclass
class FileStartingPositionArgs(StartingPositionArgs):
    """
    Dataclass representing the starting position arguments specified by a file.
    """
    file_name: str


AllStartingPositionArgs = FenStartingPositionArgs | FileStartingPositionArgs
