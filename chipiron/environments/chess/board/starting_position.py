from dataclasses import dataclass
from enum import Enum


class StartingPositionArgsType(str, Enum):
    fromFile = 'from_file'
    fen = 'fen'


@dataclass
class StartingPositionArgs:
    type: StartingPositionArgsType


@dataclass
class FenStartingPositionArgs(StartingPositionArgs):
    fen: str


@dataclass
class FileStartingPositionArgs(StartingPositionArgs):
    file_name: str


AllStartingPositionArgs = FenStartingPositionArgs | FileStartingPositionArgs
