from dataclasses import dataclass
from enum import Enum


class StaringPositionArgsType(str, Enum):
    fromFile = 'from_file'
    fen = 'fen'


@dataclass
class StaringPositionArgs:
    type: StaringPositionArgsType


@dataclass
class FenStaringPositionArgs(StaringPositionArgs):
    fen: str


@dataclass
class FileStaringPositionArgs(StaringPositionArgs):
    file_name: str
