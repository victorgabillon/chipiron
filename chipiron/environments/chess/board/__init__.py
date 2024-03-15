import typing

from .board import BoardChi
from .board_modification import BoardModification
from .factory import create_board

fen = typing.Annotated[str, 'fen']

__all__ = [
    "BoardModification",
    "BoardChi",
    "create_board"
]
