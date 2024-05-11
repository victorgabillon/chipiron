"""
init file for board module
"""
import typing

from .board import BoardChi
from .board_modification import BoardModification
from .factory import create_board, create_board_factory, BoardFactory
from .iboard import IBoard


fen = typing.Annotated[str, 'fen']

__all__ = [
    "BoardModification",
    "BoardChi",
    "create_board",
    "BoardFactory"
]
