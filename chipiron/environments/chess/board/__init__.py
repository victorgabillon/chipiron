from .board_modification import BoardModification
from .board import BoardChi
from .factory import create_board
import typing

fen = typing.Annotated[str, 'fen']
