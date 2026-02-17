"""Module for move selection in a chess game."""

from .chess.register import register_chess_move_selectors
from .factory import create_main_move_selector, create_tree_and_value_move_selector
from .stockfish import StockfishPlayer

# Register chess-specific move selectors at import time
register_chess_move_selectors()

__all__ = [
    "StockfishPlayer",
    "create_main_move_selector",
    "create_tree_and_value_move_selector",
]
