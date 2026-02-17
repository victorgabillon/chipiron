"""Module for move selection in a chess game."""

from .chess.register import register_chess_move_selectors
from .factory import create_main_move_selector, create_tree_and_value_move_selector
from .stockfish_args import StockfishSelectorArgs

# Register chess-specific move selectors at import time
register_chess_move_selectors()

__all__ = [
    "StockfishSelectorArgs",
    "create_main_move_selector",
    "create_tree_and_value_move_selector",
]
