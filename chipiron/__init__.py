"""
init for chipiron
"""

from . import games as game
from . import players as player
from . import utils as tool
from .utils.my_random import set_seeds

__all__ = ["set_seeds", "tool", "game", "player"]
