"""
init for chipiron
"""

from chipiron.utils.my_random import set_seeds

from . import games as game
from . import players as player
from . import utils as tool

__all__ = ["set_seeds", "tool", "game", "player"]
