"""
players utils
"""

import os
from shutil import copyfile
from typing import Any

from chipiron.players.player_args import PlayerArgs
from chipiron.players.player_ids import PlayerConfigTag
from chipiron.utils.small_tools import fetch_args_modify_and_convert, path
