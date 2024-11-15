"""
This module contains the implementation of the BaseTreeExplorationScript class.

The BaseTreeExplorationScript class is responsible for running a script that performs base tree exploration in a chess game.
"""

import cProfile
import random
from dataclasses import dataclass

from chipiron.environments.chess.board.factory import create_rust_board
from chipiron.players.boardevaluators.table_base.factory import create_syzygy
from chipiron.players.factory import create_player
from chipiron.players.player_args import PlayerArgs
from chipiron.players.utils import fetch_player_args_convert_and_save
from chipiron.scripts.script import Script


@dataclass
class BaseTreeExplorationArgs:
    ...


class BaseTreeExplorationScript:
    """
    The BaseTreeExplorationScript
    """

    args_dataclass_name: type[BaseTreeExplorationArgs] = BaseTreeExplorationArgs

    def __init__(
            self,
            base_script: Script
    ) -> None:
        """
        Initializes a new instance of the BaseTreeExploration class.

        Args:
            base_script (Script): The base script to be used for tree exploration.
        """
        self.base_script = base_script

    def run(self) -> None:
        """
        Runs the base tree exploration script.
        """
        syzygy = create_syzygy(use_rust=False)

        profile = cProfile.Profile()
        profile.enable()

        file_name_player = 'RecurZipfBase3.yaml'
        # file_name_player: str = 'Uniform.yaml'

        player_one_args: PlayerArgs = fetch_player_args_convert_and_save(
            file_name_player=file_name_player
        )

        # player_one_args.main_move_selector.stopping_criterion.tree_move_limit = 1000000
        random_generator = random.Random()
        player = create_player(args=player_one_args, syzygy=syzygy, random_generator=random_generator)

        board = create_rust_board()
        player.select_move(
            board=board,
            seed_int=0
        )

    def terminate(self) -> None:
        """
        Terminates the base tree exploration script.
        """
        self.base_script.terminate()
