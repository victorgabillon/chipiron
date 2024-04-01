import cProfile
import random

from chipiron.environments.chess.board import create_board
from chipiron.players.boardevaluators.table_base.syzygy import SyzygyTable
from chipiron.players.factory import create_player
from chipiron.players.player_args import PlayerArgs
from chipiron.players.utils import fetch_player_args_convert_and_save
from chipiron.scripts.script import Script


class BaseTreeExplorationScript:

    def __init__(
            self,
            base_script: Script
    ) -> None:
        self.base_script = base_script

    def run(self) -> None:
        syzygy = SyzygyTable('')

        profile = cProfile.Profile()
        profile.enable()

        # file_name_player_one = 'RecurZipfBase3.yaml'
        file_name_player: str = 'Uniform.yaml'

        player_one_args: PlayerArgs = fetch_player_args_convert_and_save(
            file_name_player=file_name_player
        )

        # player_one_args.main_move_selector.stopping_criterion.tree_move_limit = 1000000
        random_generator = random.Random()
        player = create_player(args=player_one_args, syzygy=syzygy, random_generator=random_generator)

        board = create_board()
        player.select_move(
            board=board,
            seed_=0
        )

    def terminate(self) -> None:
        self.base_script.terminate()
