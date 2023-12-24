import chess
import os
from chipiron.games.match_manager import MatchManager
from chipiron.games.game_manager_factory import GameManagerFactory
from chipiron.games.match_results_factory import MatchResultsFactory
from chipiron.players.factory import create_player
import multiprocessing
from chipiron.players.boardevaluators.factory import create_game_board_evaluator
import random
from chipiron.utils.small_tools import unique_int_from_list
import queue
from utils import path
from chipiron.players.boardevaluators.table_base.syzygy import SyzygyTable
from chipiron.players.boardevaluators.table_base.factory import create_syzygy_thread
from players.factory import PlayerArgs

from dataclasses import dataclass


@dataclass
class MatchArgs:
    number_of_games_player_one_white: int
    number_of_games_player_one_black: int
    game_setting_file: str | bytes | os.PathLike


@dataclass
class FenStaringPositionArgs:
    fen: str


@dataclass
class FileStaringPositionArgs:
    file_name: str


@dataclass
class GameArgs:
    starting_position: FenStaringPositionArgs | FileStaringPositionArgs


def create_match_manager(
        args_match: MatchArgs,
        args_player_one: PlayerArgs,
        args_player_two: PlayerArgs,
        output_folder_path: path,
        seed: int | None,
        args_game: dict,
        gui: bool
) -> MatchManager:
    main_thread_mailbox: queue.Queue = multiprocessing.Manager().Queue()

    # Creation of the Syzygy table for perfect play in low pieces cases, needed by the GameManager
    # and can also be used by the players
    syzygy_mailbox: SyzygyTable = create_syzygy_thread()

    player_one_name: str = args_player_one.name
    player_two_name: str = args_player_two.name

    game_board_evaluator = create_game_board_evaluator(gui=gui)

    game_manager_factory: GameManagerFactory = GameManagerFactory(
        syzygy_table=syzygy_mailbox,
        game_manager_board_evaluator=game_board_evaluator,
        output_folder_path=output_folder_path,
        main_thread_mailbox=main_thread_mailbox
    )

    match_results_factory: MatchResultsFactory = MatchResultsFactory(
        player_one_name=player_one_name,
        player_two_name=player_two_name
    )

    game_args_factory: GameArgsFactory = GameArgsFactory(
        args_match=args_match,
        args_player_one=args_player_one,
        args_player_two=args_player_two,
        seed=seed,
        args_game=args_game
    )

    match_manager: MatchManager = MatchManager(
        player_one_id=player_one_name,
        player_two_id=player_two_name,
        game_manager_factory=game_manager_factory,
        game_args_factory=game_args_factory,
        match_results_factory=match_results_factory,
        output_folder_path=output_folder_path
    )
    return match_manager


class GameArgsFactory:
    # TODO MAYBE CHANGE THE NAME, ALSO MIGHT BE SPLIT IN TWO (players and rules)?
    """
    The GameArgsFactory creates the players and decides the rules.
    So far quite simple
    This class is supposed to be dependent of Match-related classes (contrarily to the GameArgsFactory)

    """

    args_match: MatchArgs
    seed: int | None
    args_player_one: PlayerArgs
    args_player_two: PlayerArgs
    args_game: GameArgs
    game_number: int

    def __init__(self,
                 args_match: MatchArgs,
                 args_player_one: PlayerArgs,
                 args_player_two: PlayerArgs,
                 seed: int | None,
                 args_game: GameArgs):
        self.args_match = args_match
        self.seed = seed
        self.args_player_one = args_player_one
        self.args_player_two = args_player_two
        self.args_game = args_game
        self.game_number = 0

    def generate_game_args(self,
                           game_number: int):
        print('args_game', self.args_game)

        # Creating the players
        syzygy_table = SyzygyTable('')
        merged_seed = unique_int_from_list([self.seed, game_number])
        random_generator: random.Random = random.Random(merged_seed)
        player_one = create_player(args=self.args_player_one,
                                   syzygy=syzygy_table,
                                   random_generator=random_generator)
        player_two = create_player(args=self.args_player_two,
                                   syzygy=syzygy_table,
                                   random_generator=random_generator)

        if game_number < self.args_match.number_of_games_player_one_white:
            player_color_to_player = {chess.WHITE: player_one, chess.BLACK: player_two}
        else:
            player_color_to_player = {chess.WHITE: player_two, chess.BLACK: player_one}
        self.game_number += 1

        return player_color_to_player, self.args_game

    def is_match_finished(self):
        return (self.game_number >= self.args_match.number_of_games_player_one_white
                + self.args_match.number_of_games_player_one_black)
