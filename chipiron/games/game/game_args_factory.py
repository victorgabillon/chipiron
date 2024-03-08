import chess
from chipiron.players.factory import create_player
import random
from chipiron.utils.small_tools import unique_int_from_list
from chipiron.players.boardevaluators.table_base import create_syzygy, SyzygyTable
import chipiron.players as players
from enum import Enum
from dataclasses import dataclass
import typing
from .game_args import GameArgs
from chipiron.utils import seed

if typing.TYPE_CHECKING:
    import chipiron.games.match as match


class StaringPositionArgsType(str, Enum):
    fromFile = 'from_file'
    fen = 'fen'


@dataclass
class StaringPositionArgs:
    type: StaringPositionArgsType


@dataclass
class FenStaringPositionArgs(StaringPositionArgs):
    fen: str


@dataclass
class FileStaringPositionArgs(StaringPositionArgs):
    file_name: str


class GameArgsFactory:
    # TODO MAYBE CHANGE THE NAME, ALSO MIGHT BE SPLIT IN TWO (players and rules)?
    """
    The GameArgsFactory creates the players and decides the rules.
    So far quite simple
    This class is supposed to be dependent of Match-related classes (contrarily to the GameArgsFactory)

    """

    args_match: 'match.MatchSettingsArgs'
    seed_: int | None
    args_player_one: players.PlayerArgs
    args_player_two: players.PlayerArgs
    args_game: GameArgs
    game_number: int

    def __init__(
            self,
            args_match: 'match.MatchSettingsArgs',
            args_player_one: players.PlayerArgs,
            args_player_two: players.PlayerArgs,
            seed_: int | None,
            args_game: GameArgs
    ):
        self.args_match = args_match
        self.seed_ = seed_
        self.args_player_one = args_player_one
        self.args_player_two = args_player_two
        self.args_game = args_game
        self.game_number = 0

    def generate_game_args(
            self,
            game_number: int
    ) -> tuple[dict[chess.Color, players.Player], GameArgs, seed | None]:

        # Creating the players
        syzygy_table: SyzygyTable | None = create_syzygy()

        merged_seed: seed | None = unique_int_from_list([self.seed_, game_number])

        # if seed is None random uses the current system time as seed
        random_generator: random.Random = random.Random(merged_seed)
        player_one: players.Player = create_player(
            args=self.args_player_one,
            syzygy=syzygy_table,
            random_generator=random_generator
        )
        player_two: players.Player = create_player(
            args=self.args_player_two,
            syzygy=syzygy_table,
            random_generator=random_generator
        )

        player_color_to_player: dict[chess.Color, players.Player]
        if game_number < self.args_match.number_of_games_player_one_white:
            player_color_to_player = {chess.WHITE: player_one, chess.BLACK: player_two}
        else:
            player_color_to_player = {chess.WHITE: player_two, chess.BLACK: player_one}
        self.game_number += 1

        return player_color_to_player, self.args_game, merged_seed

    def is_match_finished(self):
        return (self.game_number >= self.args_match.number_of_games_player_one_white
                + self.args_match.number_of_games_player_one_black)
