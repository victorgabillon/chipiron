import typing
from dataclasses import dataclass
from enum import Enum

import chess

import chipiron.players as players
from chipiron.utils import seed
from chipiron.utils.small_tools import unique_int_from_list
from .game_args import GameArgs

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
    ) -> tuple[dict[chess.Color, players.PlayerFactoryArgs], GameArgs, seed | None]:

        merged_seed: seed | None = unique_int_from_list([self.seed_, game_number])
        assert merged_seed is not None

        player_one_factory_args: players.PlayerFactoryArgs = players.PlayerFactoryArgs(
            player_args=self.args_player_one,
            seed=merged_seed
        )
        player_two_factory_args: players.PlayerFactoryArgs = players.PlayerFactoryArgs(
            player_args=self.args_player_two,
            seed=merged_seed
        )

        player_color_to_factory_args: dict[chess.Color, players.PlayerFactoryArgs]
        if game_number < self.args_match.number_of_games_player_one_white:
            player_color_to_factory_args = {
                chess.WHITE: player_one_factory_args,
                chess.BLACK: player_two_factory_args
            }
        else:
            player_color_to_factory_args = {
                chess.WHITE: player_two_factory_args,
                chess.BLACK: player_one_factory_args
            }
        self.game_number += 1

        return player_color_to_factory_args, self.args_game, merged_seed

    def is_match_finished(self) -> bool:
        return (self.game_number >= self.args_match.number_of_games_player_one_white
                + self.args_match.number_of_games_player_one_black)
