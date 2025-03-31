"""This module contains the classes for match results."""

from dataclasses import dataclass
from typing import Protocol

from chipiron.environments.chess.move import moveUci
from chipiron.games.game.final_game_result import FinalGameResult


@dataclass
class SimpleResults:
    """
    Represents the simple results of a match.
    """

    player_one_wins: int
    player_two_wins: int
    draws: int


class IMatchResults(Protocol):
    """
    Interface for match results.
    """

    def add_result_one_game(
        self, white_player_name_id: str, game_result: FinalGameResult
    ) -> None:
        """
        Adds the result of one game to the match results.

        Args:
            white_player_name_id (str): The ID of the white player.
            game_result (FinalGameResult): The result of the game.
        """

    def get_simple_result(self) -> SimpleResults:
        """
        Returns the simple results of the match.

        Returns:
            SimpleResults: The simple results of the match.
        """

    def __str__(self) -> str:
        """
        Returns a string representation of the match results.

        Returns:
            str: A string representation of the match results.
        """

    def finish(self) -> None:
        """
        Finishes the match and marks it as finished.
        """


@dataclass
class MatchResults:
    """
    Represents the results of a match between two players.
    """

    player_one_name_id: str
    player_two_name_id: str
    number_of_games: int = 0
    player_one_is_white_white_wins: int = 0
    player_one_is_white_black_wins: int = 0
    player_one_is_white_draws: int = 0
    player_two_is_white_white_wins: int = 0
    player_two_is_white_black_wins: int = 0
    player_two_is_white_draws: int = 0
    match_finished: bool = False

    def get_player_one_wins(self) -> int:
        """
        Returns the number of wins for player one.

        Returns:
            int: The number of wins for player one.
        """
        return self.player_one_is_white_white_wins + self.player_two_is_white_black_wins

    def get_player_two_wins(self) -> int:
        """
        Returns the number of wins for player two.

        Returns:
            int: The number of wins for player two.
        """
        return self.player_one_is_white_black_wins + self.player_two_is_white_white_wins

    def get_draws(self) -> int:
        """
        Returns the number of draws.

        Returns:
            int: The number of draws.
        """
        return self.player_one_is_white_draws + self.player_two_is_white_draws

    def get_simple_result(self) -> SimpleResults:
        """
        Returns the simple results of the match.

        Returns:
            SimpleResults: The simple results of the match.
        """
        simple_result: SimpleResults = SimpleResults(
            player_one_wins=self.get_player_one_wins(),
            player_two_wins=self.get_player_two_wins(),
            draws=self.get_draws(),
        )
        return simple_result

    def add_result_one_game(
        self, white_player_name_id: str, game_result: FinalGameResult
    ) -> None:
        """
        Adds the result of one game to the match results.

        Args:
            white_player_name_id (str): The ID of the white player.
            game_result (FinalGameResult): The result of the game.
        """
        self.number_of_games += 1
        if white_player_name_id == self.player_one_name_id:
            if game_result == FinalGameResult.WIN_FOR_WHITE:
                self.player_one_is_white_white_wins += 1
            elif game_result == FinalGameResult.WIN_FOR_BLACK:
                self.player_one_is_white_black_wins += 1
            elif game_result == FinalGameResult.DRAW:
                self.player_one_is_white_draws += 1
            else:
                pass
                # raise Exception('#')
        elif white_player_name_id == self.player_two_name_id:
            if game_result == FinalGameResult.WIN_FOR_WHITE:
                self.player_two_is_white_white_wins += 1
            elif game_result == FinalGameResult.WIN_FOR_BLACK:
                self.player_two_is_white_black_wins += 1
            elif game_result == FinalGameResult.DRAW:
                self.player_two_is_white_draws += 1
            else:
                raise Exception("!")
        else:
            raise Exception("?")

    def finish(self) -> None:
        """
        Finishes the match and marks it as finished.
        """
        self.match_finished = True

    def __str__(self) -> str:
        """
        Returns a string representation of the match results.

        Returns:
            str: A string representation of the match results.
        """
        str_ = (
            "Main result: "
            + self.player_one_name_id
            + " wins "
            + str(self.get_player_one_wins())
            + " "
        )
        str_ += self.player_two_name_id + " wins " + str(self.get_player_two_wins())
        str_ += " draws " + str(self.get_draws()) + "\n"

        str_ += self.player_one_name_id + " with white: "
        str_ += "Wins " + str(self.player_one_is_white_white_wins)
        str_ += ", Losses " + str(self.player_one_is_white_black_wins)
        str_ += ", Draws " + str(self.player_one_is_white_draws)
        str_ += "\n           with black: "
        str_ += "Wins " + str(self.player_two_is_white_black_wins)
        str_ += ", Losses " + str(self.player_two_is_white_white_wins)
        str_ += ", Draws " + str(self.player_two_is_white_draws) + "\n"

        str_ += self.player_two_name_id + " with white: "
        str_ += "Wins " + str(self.player_two_is_white_white_wins)
        str_ += ", Losses " + str(self.player_two_is_white_black_wins)
        str_ += ", Draws " + str(self.player_two_is_white_draws)
        str_ += "\n           with black: "
        str_ += "Wins " + str(self.player_one_is_white_black_wins)
        str_ += ", Losses " + str(self.player_one_is_white_white_wins)
        str_ += ", Draws " + str(self.player_one_is_white_draws)
        return str_


@dataclass
class MatchReport:
    """
    Represents a match report containing the match results and move history.
    """

    match_results: MatchResults
    match_move_history: dict[int, list[moveUci]]
