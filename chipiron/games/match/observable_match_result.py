"""
This module contains the ObservableMatchResults class, which is a wrapper around the MatchResults class.
"""

import copy
from dataclasses import dataclass, field

from chipiron.games.game.final_game_result import FinalGameResult
from chipiron.utils.communication.gui_messages.gui_messages import UpdMatchResults
from chipiron.utils.communication.gui_publisher import GuiPublisher

from .match_results import MatchResults, SimpleResults


@dataclass(slots=True)
class ObservableMatchResults:
    """
    The ObservableMatchResults class provides a way to observe and subscribe to changes in the match results.
    It maintains a list of mailboxes (queue.Queue) to which it sends notifications whenever there are new match results.

    Attributes:
        match_results (MatchResults): The underlying MatchResults object that stores the match results.
        mailboxes (list[queue.Queue[IsDataclass]]): A list of mailboxes to which notifications are sent.

    """

    match_results: MatchResults
    publishers: list[GuiPublisher] = field(default_factory=list)


    def subscribe(self, pub: GuiPublisher) -> None:

        """
        Subscribes a publisher to receive notifications.

        Args:
            pub (GuiPublisher): The publisher to subscribe.

        Returns:
            None
        """
        self.publishers.append(pub)


    def copy_match_result(self) -> MatchResults:
        """Creates a deep copy of the match results.

        Returns:
            MatchResults: A deep copy of the match results.
        """
        match_result_copy: MatchResults = copy.deepcopy(self.match_results)
        return match_result_copy

    def add_result_one_game(
        self, white_player_name_id: str, game_result: FinalGameResult
    ) -> None:
        """Adds the result of a single game to the match results.

        Args:
            white_player_name_id (str): The ID of the white player.
            game_result (FinalGameResult): The result of the game.

        Returns:
            None
        """
        self.match_results.add_result_one_game(white_player_name_id, game_result)
        self.notify_new_results()


    def notify_new_results(self) -> None:
        """Notifies all subscribed mailboxes about the new match results.

        Returns:
            None
        """
        # Map your MatchResults into stable scalar fields
        simple = self.match_results.get_simple_result()
        # If you don't have games_played already, define it consistently:
        games_played = simple.player_one_wins + simple.player_two_wins + simple.draws

        payload = UpdMatchResults(
            kind="match_results",
            wins_white=simple.player_one_wins,
            wins_black=simple.player_two_wins,
            draws=simple.draws,
            games_played=games_played,
            match_finished=self.match_results.match_finished,
        )

        for pub in self.publishers:
            pub.publish(payload)


    def get_simple_result(self) -> SimpleResults:
        """Retrieves a simplified version of the match results.

        Returns:
            SimpleResults: A simplified version of the match results.
        """
        simple_results = SimpleResults(
            player_one_wins=self.match_results.get_player_one_wins(),
            player_two_wins=self.match_results.get_player_two_wins(),
            draws=self.match_results.get_draws(),
        )
        return simple_results

    def finish(self) -> None:
        """Marks the match as finished and notifies all subscribed mailboxes.

        Returns:
            None
        """
        self.match_results.match_finished = True
        self.notify_new_results()

    def __str__(self) -> str:
        """Returns a string representation of the match results.

        Returns:
            str: A string representation of the match results.
        """
        return str(self.match_results)
