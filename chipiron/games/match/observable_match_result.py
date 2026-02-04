"""Document the module contains the ObservableMatchResults class, which is a wrapper around the MatchResults class."""

from dataclasses import dataclass, field

from chipiron.displays.gui_protocol import UpdMatchResults
from chipiron.displays.gui_publisher import GuiPublisher
from chipiron.games.game.final_game_result import FinalGameResult

from .match_results import MatchResults, SimpleResults


def make_publishers() -> list[GuiPublisher]:
    """Create an empty list of GuiPublisher."""
    return []


@dataclass(slots=True)
class ObservableMatchResults:
    """The ObservableMatchResults class provides a way to observe and subscribe to changes in the match results.

    It maintains a list of mailboxes (queue.Queue) to which it sends notifications whenever there are new match results.

    Attributes:
        match_results (MatchResults): The underlying MatchResults object that stores the match results.
        mailboxes (list[queue.Queue[IsDataclass]]): A list of mailboxes to which notifications are sent.

    """

    match_results: MatchResults
    publishers: list[GuiPublisher] = field(default_factory=make_publishers)

    def subscribe(self, pub: GuiPublisher) -> None:
        """Subscribe a publisher to receive notifications.

        Args:
            pub (GuiPublisher): The publisher to subscribe.

        Returns:
            None

        """
        self.publishers.append(pub)

    def replace_publishers(self, pubs: list[GuiPublisher]) -> None:
        """Replace publishers."""
        self.publishers = list(pubs)

    def add_result_one_game(
        self, white_player_name_id: str, game_result: FinalGameResult
    ) -> None:
        """Add the result of a single game to the match results.

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
            wins_white=simple.player_one_wins,
            wins_black=simple.player_two_wins,
            draws=simple.draws,
            games_played=games_played,
            match_finished=self.match_results.match_finished,
        )

        for pub in self.publishers:
            pub.publish(payload)

    def get_simple_result(self) -> SimpleResults:
        """Retrieve a simplified version of the match results.

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
        """Mark the match as finished and notifies all subscribed mailboxes.

        Returns:
            None

        """
        self.match_results.match_finished = True
        self.notify_new_results()

    def __str__(self) -> str:
        """Return a string representation of the match results.

        Returns:
            str: A string representation of the match results.

        """
        return str(self.match_results)
