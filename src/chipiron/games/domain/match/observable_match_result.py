"""Observable wrapper around role-aware match results."""

from dataclasses import dataclass, field

from chipiron.displays.gui_protocol import ParticipantMatchStats, UpdMatchResults
from chipiron.displays.gui_publisher import GuiPublisher
from chipiron.games.domain.game.final_game_result import FinalGameResult, GameReport

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
        self,
        *,
        game_report: GameReport | None = None,
        white_player_name_id: str | None = None,
        game_result: FinalGameResult | None = None,
    ) -> None:
        """Add the result of a single game to the match results."""
        self.match_results.add_result_one_game(
            game_report=game_report,
            white_player_name_id=white_player_name_id,
            game_result=game_result,
        )
        self.notify_new_results()

    def notify_new_results(self) -> None:
        """Notifies all subscribed mailboxes about the new match results.

        Returns:
            None

        """
        simple = self.match_results.get_simple_result()

        payload = UpdMatchResults(
            participant_stats=[
                ParticipantMatchStats(
                    participant_id=participant_id,
                    wins=simple.stats_by_participant[participant_id].wins,
                    losses=simple.stats_by_participant[participant_id].losses,
                    draws=simple.stats_by_participant[participant_id].draws,
                    unknown=simple.stats_by_participant[participant_id].unknown,
                )
                for participant_id in simple.participant_order
            ],
            draws=simple.draws,
            games_played=simple.games_played,
            match_finished=self.match_results.match_finished,
        )

        for pub in self.publishers:
            pub.publish(payload)

    def get_simple_result(self) -> SimpleResults:
        """Retrieve a simplified version of the match results.

        Returns:
            SimpleResults: A simplified version of the match results.

        """
        return SimpleResults(
            participant_order=self.match_results.participant_ids,
            stats_by_participant=self.match_results.stats_by_participant,
            draws=self.match_results.get_draws(),
            games_played=self.match_results.number_of_games,
        )

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
