"""
Module to create a MatchResults object.

This module provides a MatchResultsFactory class that is responsible for creating MatchResults objects.
It also provides a way to subscribe to the MatchResults object and receive updates.

Classes:
- MatchResultsFactory: A factory class to create MatchResults objects and manage subscribers.

"""

import queue

from chipiron.displays.gui_protocol import GuiUpdate, Scope
from chipiron.environments.types import GameKind
from chipiron.games.match.match_results import IMatchResults, MatchResults
from chipiron.games.match.observable_match_result import ObservableMatchResults
from chipiron.utils.communication.gui_publisher import GuiPublisher


class MatchResultsFactory:
    """A factory class for creating MatchResults objects.

    This class provides methods to create MatchResults objects and subscribe subscribers to receive match results.

    Attributes:
        player_one_name (str): The name of player one.
        player_two_name (str): The name of player two.
        subscriber_queues (list[queue.Queue[GuiUpdate]]): A list of GUI queues to receive match results.

    """

    player_one_name: str
    player_two_name: str
    subscriber_queues: list[queue.Queue[GuiUpdate]]

    def __init__(self, player_one_name: str, player_two_name: str) -> None:
        """
        Initialize the MatchResultsFactory.

        Args:
            player_one_name (str): The name of player one.
            player_two_name (str): The name of player two.
        """
        self.player_one_name = player_one_name
        self.player_two_name = player_two_name
        self.subscriber_queues = []

    def create(self) -> IMatchResults:
        """
        Create a MatchResults object.

        Returns:
            IMatchResults: The created MatchResults object.
        """
        match_result: MatchResults = MatchResults(
            player_one_name_id=self.player_one_name,
            player_two_name_id=self.player_two_name,
        )
        if self.subscriber_queues:
            return ObservableMatchResults(match_result)
        return match_result

    def subscribe(self, subscriber_queue: queue.Queue[GuiUpdate]) -> None:
        """Register a GUI queue to receive match result updates."""
        self.subscriber_queues.append(subscriber_queue)

    def build_publishers(
        self, *, scope: Scope, game_kind: GameKind
    ) -> list[GuiPublisher]:
        return [
            GuiPublisher(out=q, schema_version=1, game_kind=game_kind, scope=scope)
            for q in self.subscriber_queues
        ]
