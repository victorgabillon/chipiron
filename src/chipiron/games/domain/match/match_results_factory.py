"""Module to create a MatchResults object.

This module provides a MatchResultsFactory class that is responsible for creating MatchResults objects.
It also provides a way to subscribe to the MatchResults object and receive updates.

Classes:
- MatchResultsFactory: A factory class to create MatchResults objects and manage subscribers.
"""

import queue

from chipiron.displays.gui_protocol import GuiUpdate, Scope
from chipiron.displays.gui_publisher import GuiPublisher
from chipiron.environments.types import GameKind
from chipiron.games.domain.match.match_results import IMatchResults, MatchResults
from chipiron.games.domain.match.observable_match_result import ObservableMatchResults


class MatchResultsFactory:
    """A factory class for creating MatchResults objects.

    This class provides methods to create MatchResults objects and subscribe
    GUI listeners to receive match-result updates for the configured
    participants.

    """

    participant_ids: tuple[str, ...]
    subscriber_queues: list[queue.Queue[GuiUpdate]]

    def __init__(self, participant_ids: tuple[str, ...]) -> None:
        """Initialize the MatchResultsFactory.

        Args:
            participant_ids (tuple[str, ...]): Ordered participant identifiers.

        """
        self.participant_ids = participant_ids
        self.subscriber_queues = []

    def create(self) -> IMatchResults:
        """Create a MatchResults object.

        Returns:
            IMatchResults: The created MatchResults object.

        """
        match_result: MatchResults = MatchResults(
            participant_ids=self.participant_ids,
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
        """Build publishers."""
        return [
            GuiPublisher(out=q, schema_version=1, game_kind=game_kind, scope=scope)
            for q in self.subscriber_queues
        ]
