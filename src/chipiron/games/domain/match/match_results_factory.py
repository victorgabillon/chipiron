"""Create match-results objects from validated match plans."""

import queue

from chipiron.displays.gui_protocol import GuiUpdate, Scope
from chipiron.displays.gui_publisher import GuiPublisher
from chipiron.environments.types import GameKind
from chipiron.games.domain.match.match_results import IMatchResults, MatchResults
from chipiron.games.domain.match.match_role_schedule import ValidatedMatchPlan
from chipiron.games.domain.match.observable_match_result import ObservableMatchResults


class MatchResultsFactory:
    """Create match-results objects for one validated match plan."""

    match_plan: ValidatedMatchPlan
    subscriber_queues: list[queue.Queue[GuiUpdate]]

    def __init__(self, match_plan: ValidatedMatchPlan) -> None:
        """Initialize the MatchResultsFactory.

        Args:
            match_plan (ValidatedMatchPlan): The validated plan whose
                participants and schedule this factory reports on.

        """
        self.match_plan = match_plan
        self.subscriber_queues = []

    @property
    def participant_ids(self) -> tuple[str, ...]:
        """Return the ordered participant identifiers for this validated plan."""
        return self.match_plan.participant_ids

    def create(self) -> IMatchResults:
        """Create a match-results object for the validated plan.

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
