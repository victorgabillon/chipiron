import queue

from chipiron.games.match.match_results import MatchResults, IMatchResults
from chipiron.games.match.observable_match_result import ObservableMatchResults
from chipiron.utils.is_dataclass import IsDataclass


class MatchResultsFactory:
    player_one_name: str
    player_two_name: str
    subscribers: list[queue.Queue[IsDataclass]] = []

    def __init__(
            self,
            player_one_name: str,
            player_two_name: str
    ) -> None:
        self.player_one_name = player_one_name
        self.player_two_name = player_two_name
        self.subscribers = []

    def create(
            self
    ) -> IMatchResults:
        match_result: MatchResults = MatchResults(
            player_one_name_id=self.player_one_name,
            player_two_name_id=self.player_two_name
        )
        if self.subscribers:
            obs_match_result: ObservableMatchResults = ObservableMatchResults(match_result)
            for subscriber in self.subscribers:
                obs_match_result.subscribe(subscriber)
            return obs_match_result
        else:
            return match_result

    def subscribe(
            self,
            subscriber: queue.Queue[IsDataclass]
    ) -> None:
        self.subscribers.append(subscriber)
