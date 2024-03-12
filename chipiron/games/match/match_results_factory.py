from chipiron.games.match.match_results import MatchResults, IMatchResults
from chipiron.games.match.observable_match_result import ObservableMatchResults


class MatchResultsFactory:
    def __init__(self, player_one_name, player_two_name):
        self.player_one_name = player_one_name
        self.player_two_name = player_two_name
        self.subscribers = []

    def create(
            self
    ) -> IMatchResults:
        match_result: MatchResults = MatchResults(self.player_one_name, self.player_two_name)
        if self.subscribers:
            obs_match_result: ObservableMatchResults = ObservableMatchResults(match_result)
            for subscriber in self.subscribers:
                obs_match_result.subscribe(subscriber)
            return obs_match_result
        else:
            return match_result

    def subscribe(self, subscriber):
        self.subscribers.append(subscriber)
