from chipiron.games.match.math_results import MatchResults, ObservableMatchResults


class MatchResultsFactory:
    def __init__(self, player_one_name, player_two_name):
        self.player_one_name = player_one_name
        self.player_two_name = player_two_name
        self.subscribers = []

    def create(self):
        match_result = MatchResults(self.player_one_name, self.player_two_name)
        if self.subscribers:
            match_result = ObservableMatchResults(match_result)
            for subscriber in self.subscribers:
                match_result.subscribe(subscriber)
        return match_result

    def subscribe(self, subscriber):
        self.subscribers.append(subscriber)
