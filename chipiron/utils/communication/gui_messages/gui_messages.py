from dataclasses import dataclass

from chipiron.games.match.match_results import MatchResults


@dataclass
class MatchResultsMessage:
    match_results: MatchResults
