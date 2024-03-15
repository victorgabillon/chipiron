from dataclasses import dataclass
from typing import Any

from chipiron.games.game.game_playing_status import PlayingStatus
from chipiron.games.match.match_results import MatchResults


@dataclass
class GameStatusMessage:
    status: PlayingStatus


@dataclass
class BackMessage:
    ...


@dataclass
class EvaluationMessage:
    evaluation_stock: Any
    evaluation_chipiron: Any
    evaluation_player_black: Any = None
    evaluation_player_white: Any = None


@dataclass
class MatchResultsMessage:
    match_results: MatchResults
