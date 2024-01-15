from dataclasses import dataclass
from typing import Any
from chipiron.games.game.game_playing_status import PlayingStatus


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
    evaluation_player_one: Any = None
    evaluation__player_two: Any = None


@dataclass
class EvaluationMessage:
    evaluation_stock: Any
    evaluation_chipiron: Any
    evaluation_player_black: Any = None
    evaluation_player_white: Any = None


@dataclass
class MatchResultsMessage:
    match_results: dict
