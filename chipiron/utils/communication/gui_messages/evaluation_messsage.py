from dataclasses import dataclass
from typing import Any


@dataclass
class EvaluationMessage:
    evaluation_stock: Any
    evaluation_chipiron: Any
    evaluation_player_black: Any = None
    evaluation_player_white: Any = None
