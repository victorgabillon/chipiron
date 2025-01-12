"""
Module for the EvaluationMessage class.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class EvaluationMessage:
    """
    Represents a message containing evaluation information.

    Attributes:
        evaluation_stock (Any): The evaluation for the stock.
        evaluation_chipiron (Any): The evaluation for the chipiron.
        evaluation_player_black (Any, optional): The evaluation for the black player. Defaults to None.
        evaluation_player_white (Any, optional): The evaluation for the white player. Defaults to None.
    """

    evaluation_stock: Any
    evaluation_chipiron: Any
    evaluation_player_black: Any = None
    evaluation_player_white: Any = None
