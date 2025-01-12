"""
Module for the ProgressMessage class.
"""

from dataclasses import dataclass

import chess


@dataclass
class PlayerProgressMessage:
    """
    Represents a message containing evaluation information.

    Attributes:
        evaluation_stock (Any): The evaluation for the stock.
    """

    progress_percent: int | None
    player_color: chess.Color
