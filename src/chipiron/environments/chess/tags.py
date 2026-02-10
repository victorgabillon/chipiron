"""Module for tags."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChessStartTag:
    """Chessstarttag implementation."""

    fen: str
