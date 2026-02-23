"""Checkers start-tag representations."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CheckersStartTag:
    """Lossless serialized checkers start position."""

    text: str
