"""Anemone hooks implementations for Chipiron chess states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from anemone.hooks.search_hooks import FeatureExtractor
from atomheart.utils.color import valanga_color_to_chess

if TYPE_CHECKING:
    from collections.abc import Mapping

    from valanga import TurnState


class BoardWithAttackCheck(Protocol):
    """Board protocol with attack-query capability."""

    def is_attacked(self, color: object) -> bool:
        """Return whether a side is currently attacked."""
        ...


class ChessLikeState(Protocol):
    """State protocol with chess board and turn accessors."""

    @property
    def board(self) -> BoardWithAttackCheck:
        """Return board-like object with attack checks."""
        ...

    @property
    def turn(self) -> object:
        """Return side to move."""
        ...


@dataclass(frozen=True)
class ChessFeatureExtractor(FeatureExtractor):
    """Compute tactical and diagnostic features for chess states."""

    def features(self, state: TurnState) -> Mapping[str, Any]:
        """Return a stable set of optional features for priority checks."""
        board = getattr(state, "board", None)
        turn = getattr(state, "turn", None)

        if board is None or turn is None or not hasattr(board, "is_attacked"):
            return {}

        try:
            tactical_threat = bool(board.is_attacked(not valanga_color_to_chess(turn)))
        except Exception:
            return {}

        return {
            "tactical_threat": tactical_threat,
        }
