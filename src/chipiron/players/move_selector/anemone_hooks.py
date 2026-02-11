"""Anemone hooks implementations for Chipiron chess states."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from anemone.hooks.search_hooks import FeatureExtractor
from atomheart.board.valanga_adapter import ValangaChessState
from valanga import TurnState


@dataclass(frozen=True)
class ChessFeatureExtractor(FeatureExtractor):
    """Compute tactical and diagnostic features for chess states."""

    def features(self, state: TurnState) -> Mapping[str, Any]:
        """Return a stable set of optional features for priority checks."""
        if not isinstance(state, ValangaChessState):
            return {}

        tactical_threat = False
        board = getattr(state, "board", None)
        player_to_move = getattr(state, "player_to_move", None)

        if board is not None and player_to_move is not None and hasattr(board, "is_attacked"):
            try:
                tactical_threat = bool(board.is_attacked(not player_to_move))
            except Exception:
                tactical_threat = False

        return {
            "tactical_threat": tactical_threat,
        }
