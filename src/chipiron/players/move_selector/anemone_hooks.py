"""Anemone hooks implementations for Chipiron chess states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from anemone.hooks.search_hooks import FeatureExtractor
from atomheart.board.valanga_adapter import ValangaChessState
from atomheart.utils.color import valanga_color_to_chess

if TYPE_CHECKING:
    from collections.abc import Mapping

    from valanga import Color, TurnState


@dataclass(frozen=True)
class ChessFeatureExtractor(FeatureExtractor):
    """Compute tactical and diagnostic features for chess states."""

    def features(self, state: TurnState) -> Mapping[str, Any]:
        """Return a stable set of optional features for priority checks."""
        if not isinstance(state, ValangaChessState):
            return {}

        tactical_threat = False

        assert isinstance(state, ValangaChessState)
        player_to_move: Color = state.turn

        tactical_threat = bool(
            state.board.is_attacked(not valanga_color_to_chess(player_to_move))
        )

        return {
            "tactical_threat": tactical_threat,
        }
