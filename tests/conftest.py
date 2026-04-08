"""Pytest-local environment setup for PR1 characterization tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import Any

TESTS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TESTS_ROOT.parent
TEST_OUTPUT_DIR = Path("/tmp/chipiron-test-output")

if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

for path in (
    REPO_ROOT / "src",
    REPO_ROOT.parent / "atomheart" / "src",
    REPO_ROOT.parent / "anemone" / "src",
):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _patch_valanga_compat() -> None:
    """Backfill newer Valanga APIs when the installed package is older."""
    try:
        import valanga
        import valanga.evaluations
        import valanga.game
        import valanga.over_event
    except ImportError:
        return

    color_type = valanga.Color

    class Outcome(Enum):
        """Compatibility outcome enum matching the newer Valanga API."""

        WIN = "win"
        DRAW = "draw"
        UNKNOWN = "unknown"

    class Certainty(StrEnum):
        """Compatibility certainty enum matching the newer Valanga API."""

        ESTIMATE = "estimate"
        FORCED = "forced"
        TERMINAL = "terminal"

    @dataclass(slots=True, frozen=True)
    class Value:
        """Compatibility state-evaluation value used by Chipiron/Anemone."""

        score: float
        certainty: Certainty
        over_event: object | None = None

    @dataclass(slots=True, frozen=True)
    class OverEvent:
        """Compatibility over-event exposing both old and new access patterns."""

        outcome: Outcome = Outcome.UNKNOWN
        termination: Enum | None = None
        winner: color_type | None = None

        def __class_getitem__(cls, _item: object) -> type["OverEvent"]:
            """Support ``OverEvent[T]`` annotations used by newer packages."""
            return cls

        @property
        def how_over(self) -> object:
            """Expose the legacy field name expected by older GUI formatting."""
            if self.outcome is Outcome.WIN:
                return getattr(valanga.over_event, "HowOver").WIN
            if self.outcome is Outcome.DRAW:
                return getattr(valanga.over_event, "HowOver").DRAW
            return getattr(valanga.over_event, "HowOver").DO_NOT_KNOW_OVER

        @property
        def who_is_winner(self) -> object:
            """Expose the legacy field name expected by older GUI formatting."""
            winner_enum = getattr(valanga.over_event, "Winner")
            if self.winner == color_type.WHITE:
                return winner_enum.WHITE
            if self.winner == color_type.BLACK:
                return winner_enum.BLACK
            return winner_enum.NO_KNOWN_WINNER

        def is_over(self) -> bool:
            """Return whether this event represents a finished game."""
            return self.outcome in {Outcome.WIN, Outcome.DRAW}

        def is_draw(self) -> bool:
            """Return whether the outcome is a draw."""
            return self.outcome is Outcome.DRAW

        def is_win(self) -> bool:
            """Return whether the outcome is a win."""
            return self.outcome is Outcome.WIN

        def is_win_for(self, role: object) -> bool:
            """Return whether the provided role is the winner."""
            return self.outcome is Outcome.WIN and self.winner == role

    if not hasattr(valanga, "Role"):
        valanga.Role = color_type
    if not hasattr(valanga.game, "Role"):
        valanga.game.Role = color_type

    if not hasattr(valanga, "SoloRole"):
        valanga.SoloRole = str
    if not hasattr(valanga.game, "SoloRole"):
        valanga.game.SoloRole = str

    if not hasattr(valanga, "SOLO"):
        valanga.SOLO = "SOLO"
    if not hasattr(valanga.game, "SOLO"):
        valanga.game.SOLO = valanga.SOLO

    if not hasattr(valanga, "Outcome"):
        valanga.Outcome = Outcome
    if not hasattr(valanga.over_event, "Outcome"):
        valanga.over_event.Outcome = Outcome

    if not hasattr(valanga.evaluations, "Certainty"):
        valanga.evaluations.Certainty = Certainty
    if not hasattr(valanga.evaluations, "Value"):
        valanga.evaluations.Value = Value

    # Replace the old non-generic over-event with a compatibility wrapper when needed.
    if not hasattr(valanga.over_event.OverEvent, "__class_getitem__"):
        valanga.over_event.OverEvent = OverEvent
        valanga.OverEvent = OverEvent


_patch_valanga_compat()

TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("CHIPIRON_OUTPUT_DIR", str(TEST_OUTPUT_DIR))
os.environ.setdefault(
    "ML_FLOW_URI_PATH",
    f"sqlite:///{(TEST_OUTPUT_DIR / 'mlruns.db').as_posix()}",
)
os.environ.setdefault(
    "ML_FLOW_URI_PATH_TEST",
    f"sqlite:///{(TEST_OUTPUT_DIR / 'mlruns_test.db').as_posix()}",
)
