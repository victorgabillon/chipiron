"""Checkers starting position args."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from valanga import StateTag

from chipiron.environments.checkers.tags import CheckersStartTag
from chipiron.environments.checkers.types import CheckersState
from chipiron.environments.starting_position import StartingPositionArgs


class CheckersStartingPositionArgsType(StrEnum):
    """Kinds of checkers starting-position args."""

    STANDARD = "checkers_standard"
    TEXT = "checkers_text"


@dataclass(frozen=True)
class CheckersStandardStartingPositionArgs(StartingPositionArgs):
    """Build the canonical standard checkers starting position."""

    type: Literal[CheckersStartingPositionArgsType.STANDARD] = (
        CheckersStartingPositionArgsType.STANDARD
    )

    def get_start_tag(self) -> StateTag:
        """Return standard checkers start tag."""
        return CheckersStartTag(text=CheckersState.standard().to_text())


@dataclass(frozen=True)
class CheckersTextStartingPositionArgs(StartingPositionArgs):
    """Build a checkers starting position from serialized text."""

    type: Literal[CheckersStartingPositionArgsType.TEXT] = (
        CheckersStartingPositionArgsType.TEXT
    )
    text: str = ""

    def get_start_tag(self) -> StateTag:
        """Return explicit checkers start tag."""
        return CheckersStartTag(text=self.text)
