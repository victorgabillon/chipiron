"""Morpion starting-position args."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from atomheart.games.morpion.state import Variant as MorpionVariant
from valanga import StateTag

from chipiron.environments.morpion.tags import MorpionStartTag
from chipiron.environments.starting_position import StartingPositionArgs


class MorpionStartingPositionArgsType(StrEnum):
    """Kinds of Morpion starting-position args."""

    STANDARD = "morpion_standard"


@dataclass(frozen=True)
class MorpionStandardStartingPositionArgs(StartingPositionArgs):
    """Build the canonical Morpion starting position."""

    type: Literal[MorpionStartingPositionArgsType.STANDARD] = (
        MorpionStartingPositionArgsType.STANDARD
    )
    variant: MorpionVariant = MorpionVariant.TOUCHING_5T

    def get_start_tag(self) -> StateTag:
        """Return standard Morpion start tag."""
        return MorpionStartTag(variant=self.variant)
