"""Morpion start-tag representations."""

from dataclasses import dataclass

from atomheart.games.morpion.state import Variant as MorpionVariant


@dataclass(frozen=True, slots=True)
class MorpionStartTag:
    """Lossless Morpion starting tag."""

    variant: MorpionVariant = MorpionVariant.TOUCHING_5T
