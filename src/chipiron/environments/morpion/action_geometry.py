"""Canonical point and unit-segment geometry for Morpion actions."""
# pyright: reportMissingImports=false

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from atomheart.games.morpion.dynamics import DIRECTIONS
from atomheart.games.morpion.state import norm_seg

if TYPE_CHECKING:
    from atomheart.games.morpion.state import Point, Segment

    from .types import MorpionAction


def morpion_action_points(
    action: MorpionAction,
) -> tuple[Point, Point, Point, Point, Point]:
    """Return the five ordered lattice points in one legal-action window."""
    direction_index, x0, y0, _missing_index = action
    dx, dy = DIRECTIONS[direction_index]
    return cast(
        "tuple[Point, Point, Point, Point, Point]",
        tuple((x0 + slot * dx, y0 + slot * dy) for slot in range(5)),
    )


def morpion_action_segments(
    action: MorpionAction,
) -> tuple[Segment, Segment, Segment, Segment]:
    """Return all four prospective unit segments in one action window."""
    points = morpion_action_points(action)
    return cast(
        "tuple[Segment, Segment, Segment, Segment]",
        tuple(norm_seg(points[index], points[index + 1]) for index in range(4)),
    )


def morpion_action_new_point(action: MorpionAction) -> Point:
    """Return the absent point represented by one legal action."""
    return morpion_action_points(action)[action[3]]


__all__ = [
    "morpion_action_new_point",
    "morpion_action_points",
    "morpion_action_segments",
]
