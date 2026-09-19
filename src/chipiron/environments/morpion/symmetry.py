"""Exact rooted D4 transformations of complete Morpion positions.

The fixed start is centered at (-1/2, -1/2), not the origin. Atomheart owns
the point and action conventions, including reversal of missing-dot slots.
Directional usage values encode endpoint/interior occupancy, so reflection
permutes their direction keys without changing the values.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from typing import TYPE_CHECKING

from atomheart.games.morpion.canonical import apply_rooted_symmetry
from atomheart.games.morpion.dynamics import transform_action_rooted
from atomheart.games.morpion.state import norm_seg

if TYPE_CHECKING:
    from chipiron.environments.morpion.types import MorpionState


@lru_cache(maxsize=8)
def direction_permutation(symmetry: int) -> tuple[int, ...]:
    """Map the four undirected line orientations using canonical actions."""
    return tuple(transform_action_rooted((d, 0, 0, 0), symmetry)[0] for d in range(4))


def transform_morpion_state(state: MorpionState, symmetry: int) -> MorpionState:
    """Transform all state geometry while preserving value-relevant metadata.

    Conversion must run after this operation. Canonical token order, action
    slots, relation slots and any converter feature are then recomputed from
    the transformed state, including the converter's normal truncation rule.
    """
    directions = direction_permutation(symmetry)
    if symmetry == 0:
        return state
    moves = []
    for x1, y1, x2, y2 in state.played_moves:
        a, b = norm_seg(
            apply_rooted_symmetry((x1, y1), symmetry),
            apply_rooted_symmetry((x2, y2), symmetry),
        )
        moves.append((*a, *b))
    return replace(
        state,
        points=frozenset(apply_rooted_symmetry(p, symmetry) for p in state.points),
        used_unit_segments=frozenset(
            norm_seg(
                apply_rooted_symmetry(a, symmetry), apply_rooted_symmetry(b, symmetry)
            )
            for a, b in state.used_unit_segments
        ),
        dir_usage_entries=tuple(
            sorted(
                ((apply_rooted_symmetry(p, symmetry), directions[d]), usage)
                for (p, d), usage in state.dir_usage_entries
            )
        ),
        played_moves=tuple(sorted(moves)),
    )
