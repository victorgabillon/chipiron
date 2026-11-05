"""Exact complete-state opaque keys; cached bytes are never deserialized."""

# Exact types deliberately exclude bool/int coercion and altered subclasses.
# pylint: disable=unidiomatic-typecheck

import dataclasses
import marshal

from atomheart.games.morpion.state import Variant
from valanga import SOLO

from chipiron.environments.morpion.types import MorpionState

FIELDS = (
    "points",
    "used_unit_segments",
    "dir_usage_entries",
    "played_moves",
    "moves",
    "variant",
    "turn",
    "is_terminal",
)


def _point(v: object) -> bool:
    return type(v) is tuple and len(v) == 2 and all(type(i) is int for i in v)


def state_payload(s: MorpionState) -> bytes:
    """Encode every field with exact types, without decoding cache data."""
    if tuple(f.name for f in dataclasses.fields(MorpionState)) != FIELDS:
        message = "Unsupported state schema"
        raise TypeError(message)
    # Refuse unsupported objects; do not canonicalize orientation/history or merge
    # int/bool/subclass semantics. marshal is used only to WRITE/compare bytes;
    # no untrusted marshal decoding, pickle, node ID or D4 identity is consumed.
    if (
        type(s) is not MorpionState
        or type(s.points) is not frozenset
        or type(s.used_unit_segments) is not frozenset
    ):
        message = "unsupported exact state type"
        raise TypeError(message)
    if not all(_point(p) for p in s.points):
        message = "unsupported point"
        raise TypeError(message)
    if not all(
        type(e) is tuple and len(e) == 2 and _point(e[0]) and _point(e[1])
        for e in s.used_unit_segments
    ):
        message = "unsupported segment"
        raise TypeError(message)
    if type(s.dir_usage_entries) is not tuple or not all(
        type(e) is tuple
        and len(e) == 2
        and type(e[0]) is tuple
        and len(e[0]) == 2
        and _point(e[0][0])
        and type(e[0][1]) is int
        and type(e[1]) is int
        for e in s.dir_usage_entries
    ):
        message = "unsupported usage"
        raise TypeError(message)
    if type(s.played_moves) is not tuple or not all(
        type(m) is tuple and len(m) == 4 and all(type(v) is int for v in m)
        for m in s.played_moves
    ):
        message = "unsupported history"
        raise TypeError(message)
    if (
        type(s.moves) is not int
        or type(s.variant) is not Variant
        or type(s.turn) is not type(SOLO)
        or type(s.is_terminal) is not bool
    ):
        message = "unsupported scalar fields"
        raise TypeError(message)
    # Version2 has no object-reference memoization: equal immutable contents do
    # not depend on Python aliasing/insertion history. Sorted sets only; ordered
    # history and dir-usage tuples stay ordered. Integers have no bounded range.
    wire = marshal.dumps(
        (
            "morpion-evaluator-complete-state-v1",
            tuple(sorted(s.points)),
            tuple(sorted(s.used_unit_segments)),
            s.dir_usage_entries,
            s.played_moves,
            s.moves,
            s.variant.value,
            s.turn.value,
            s.is_terminal,
        ),
        2,
    )
    if len(wire) > 128 * 1024:
        message = "state byte budget"
        raise ValueError(message)
    return wire
