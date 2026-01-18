from __future__ import annotations

from valanga import Color


def to_valanga_color(color: Color | bool) -> Color:
    """Convert a python-chess style color (bool) to `valanga.Color`.

    Chipiron's generic layers use `valanga.Color` exclusively.
    python-chess uses `chess.WHITE == True` and `chess.BLACK == False`.
    """

    if isinstance(color, Color):
        return color
    if isinstance(color, bool):
        return Color.WHITE if color else Color.BLACK
    raise TypeError(f"Unsupported color type: {type(color)!r}")
