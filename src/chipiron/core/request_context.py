"""Request correlation context shared across action flows."""

from dataclasses import dataclass

from valanga import Color


@dataclass(frozen=True, slots=True)
class RequestContext:
    """Correlates an action request and a player response."""

    request_id: int
    color_to_play: Color
