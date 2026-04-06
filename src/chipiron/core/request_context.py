"""Request correlation context shared across action flows."""

from dataclasses import dataclass

from chipiron.core.roles import GameRole


@dataclass(frozen=True, slots=True)
class RequestContext:
    """Correlate an action request with the role expected to answer it."""

    request_id: int
    role_to_play: GameRole
