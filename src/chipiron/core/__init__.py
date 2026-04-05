"""Core transport/domain primitives shared across chipiron."""

from .request_context import RequestContext
from .roles import (
    GameRole,
    MutableRoleAssignment,
    ParticipantId,
    RoleAssignment,
    format_game_role,
)

__all__ = [
    "GameRole",
    "MutableRoleAssignment",
    "ParticipantId",
    "RequestContext",
    "RoleAssignment",
    "format_game_role",
]
