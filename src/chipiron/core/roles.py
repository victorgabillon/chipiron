"""Shared role and participant vocabulary for generic Chipiron layers.

Current shipped games still use ``valanga.Color`` roles, but generic runtime
code should be able to iterate over any finite set of hashable roles declared
by an environment.
"""

from collections.abc import Hashable, Mapping

type GameRole = Hashable
type ParticipantId = str
type RoleAssignment[T] = Mapping[GameRole, T]
type MutableRoleAssignment[T] = dict[GameRole, T]


def format_game_role(role: GameRole) -> str:
    """Return a readable label for logging generic game roles."""
    role_name = getattr(role, "name", None)
    if isinstance(role_name, str):
        return role_name.replace("_", " ").title()
    return str(role)
