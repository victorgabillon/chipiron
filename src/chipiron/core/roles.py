"""Shared role and participant vocabulary for generic Chipiron layers.

Today Chipiron still orchestrates white/black games, so ``GameRole`` is an
alias of ``valanga.Color``. Using role-based names in generic code keeps the
current behavior while preparing later environment-driven refactors.
"""

from collections.abc import Mapping

from valanga import Color

type GameRole = Color
type ParticipantId = str
type RoleAssignment[T] = Mapping[GameRole, T]
type MutableRoleAssignment[T] = dict[GameRole, T]
