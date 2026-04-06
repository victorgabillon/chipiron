"""Shared participant selection model for the launcher."""

from dataclasses import dataclass

from chipiron.players.player_ids import PlayerConfigTag


@dataclass(frozen=True, slots=True)
class ParticipantSelection:
    """Participant choice captured by the launcher."""

    player_tag: PlayerConfigTag
    strength: int | None = None
