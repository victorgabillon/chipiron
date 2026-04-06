"""Match result aggregation with role- and participant-aware reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from chipiron.games.domain.game.final_game_result import (
    FinalGameResult,
    GameReport,
    RoleOutcome,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from atomheart.games.chess.move import MoveUci

    from chipiron.core.roles import ParticipantId


class MatchResultsError(ValueError):
    """Base error for match result handling."""


class InvalidMatchResultError(MatchResultsError):
    """Raised when a match result is not supported."""

    def __init__(self, game_result: FinalGameResult) -> None:
        """Initialize the error with the unsupported result."""
        super().__init__(f"Unsupported game result: {game_result}")


class UnknownWhitePlayerError(MatchResultsError):
    """Raised when the white player's identity is unknown."""

    def __init__(self, white_player_name_id: str) -> None:
        """Initialize the error with the unknown player id."""
        super().__init__(f"Unknown white player id: {white_player_name_id}")


class IncompleteRoleResultError(MatchResultsError):
    """Raised when a role result payload cannot be aggregated."""

    def __init__(self, missing_role: str) -> None:
        """Initialize the error with the missing role label."""
        super().__init__(f"Missing role result for role: {missing_role}")


class MissingResultPayloadError(MatchResultsError):
    """Raised when neither the generic nor the legacy result payload is present."""

    def __init__(self) -> None:
        """Initialize the error for missing result payload data."""
        super().__init__(
            "Either `game_report` or the legacy white-player result pair is required."
        )


def make_participant_stats() -> dict[ParticipantId, ParticipantResultStats]:
    """Build the default participant stats mapping."""
    return {}


@dataclass(slots=True)
class ParticipantResultStats:
    """Aggregate win/loss/draw counts for a participant across a match."""

    wins: int = 0
    losses: int = 0
    draws: int = 0
    unknown: int = 0


@dataclass(frozen=True, slots=True)
class SimpleResults:
    """Compact match summary with ordered participant stats."""

    participant_order: tuple[ParticipantId, ...]
    stats_by_participant: Mapping[ParticipantId, ParticipantResultStats]
    draws: int
    games_played: int

    @property
    def wins_by_participant(self) -> dict[ParticipantId, int]:
        """Return per-participant win totals."""
        return {
            participant_id: self.stats_by_participant[participant_id].wins
            for participant_id in self.participant_order
        }

    @property
    def player_one_wins(self) -> int:
        """Backward-compatible accessor for the first configured participant."""
        if not self.participant_order:
            return 0
        return self.stats_by_participant[self.participant_order[0]].wins

    @property
    def player_two_wins(self) -> int:
        """Backward-compatible accessor for the second configured participant."""
        if len(self.participant_order) < 2:
            return 0
        return self.stats_by_participant[self.participant_order[1]].wins


class IMatchResults(Protocol):
    """Interface for match results."""

    def add_result_one_game(
        self,
        *,
        game_report: GameReport | None = None,
        white_player_name_id: str | None = None,
        game_result: FinalGameResult | None = None,
    ) -> None:
        """Add the result of one game to the match results."""

    def get_simple_result(self) -> SimpleResults:
        """Return the simple results of the match."""
        ...

    def __str__(self) -> str:
        """Return a string representation of the match results."""
        ...

    def finish(self) -> None:
        """Finishes the match and marks it as finished."""


@dataclass
class MatchResults:
    """Represents match results for one or more ordered participants."""

    player_one_name_id: ParticipantId | None = None
    player_two_name_id: ParticipantId | None = None
    participant_ids: tuple[ParticipantId, ...] = ()
    number_of_games: int = 0
    draw_games: int = 0
    stats_by_participant: dict[ParticipantId, ParticipantResultStats] = field(
        default_factory=make_participant_stats
    )
    match_finished: bool = False

    def __post_init__(self) -> None:
        """Initialize the ordered participant registry."""
        if not self.participant_ids:
            self.participant_ids = tuple(
                participant_id
                for participant_id in (self.player_one_name_id, self.player_two_name_id)
                if participant_id is not None
            )
        if self.player_one_name_id is None and self.participant_ids:
            self.player_one_name_id = self.participant_ids[0]
        if self.player_two_name_id is None and len(self.participant_ids) > 1:
            self.player_two_name_id = self.participant_ids[1]
        for participant_id in self.participant_ids:
            self._ensure_participant(participant_id)

    def _ensure_participant(self, participant_id: ParticipantId) -> None:
        """Ensure a participant has an entry in the ordered registry."""
        if participant_id not in self.stats_by_participant:
            self.stats_by_participant[participant_id] = ParticipantResultStats()
        if participant_id not in self.participant_ids:
            self.participant_ids = (*self.participant_ids, participant_id)

    def get_player_one_wins(self) -> int:
        """Return the number of wins for player one."""
        if self.player_one_name_id is None:
            return 0
        return self.stats_by_participant[self.player_one_name_id].wins

    def get_player_two_wins(self) -> int:
        """Return the number of wins for player two."""
        if self.player_two_name_id is None:
            return 0
        return self.stats_by_participant[self.player_two_name_id].wins

    def get_draws(self) -> int:
        """Return the number of drawn games."""
        return self.draw_games

    def get_simple_result(self) -> SimpleResults:
        """Return the compact match summary."""
        return SimpleResults(
            participant_order=self.participant_ids,
            stats_by_participant=self.stats_by_participant,
            draws=self.get_draws(),
            games_played=self.number_of_games,
        )

    def add_result_one_game(
        self,
        *,
        game_report: GameReport | None = None,
        white_player_name_id: str | None = None,
        game_result: FinalGameResult | None = None,
    ) -> None:
        """Add the result of one game using the new or legacy reporting shape."""
        if game_report is not None:
            self._add_game_report(game_report)
            return
        if white_player_name_id is None or game_result is None:
            raise MissingResultPayloadError
        self._add_legacy_result(white_player_name_id, game_result)

    def _add_legacy_result(
        self, white_player_name_id: str, game_result: FinalGameResult
    ) -> None:
        """Support legacy two-player color-shaped result accounting."""
        self.number_of_games += 1
        if game_result == FinalGameResult.DRAW:
            self.draw_games += 1
            for participant_id in self.participant_ids:
                self._ensure_participant(participant_id)
                self.stats_by_participant[participant_id].draws += 1
            return

        if white_player_name_id == self.player_one_name_id:
            winner = (
                self.player_one_name_id
                if game_result == FinalGameResult.WIN_FOR_WHITE
                else self.player_two_name_id
            )
        elif white_player_name_id == self.player_two_name_id:
            winner = (
                self.player_two_name_id
                if game_result == FinalGameResult.WIN_FOR_WHITE
                else self.player_one_name_id
            )
        else:
            raise UnknownWhitePlayerError(white_player_name_id)

        if winner is None:
            raise InvalidMatchResultError(game_result)
        loser_ids = [pid for pid in self.participant_ids if pid != winner]
        self._ensure_participant(winner)
        self.stats_by_participant[winner].wins += 1
        for participant_id in loser_ids:
            self._ensure_participant(participant_id)
            self.stats_by_participant[participant_id].losses += 1

    def _add_game_report(self, game_report: GameReport) -> None:
        """Aggregate one role-aware game report."""
        self.number_of_games += 1
        normalized_result_by_role = dict(game_report.result_by_role)
        if game_report.result_by_role and all(
            outcome is RoleOutcome.DRAW
            for outcome in normalized_result_by_role.values()
        ):
            self.draw_games += 1

        for role_label, participant_id in game_report.participant_id_by_role.items():
            self._ensure_participant(participant_id)
            if role_label not in normalized_result_by_role:
                raise IncompleteRoleResultError(role_label)
            participant_stats = self.stats_by_participant[participant_id]
            outcome = normalized_result_by_role[role_label]
            if outcome is RoleOutcome.WIN:
                participant_stats.wins += 1
            elif outcome is RoleOutcome.LOSS:
                participant_stats.losses += 1
            elif outcome is RoleOutcome.DRAW:
                participant_stats.draws += 1
            else:
                participant_stats.unknown += 1

    def finish(self) -> None:
        """Finishes the match and marks it as finished."""
        self.match_finished = True

    def __str__(self) -> str:
        """Return a readable representation of the aggregated participant stats."""
        summary = self.get_simple_result()
        header = "Main result: " + ", ".join(
            f"{participant_id} wins {summary.stats_by_participant[participant_id].wins}"
            for participant_id in summary.participant_order
        )
        header += f", draws {summary.draws}\n"
        details = "\n".join(
            (
                f"{participant_id}: "
                f"Wins {summary.stats_by_participant[participant_id].wins}, "
                f"Losses {summary.stats_by_participant[participant_id].losses}, "
                f"Draws {summary.stats_by_participant[participant_id].draws}, "
                f"Unknown {summary.stats_by_participant[participant_id].unknown}"
            )
            for participant_id in summary.participant_order
        )
        return header + details


@dataclass
class MatchReport:
    """Represents a match report containing the match results and move history."""

    match_results: MatchResults
    match_move_history: dict[int, list[MoveUci]]
