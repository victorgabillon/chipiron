"""Runtime handler for player requests.

This module is intentionally game-agnostic: it extracts the snapshot from the
incoming `PlayerRequest`, asks the `GamePlayer` to select a move, and emits a
`PlayerEvent`.
"""

from typing import TYPE_CHECKING

from valanga.policy import NotifyProgressCallable

from chipiron.core.roles import GameRole
from chipiron.displays.gui_protocol import Scope
from chipiron.players.communications.player_message import (
    EvMove,
    EvProgress,
    PlayerEvent,
    PlayerRequest,
    TurnStatePlusHistory,
)
from chipiron.players.game_player import GamePlayer
from chipiron.utils.communication.mailbox import MainMailboxMessage
from chipiron.utils.logger import chipiron_logger
from chipiron.utils.queue_protocols import PutQueue

if TYPE_CHECKING:
    from valanga.policy import Recommendation


def handle_player_request[StateSnapT, RuntimeStateT](
    *,
    request: PlayerRequest[StateSnapT],
    game_player: GamePlayer[StateSnapT, RuntimeStateT],
    out_queue: PutQueue[MainMailboxMessage],
) -> None:
    """Handle player request."""
    state: TurnStatePlusHistory[StateSnapT] = request.state

    if state.role_to_play != game_player.role:
        chipiron_logger.warning(
            "Rejecting PlayerRequest: wrong role. request_role=%s player_color=%s scope=%s",
            state.role_to_play,
            game_player.role,
            request.scope,
        )
        return

    def make_progress_cb(
        scope: Scope, player_role: GameRole, queue_out: PutQueue[MainMailboxMessage]
    ) -> NotifyProgressCallable:
        def cb(progress_percent: int) -> None:
            ev = EvProgress(
                progress_percent=progress_percent, player_role=player_role
            )
            queue_out.put(
                PlayerEvent(
                    schema_version=1,
                    scope=scope,
                    payload=ev,
                )
            )

        return cb

    progress_cb = make_progress_cb(request.scope, state.role_to_play, out_queue)

    rec: Recommendation = game_player.select_move_from_snapshot(
        snapshot=state.snapshot, seed=request.seed, notify_percent_function=progress_cb
    )

    ev = EvMove(
        branch_name=rec.recommended_name,
        corresponding_state_tag=state.current_state_tag,
        ctx=request.ctx,
        player_name=game_player.player.get_id(),
        player_role=game_player.role,
        evaluation=getattr(rec, "evaluation", None),
    )
    out_queue.put(PlayerEvent(schema_version=1, scope=request.scope, payload=ev))
