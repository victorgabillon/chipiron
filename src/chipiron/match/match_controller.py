from collections.abc import Callable

import valanga

from chipiron.core.request_context import RequestContext
from chipiron.displays.gui_protocol import (
    HumanActionChosen,
    Scope,
    UpdNeedHumanAction,
    UpdNoHumanActionPending,
)
from chipiron.games.game.game_manager import GameManager
from chipiron.match.domain_events import ActionApplied, IllegalAction, MatchOver, NeedAction
from chipiron.players.communications.player_message import EvMove, PlayerRequest
from chipiron.utils.small_tools import unique_int_from_list
from chipiron.utils.logger import chipiron_logger


class MatchController:
    """Async orchestration hub for GUI and player events."""

    def __init__(
        self,
        *,
        scope: Scope,
        game_manager: GameManager,
        engine_request_by_color: dict[valanga.Color, Callable[[PlayerRequest], None]],
        human_colors: set[valanga.Color],
    ) -> None:
        self.scope = scope
        self.game_manager = game_manager
        self.engine_request_by_color = engine_request_by_color
        self.human_colors = human_colors
        self.pending_color: valanga.Color | None = None
        self.pending_request_id: int | None = None

    def start(self) -> None:
        outs = self.game_manager.start_match_sync(self.scope)
        self._handle_outputs(outs)

    def request_next_action(self) -> None:
        """Request an action for the current position without resetting match ids."""
        outs = self.game_manager.need_action_now()
        self._handle_outputs(outs)

    def clear_pending(self) -> None:
        """Clear pending action correlation state."""
        self.pending_color = None
        self.pending_request_id = None

    def _handle_outputs(self, events: list[object]) -> None:
        for ev in events:
            if isinstance(ev, NeedAction):
                self.pending_color = ev.color
                self.pending_request_id = ev.request_id

                if ev.color in self.human_colors:
                    payload = UpdNeedHumanAction(
                        ctx=RequestContext(ev.request_id, ev.color),
                        state_tag=ev.state.tag,
                    )
                    self.game_manager.game.publish_update(payload)
                    continue

                if ev.color not in self.engine_request_by_color:
                    continue

                ctx = RequestContext(ev.request_id, ev.color)
                merged_seed = unique_int_from_list(
                    [self.game_manager.game.seed, self.game_manager.game.ply]
                )
                if merged_seed is None:
                    merged_seed = int(self.game_manager.game.ply)
                base_request = self.game_manager.game.player_encoder.make_move_request(
                    state=self.game_manager.game.state,
                    seed=merged_seed,
                    scope=self.scope,
                )
                request = PlayerRequest(
                    schema_version=base_request.schema_version,
                    scope=base_request.scope,
                    seed=base_request.seed,
                    state=base_request.state,
                    ctx=ctx,
                )
                self.engine_request_by_color[ev.color](request)

            elif isinstance(ev, MatchOver):
                chipiron_logger.info("Match over for scope=%s", ev.scope)
                self.clear_pending()
                self.game_manager.game.publish_update(UpdNoHumanActionPending())
                self.game_manager.game.notify_display()
                continue

            elif isinstance(ev, IllegalAction):
                chipiron_logger.info(
                    "Illegal action rejected scope=%s color=%s request_id=%s reason=%s",
                    ev.scope,
                    ev.color,
                    ev.request_id,
                    ev.reason,
                )
                continue

            elif isinstance(ev, ActionApplied):
                self.game_manager.game.publish_update(UpdNoHumanActionPending())
                self.game_manager.game.notify_display()

    def handle_player_action(self, ev_move: EvMove) -> None:
        if ev_move.ctx is None:
            return
        if ev_move.ctx.request_id != self.pending_request_id:
            return
        if ev_move.ctx.color_to_play != self.pending_color:
            return

        try:
            action = self.game_manager.game.dynamics.action_from_name(
                self.game_manager.game.state,
                ev_move.branch_name,
            )
        except (KeyError, ValueError) as exc:
            chipiron_reason = f"invalid player action '{ev_move.branch_name}': {exc}"
            outs = [
                IllegalAction(
                    scope=self.scope,
                    color=ev_move.ctx.color_to_play,
                    request_id=ev_move.ctx.request_id,
                    action=ev_move.branch_name,
                    reason=chipiron_reason,
                ),
                NeedAction(
                    scope=self.scope,
                    color=self.game_manager.game.state.turn,
                    request_id=self.pending_request_id,
                    state=self.game_manager.game.state,
                ),
            ]
            self._handle_outputs(outs)
            return

        outs = self.game_manager.propose_action_sync(
            scope=self.scope,
            color=ev_move.ctx.color_to_play,
            request_id=ev_move.ctx.request_id,
            action=action,
        )
        self._handle_outputs(outs)

    def handle_human_action(self, human_action: HumanActionChosen) -> None:
        if self.pending_color is None or self.pending_request_id is None:
            return

        if human_action.ctx is not None:
            if human_action.ctx.request_id != self.pending_request_id:
                return
            if human_action.ctx.color_to_play != self.pending_color:
                return
        if (
            human_action.corresponding_state_tag is not None
            and human_action.corresponding_state_tag != self.game_manager.game.state.tag
        ):
            return

        state = self.game_manager.game.state
        try:
            action = self.game_manager.game.dynamics.action_from_name(
                state, human_action.action_name
            )
        except (KeyError, ValueError) as exc:
            outs = [
                IllegalAction(
                    scope=self.scope,
                    color=self.pending_color,
                    request_id=self.pending_request_id,
                    action=human_action.action_name,
                    reason=f"invalid human action '{human_action.action_name}': {exc}",
                ),
                NeedAction(
                    scope=self.scope,
                    color=state.turn,
                    request_id=self.pending_request_id,
                    state=state,
                ),
            ]
            self._handle_outputs(outs)
            return

        outs = self.game_manager.propose_action_sync(
            scope=self.scope,
            color=self.pending_color,
            request_id=self.pending_request_id,
            action=action,
        )
        self._handle_outputs(outs)
