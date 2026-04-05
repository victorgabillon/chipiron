"""Controller that routes match domain events between UI and players."""

from collections.abc import Callable, Sequence
from typing import Any

from chipiron.core.request_context import RequestContext
from chipiron.core.roles import GameRole, RoleAssignment
from chipiron.displays.gui_protocol import (
    HumanActionChosen,
    Scope,
    UpdNeedHumanAction,
    UpdNoHumanActionPending,
)
from chipiron.games.domain.game.game_manager import GameManager
from chipiron.games.runtime.orchestrator.domain_events import (
    ActionApplied,
    IllegalAction,
    MatchEvent,
    MatchOver,
    NeedAction,
)
from chipiron.players.communications.player_message import EvMove, PlayerRequest
from chipiron.utils.logger import chipiron_logger
from chipiron.utils.small_tools import unique_int_from_list


class MatchController:
    """Async orchestration hub for GUI and player events."""

    def __init__(
        self,
        *,
        scope: Scope,
        game_manager: GameManager,
        engine_request_by_role: RoleAssignment[Callable[[PlayerRequest[Any]], None]],
        human_roles: set[GameRole],
    ) -> None:
        """Initialize controller dependencies and pending action state."""
        self.scope = scope
        self.game_manager = game_manager
        self.engine_request_by_role = engine_request_by_role
        self.human_roles = human_roles
        self.pending_role: GameRole | None = None
        self.pending_request_id: int | None = None

    def start(self) -> None:
        """Start a new match and dispatch resulting domain events."""
        outs = self.game_manager.start_match_sync(self.scope)
        self._handle_outputs(outs)

    def request_next_action(self) -> None:
        """Request an action for the current position without resetting match ids."""
        outs = self.game_manager.need_action_now()
        self._handle_outputs(outs)

    def clear_pending(self) -> None:
        """Clear pending action correlation state."""
        self.pending_role = None
        self.pending_request_id = None

    @property
    def pending_color(self) -> GameRole | None:
        """Backward-compatible alias while current runtime stays color-shaped."""
        return self.pending_role

    def _handle_outputs(self, events: Sequence[MatchEvent]) -> None:
        for ev in events:
            if isinstance(ev, NeedAction):
                self.pending_role = ev.role
                self.pending_request_id = ev.request_id

                if ev.role in self.human_roles:
                    payload = UpdNeedHumanAction(
                        ctx=RequestContext(ev.request_id, ev.role),
                        state_tag=ev.state.tag,
                    )
                    self.game_manager.game.publish_update(payload)
                    continue

                if ev.role not in self.engine_request_by_role:
                    continue

                ctx = RequestContext(ev.request_id, ev.role)
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
                self.engine_request_by_role[ev.role](request)

            elif isinstance(ev, MatchOver):
                chipiron_logger.info("Match over for scope=%s", ev.scope)
                self.clear_pending()
                self.game_manager.game.publish_update(UpdNoHumanActionPending())
                self.game_manager.game.notify_display()
                continue

            elif isinstance(ev, IllegalAction):
                chipiron_logger.info(
                    "Illegal action rejected scope=%s role=%s request_id=%s reason=%s",
                    ev.scope,
                    ev.role,
                    ev.request_id,
                    ev.reason,
                )
                continue

            elif isinstance(ev, ActionApplied):
                self.game_manager.game.publish_update(UpdNoHumanActionPending())
                self.game_manager.game.notify_display()

    def handle_player_action(self, ev_move: EvMove) -> None:
        """Handle an engine/player move response correlated by request context."""
        if self.pending_request_id is None or self.pending_role is None:
            return
        pending_request_id: int = self.pending_request_id
        pending_role: GameRole = self.pending_role

        if ev_move.ctx is None:
            return
        if ev_move.ctx.request_id != pending_request_id:
            return
        if ev_move.ctx.role_to_play != pending_role:
            return

        try:
            action = self.game_manager.game.dynamics.action_from_name(
                self.game_manager.game.state,
                ev_move.branch_name,
            )
        except (KeyError, ValueError) as exc:
            chipiron_reason = f"invalid player action '{ev_move.branch_name}': {exc}"
            outs: list[MatchEvent] = [
                IllegalAction(
                    scope=self.scope,
                    role=ev_move.ctx.role_to_play,
                    request_id=ev_move.ctx.request_id,
                    action=ev_move.branch_name,
                    reason=chipiron_reason,
                ),
                NeedAction(
                    scope=self.scope,
                    role=self.game_manager.game.state.turn,
                    request_id=pending_request_id,
                    state=self.game_manager.game.state,
                ),
            ]
            self._handle_outputs(outs)
            return

        outs = self.game_manager.propose_action_sync(
            scope=self.scope,
            role=ev_move.ctx.role_to_play,
            request_id=ev_move.ctx.request_id,
            action=action,
        )
        self._handle_outputs(outs)

    def handle_human_action(self, human_action: HumanActionChosen) -> None:
        """Handle a human-selected action and dispatch resulting events."""
        if self.pending_role is None or self.pending_request_id is None:
            return
        pending_role: GameRole = self.pending_role
        pending_request_id: int = self.pending_request_id

        if human_action.ctx is not None:
            if human_action.ctx.request_id != pending_request_id:
                return
            if human_action.ctx.role_to_play != pending_role:
                return
        if (
            human_action.corresponding_state_tag is not None
            and human_action.corresponding_state_tag != self.game_manager.game.state.tag
        ):
            return

        state = self.game_manager.game.state
        try:
            legal_action_keys = list(
                self.game_manager.game.dynamics.legal_actions(state).get_all()
            )
            legal_name_to_action = {
                self.game_manager.game.dynamics.action_name(state, key): key
                for key in legal_action_keys
            }

            if human_action.action_name in legal_name_to_action:
                action = legal_name_to_action[human_action.action_name]
            else:
                parsed = self.game_manager.game.dynamics.action_from_name(
                    state, human_action.action_name
                )
                action = parsed
        except (KeyError, ValueError) as exc:
            outs: list[MatchEvent] = [
                IllegalAction(
                    scope=self.scope,
                    role=pending_role,
                    request_id=pending_request_id,
                    action=human_action.action_name,
                    reason=f"invalid human action '{human_action.action_name}': {exc}",
                ),
                NeedAction(
                    scope=self.scope,
                    role=state.turn,
                    request_id=pending_request_id,
                    state=state,
                ),
            ]
            self._handle_outputs(outs)
            return

        outs = self.game_manager.propose_action_sync(
            scope=self.scope,
            role=pending_role,
            request_id=pending_request_id,
            action=action,
        )
        self._handle_outputs(outs)
