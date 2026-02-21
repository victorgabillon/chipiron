"""MatchOrchestrator: owns the mailbox loop and transport dispatch.

Medium-refactor goal:
- GameManager: synchronous deterministic domain transitions only
- MatchController: move pipeline + request correlation + routing
- MatchOrchestrator: mailbox.get() loop + dispatch + match lifecycle
"""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_never

from valanga import Color

from chipiron.displays.gui_protocol import (
    CmdBackOneMove,
    CmdSetStatus,
    GuiCommand,
    HumanActionChosen,
    UpdNoHumanActionPending,
)
from chipiron.games.game.final_game_result import GameReport
from chipiron.games.game.game_playing_status import PlayingStatus
from chipiron.players.communications.player_message import (
    EvMove,
    EvProgress,
    PlayerEvent,
)
from chipiron.utils.logger import chipiron_logger

if TYPE_CHECKING:
    import queue

    from chipiron.games.game.game_manager import GameManager
    from chipiron.match.match_controller import MatchController
    from chipiron.utils.communication.mailbox import MainMailboxMessage


class MatchOrchestrator:
    """Runs one game by consuming mailbox messages and dispatching them."""

    def __init__(
        self,
        mailbox: queue.Queue[MainMailboxMessage],
    ) -> None:
        """Store the mailbox used by the outer orchestrator loop."""
        self._mailbox = mailbox

    def play_one_game(
        self,
        *,
        game_manager: GameManager,
        controller: MatchController,
    ) -> GameReport:
        """Run the match loop until terminal condition, then return GameReport."""
        color_names = {Color.WHITE: "White", Color.BLACK: "Black"}

        game_manager.game.notify_display()

        if game_manager.game.is_play():
            controller.start()

        while True:
            state = game_manager.game.state
            ply = game_manager.game.ply
            chipiron_logger.info(
                "Half Move: %s playing status %s",
                ply,
                game_manager.game.playing_status.status,
            )
            color_to_move = state.turn
            chipiron_logger.info(
                "%s (%s) to play now...",
                color_names[color_to_move],
                game_manager.player_color_to_id[color_to_move],
            )

            mail: MainMailboxMessage = self._mailbox.get()
            if isinstance(mail, GuiCommand):
                if self._ignore_if_stale_scope(game_manager, mail.scope):
                    continue
                match mail.payload:
                    case HumanActionChosen():
                        controller.handle_human_action(mail.payload)
                    case _:
                        self._handle_gui_command(game_manager, controller, mail)

            elif isinstance(mail, PlayerEvent):
                if self._ignore_if_stale_scope(game_manager, mail.scope):
                    continue
                match mail.payload:
                    case EvMove():
                        controller.handle_player_action(mail.payload)
                    case _:
                        self._handle_player_event(game_manager, mail)

            else:
                assert_never(mail)

            state = game_manager.game.state
            is_terminal = game_manager.rules.outcome(state) is not None
            if is_terminal or not game_manager.game_continue_conditions():
                if is_terminal:
                    chipiron_logger.info("The game is over")
                if not game_manager.game_continue_conditions():
                    chipiron_logger.info("Game continuation not met")
                break
            chipiron_logger.info("Not game over at %s", state)

        game_manager.tell_results()
        game_manager.terminate_processes()
        chipiron_logger.info("End play_one_game")

        game_results = game_manager.simple_results()
        return GameReport(
            final_game_result=game_results,
            action_history=list(game_manager.game.action_history),
            state_tag_history=game_manager.game.state_tag_history,
        )

    def _ignore_if_stale_scope(self, game_manager: GameManager, scope: object) -> bool:
        if scope != game_manager.game.scope:
            chipiron_logger.debug(
                "Ignoring stale message scope=%s (current=%s)",
                scope,
                game_manager.game.scope,
            )
            return True
        return False

    def _handle_gui_command(
        self,
        game_manager: GameManager,
        controller: MatchController,
        message: GuiCommand,
    ) -> None:
        match message.payload:
            case CmdSetStatus():
                if message.payload.status == PlayingStatus.PLAY:
                    game_manager.game.set_play_status()
                    controller.request_next_action()

                elif message.payload.status == PlayingStatus.PAUSE:
                    game_manager.game.set_pause_status()
                    game_manager.invalidate_pending_request()
                    controller.clear_pending()
                    game_manager.game.publish_update(UpdNoHumanActionPending())

                else:
                    chipiron_logger.warning(
                        "Unhandled PlayingStatus: %s", message.payload.status
                    )

            case CmdBackOneMove():
                game_manager.game.set_pause_status()
                game_manager.rewind_one_move()
                game_manager.invalidate_pending_request()
                controller.clear_pending()
                game_manager.game.publish_update(UpdNoHumanActionPending())

            case _:
                chipiron_logger.warning(
                    "Unhandled GuiCommand payload in %s: %r",
                    __name__,
                    message.payload,
                )

    def _handle_player_event(
        self,
        game_manager: GameManager,
        message: PlayerEvent,
    ) -> None:
        match message.payload:
            case EvMove():
                return

            case EvProgress():
                if message.payload.player_color == Color.WHITE:
                    game_manager.progress_collector.progress_white(
                        value=message.payload.progress_percent
                    )
                else:
                    game_manager.progress_collector.progress_black(
                        value=message.payload.progress_percent
                    )

            case _:
                chipiron_logger.warning(
                    "Unhandled PlayerEvent payload in %s: %r",
                    __name__,
                    message.payload,
                )
