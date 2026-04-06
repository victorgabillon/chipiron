#! /usr/bin/env python

"""Document the module is the execution point of the chess GUI application.

It provides the `MainWindow` class, which creates a surface for the chessboard and handles user interactions.
"""

import os
import queue
import time
import typing
from dataclasses import dataclass

from atomheart.games.chess.board import BoardFactory
from PySide6 import QtGui
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QIcon
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import (
    QLabel,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)
from valanga import Color, StateTag
from valanga.evaluations import Value

from chipiron.core.roles import GameRole
from chipiron.displays.gui_protocol import (
    CmdBackOneMove,
    CmdSetStatus,
    GuiCommand,
    GuiUpdate,
    HumanActionChosen,
    ParticipantUiInfo,
    Scope,
    UpdEvaluation,
    UpdGameStatus,
    UpdMatchResults,
    UpdNeedHumanAction,
    UpdNoHumanActionPending,
    UpdParticipantProgress,
    UpdParticipantsInfo,
    UpdStateGeneric,
)
from chipiron.displays.svg_adapter_factory import make_svg_adapter
from chipiron.games.domain.game.game_playing_status import PlayingStatus
from chipiron.utils.communication.mailbox import MainMailboxMessage
from chipiron.utils.logger import chipiron_logger
from chipiron.utils.path_variables import GUI_DIR

if typing.TYPE_CHECKING:
    from chipiron.core.request_context import RequestContext
    from chipiron.displays.svg_adapter_protocol import SvgGameAdapter, SvgPosition
    from chipiron.environments.types import GameKind


class GuiUpdateError(AssertionError):
    """Raised when a GUI update payload is not handled."""

    def __init__(self, payload: object) -> None:
        """Initialize the error with the unhandled payload."""
        super().__init__(f"Unhandled GuiUpdate payload: {payload!r}")


@dataclass(slots=True)
class ParticipantRowWidgets:
    """Widgets used to display one participant row in the side panel."""

    button: QPushButton
    progress: QProgressBar
    role: GameRole | None = None
    is_human: bool = False


def format_state_eval(ev: Value | None) -> str:
    """Format state eval."""
    if ev is None:
        return "—"

    score = getattr(ev, "score", None)
    certainty = getattr(ev, "certainty", None)
    over_event = getattr(ev, "over_event", None)

    base = f"{float(score):+.2f}" if isinstance(score, int | float) else str(ev)

    certainty_name = None
    if certainty is not None:
        certainty_name = getattr(certainty, "name", str(certainty))

    over_str = None
    if over_event is not None:
        how_over = getattr(over_event, "how_over", None)
        winner = getattr(over_event, "who_is_winner", None)
        how_over_name = getattr(how_over, "name", str(how_over))
        winner_name = getattr(winner, "name", str(winner))
        over_str = f"{how_over_name}:{winner_name}"

    suffix_parts = [part for part in (certainty_name, over_str) if part]
    if suffix_parts:
        return f"{base} ({' | '.join(suffix_parts)})"
    return base


class MainWindow(QWidget):
    """Create a surface for the chessboard and handle user interactions.

    This class provides the main window for the chess GUI application. It handles user interactions such as
    clicking on chess pieces, making moves, and controlling the game status.

    Attributes:
        playing_status (PlayingStatus): The current playing status of the game.
        gui_mailbox (queue.Queue[IsDataclass]): The mailbox for receiving messages from the GUI thread.
        main_thread_mailbox (queue.Queue[IsDataclass]): The mailbox for sending messages to the main thread.

    """

    def __init__(
        self,
        gui_mailbox: queue.Queue[GuiUpdate],
        main_thread_mailbox: queue.Queue[MainMailboxMessage],
        board_factory: BoardFactory,
    ) -> None:
        """Initialize the chessboard and the main window.

        Args:
            gui_mailbox (queue.Queue[IsDataclass]): The mailbox for receiving messages from the GUI thread.
            main_thread_mailbox (queue.Queue[IsDataclass]): The mailbox for sending messages to the main thread.

        """
        super().__init__()

        self.current_state_tag: StateTag | None = None
        self.pending_human_ctx: RequestContext | None = None
        self.pending_human_state_tag: StateTag | None = None

        self.play_button_clicked_last_time: float | None = None
        self.pause_button_clicked_last_time: float | None = None

        self.board_factory = board_factory
        self.playing_status = PlayingStatus.PLAY

        self.gui_mailbox = gui_mailbox
        self.main_thread_mailbox = main_thread_mailbox

        self.scope: Scope | None = None
        self.adapter: SvgGameAdapter | None = None
        self.current_pos: SvgPosition | None = None
        self.adapter_kind: GameKind | None = None
        self.action_name_history: list[str] = []
        self.participant_rows: list[ParticipantRowWidgets] = []
        self.participant_rows_by_role: dict[GameRole, ParticipantRowWidgets] = {}
        # Set window icon with existence check
        window_icon_path = os.path.join(GUI_DIR, "chipicon.png")
        if os.path.exists(window_icon_path):
            self.setWindowIcon(QIcon(window_icon_path))
        else:
            chipiron_logger.warning("Window icon file not found: %s", window_icon_path)

        self.setWindowTitle("🐙 Chipiron GUI 🐙")
        self.setGeometry(300, 300, 1400, 800)

        self.widget_svg = QSvgWidget(parent=self)
        self.widget_svg.setGeometry(10, 10, 600, 600)

        self.close_button = QPushButton(self)
        self.close_button.setText("Close")  # text
        self._check_and_set_icon(
            self.close_button, os.path.join(GUI_DIR, "close.png")
        )  # icon
        self.close_button.setShortcut("Ctrl+D")  # shortcut key
        self.close_button.clicked.connect(self.stopppy)  # pylint: disable=no-member
        self.close_button.setToolTip("Close the widget")  # Tool tip
        self.close_button.move(800, 20)

        self.pause_button = QPushButton(self)
        self.pause_button.setText("Pause")  # text
        self._check_and_set_icon(
            self.pause_button, os.path.join(GUI_DIR, "pause.png")
        )  # icon
        self.pause_button.clicked.connect(self.pause_button_clicked)  # pylint: disable=no-member
        self.pause_button.setToolTip("pause the game")  # Tool tip
        self.pause_button.move(700, 100)

        self.back_button = QPushButton(self)
        self.back_button.setText("Back")  # text
        self._check_and_set_icon(
            self.back_button, os.path.join(GUI_DIR, "back.png")
        )  # icon
        self.back_button.clicked.connect(self.back_button_clicked)  # pylint: disable=no-member
        self.back_button.setToolTip("back one move")  # Tool tip
        self.back_button.move(900, 100)

        self.tablewidget = QTableWidget(1, 2, self)
        self.tablewidget.setGeometry(1100, 200, 260, 330)

        self.score_button = QPushButton(self)
        self.score_button.setText("⚖ Score 0-0")  # text
        self.score_button.setStyleSheet(
            "QPushButton {background-color: white; color: black;}"
        )
        self.score_button.setGeometry(620, 400, 370, 30)

        self.round_button = QPushButton(self)
        self.round_button.setText("🎲 Round")  # text
        self.round_button.setStyleSheet(
            "QPushButton {background-color: white; color: black;}"
        )
        self.round_button.setGeometry(620, 500, 370, 30)

        self.fen_button = QLabel(self)
        self.fen_button.setText("fen")  # text
        self.fen_button.setStyleSheet(
            "QPushButton { background-color: white; color: black;}"
        )
        self.fen_button.setGeometry(20, 620, 550, 30)
        self.fen_button.setTextInteractionFlags(
            QtGui.Qt.TextInteractionFlag.TextSelectableByMouse
        )

        self.legal_moves_button = QLabel(self)
        self.legal_moves_button.setText("legal moves")  # text
        self.legal_moves_button.setStyleSheet(
            "QPushButton { background-color: white; color: black;}"
        )
        self.legal_moves_button.setGeometry(20, 650, 550, 80)
        self.legal_moves_button.setTextInteractionFlags(
            QtGui.Qt.TextInteractionFlag.TextSelectableByMouse
        )

        self.eval_button = QPushButton(self)
        self.eval_button.setText("🐟 Eval")  # text
        self.eval_button.setStyleSheet(
            "QPushButton {background-color: white; color: black;}"
        )
        self.eval_button.setGeometry(620, 600, 470, 30)

        self.eval_button_chi = QPushButton(self)
        self.eval_button_chi.setText("🐙 Eval")  # text
        self.eval_button_chi.setStyleSheet(
            "QPushButton {background-color: white; color: black;}"
        )
        self.eval_button_chi.setGeometry(620, 650, 470, 30)

        self.eval_button_white = QPushButton(self)
        self.eval_button_white.setText("♕ White Eval")  # text
        self.eval_button_white.setStyleSheet(
            "QPushButton {background-color: white; color: black;}"
        )
        self.eval_button_white.setGeometry(620, 700, 470, 30)

        self.eval_button_black = QPushButton(self)
        self.eval_button_black.setText("♛ Black Eval")  # text
        self.eval_button_black.setStyleSheet(
            "QPushButton {background-color: white; color: black;}"
        )
        self.eval_button_black.setGeometry(620, 750, 470, 30)

        self.board_size = min(self.widget_svg.width(), self.widget_svg.height())
        self.coordinates = True
        self.margin = 0.05 * self.board_size if self.coordinates else 0
        self.square_size = (self.board_size - 2 * self.margin) / 8.0

        self.check_thread_timer = QTimer(self)
        self.check_thread_timer.setInterval(5)  # .5 seconds
        self.check_thread_timer.timeout.connect(self.process_message)  # pylint: disable=no-member
        self.check_thread_timer.start()

    def _should_accept_scope(self, incoming: Scope) -> bool:
        """Routing policy.

        For now (single GUI view):
        - accept first scope
        - accept scope changes within the same session (and same match_id when both are set)
        - ignore updates from other sessions/matches to avoid mixing across runs
        """
        if self.scope is None:
            return True

        if incoming.session_id != self.scope.session_id:
            return False

        return not (
            self.scope.match_id is not None
            and incoming.match_id is not None
            and incoming.match_id != self.scope.match_id
        )

    def _check_and_set_icon(self, button: QPushButton, icon_path: str) -> None:
        """Check if icon file exists and set it, otherwise log a warning.

        Args:
            button: The button to set the icon for
            icon_path: The path to the icon file

        """
        if os.path.exists(icon_path):
            button.setIcon(QIcon(icon_path))
        else:
            chipiron_logger.warning("Icon file not found: %s", icon_path)
            # Set a default icon or leave empty
            button.setIcon(QIcon())  # Empty icon

    def _participant_row_top(self, index: int) -> int:
        """Return the top y-position for a participant row."""
        return 200 + index * 70

    def _ensure_participant_rows(self, count: int) -> None:
        """Create enough participant widgets for the current metadata payload."""
        while len(self.participant_rows) < count:
            index = len(self.participant_rows)
            top = self._participant_row_top(index)
            button = QPushButton(self)
            button.setText("Participant")
            button.setGeometry(620, top, 470, 30)
            progress = QProgressBar(self)
            progress.setGeometry(620, top + 30, 470, 20)
            progress.setTextVisible(False)
            self.participant_rows.append(
                ParticipantRowWidgets(button=button, progress=progress)
            )

        for index, row in enumerate(self.participant_rows):
            visible = index < count
            row.button.setVisible(visible)
            row.progress.setVisible(visible)

    def _apply_participant_style(self, button: QPushButton, role: GameRole) -> None:
        """Apply lightweight styling for one participant role."""
        if role == Color.WHITE:
            self._check_and_set_icon(button, os.path.join(GUI_DIR, "white_king.png"))
            button.setStyleSheet("QPushButton {background-color: white; color: black;}")
        elif role == Color.BLACK:
            self._check_and_set_icon(button, os.path.join(GUI_DIR, "black_king.png"))
            button.setStyleSheet("QPushButton {background-color: black; color: white;}")
        else:
            button.setIcon(QIcon())
            button.setStyleSheet(
                "QPushButton {background-color: #eef3f8; color: black;}"
            )

    def _refresh_participant_progress_widgets(self) -> None:
        """Refresh participant progress bars after metadata or pending-role changes."""
        pending_role = (
            self.pending_human_ctx.role_to_play
            if self.pending_human_ctx is not None
            else None
        )
        for row in self.participant_rows:
            if row.role is None or not row.button.isVisible():
                continue
            if row.is_human:
                row.progress.setTextVisible(True)
                if row.role == pending_role:
                    row.progress.setValue(100)
                    row.progress.setFormat("Think Human!")
                else:
                    row.progress.setValue(0)
                    row.progress.setFormat("Human")

    def stopppy(self) -> None:
        """Stop the execution of the GUI.

        This method closes the GUI window and sends a kill message to the main thread.

        Returns:
            None

        """
        # should we send a kill message to the main thread?
        self.close()

    def play_button_clicked(self) -> None:
        """Handle the event when the play button is clicked.

        This method prints a message indicating that the play button has been clicked,
        and sends a GameStatusMessage with the status set to PlayingStatus.PLAY to the main thread mailbox.
        """
        if self.playing_status != PlayingStatus.PAUSE:
            return
        if self.scope is None:
            return
        if (
            self.play_button_clicked_last_time is None
            or abs(self.play_button_clicked_last_time - time.time()) > 0.01
        ):
            chipiron_logger.info("play_button_clicked")
            cmd = GuiCommand(
                schema_version=1,
                scope=self.scope,
                payload=CmdSetStatus(status=PlayingStatus.PLAY),
            )
            self.main_thread_mailbox.put(cmd)
            self.play_button_clicked_last_time = time.time()

    def back_button_clicked(self) -> None:
        """Handle the event when the back button is clicked.

        This method prints a message and puts a `BackMessage` object into the main thread mailbox.
        """
        chipiron_logger.info("back_button_clicked")
        if self.scope is None:
            return
        cmd = GuiCommand(
            schema_version=1,
            scope=self.scope,
            payload=CmdBackOneMove(),
        )
        self.main_thread_mailbox.put(cmd)

    def pause_button_clicked(self) -> None:
        """Handle the click event of the pause button.

        Prints 'pause_button_clicked' and sends a GameStatusMessage with the status set to PlayingStatus.PAUSE
        to the main thread mailbox.
        """
        if self.playing_status != PlayingStatus.PLAY:
            return
        if self.scope is None:
            return
        if (
            self.pause_button_clicked_last_time is None
            or abs(self.pause_button_clicked_last_time - time.time()) > 0.01
        ):
            chipiron_logger.info("pause_button_clicked")
            cmd = GuiCommand(
                schema_version=1,
                scope=self.scope,
                payload=CmdSetStatus(status=PlayingStatus.PAUSE),
            )
            self.main_thread_mailbox.put(cmd)
            self.pause_button_clicked_last_time = time.time()

    @typing.no_type_check
    @typing.override
    @Slot(QWidget)
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse press events on the board through game adapter."""
        if self.adapter is None or self.current_pos is None:
            return

        local_x = event.x() - self.widget_svg.x()
        local_y = event.y() - self.widget_svg.y()

        if not (
            0 <= local_x <= self.board_size
            and 0 <= local_y <= self.board_size
            and event.buttons() == Qt.LeftButton
            and self.margin < local_x < self.board_size - self.margin
            and self.margin < local_y < self.board_size - self.margin
        ):
            return

        res = self.adapter.handle_click(
            self.current_pos,
            x=int(local_x),
            y=int(local_y),
            board_size=int(self.board_size),
            margin=int(self.margin),
        )

        if res.action_name is not None:
            self.send_action_to_main_thread(action_name=res.action_name)

        if not res.interaction_continues:
            self.adapter.reset_interaction()

    def send_action_to_main_thread(self, *, action_name: str) -> None:
        """Send a human action name to the main thread for processing."""
        if self.scope is None:
            return
        cmd = GuiCommand(
            schema_version=1,
            scope=self.scope,
            payload=HumanActionChosen(
                action_name=action_name,
                ctx=self.pending_human_ctx,
                corresponding_state_tag=self.pending_human_state_tag,
            ),
        )
        self.main_thread_mailbox.put(cmd)

    def reset_for_new_game(self) -> None:
        """Reset for new game."""
        self.tablewidget.clearContents()
        self.tablewidget.setRowCount(0)
        self.participant_rows_by_role.clear()
        for row in self.participant_rows:
            row.role = None
            row.is_human = False
            row.button.setVisible(False)
            row.progress.setVisible(False)
            row.progress.reset()
            row.progress.setValue(0)
            row.progress.setTextVisible(False)
        self.eval_button.setText("🐟 Eval")
        self.eval_button_chi.setText("🐙 Eval")
        self.eval_button_white.setText("♕ White Eval")
        self.eval_button_black.setText("♛ Black Eval")
        if self.adapter is not None:
            self.adapter.reset_interaction()
        self.adapter = None
        self.current_pos = None
        self.adapter_kind = None
        self.action_name_history = []
        self.pending_human_ctx = None
        self.pending_human_state_tag = None

    def process_message(self) -> None:
        """Process a message received by the GUI.

        Draw a chessboard with the starting position and then redraw
        it for every new move.

        This method is responsible for handling different types of messages
        received by the GUI and taking appropriate actions based on the message type.

        Supported message types:
        - BoardMessage: Updates the board and redraws it.
        - EvaluationMessage: Updates the evaluation values.
        - PlayersColorToPlayerMessage: Updates the mapping of player colors to player information.
        - MatchResultsMessage: Updates the match results.
        - GameStatusMessage: Updates the game play status.
        - Other: Raises a ValueError indicating an unknown message type.

        Returns:
        None

        """
        if self.gui_mailbox.empty():
            return

        msg: GuiUpdate = self.gui_mailbox.get()

        if not self._should_accept_scope(msg.scope):
            chipiron_logger.debug(
                "Ignoring GuiUpdate from other scope: %s (current=%s)",
                msg.scope,
                self.scope,
            )
            return

        if self.scope is None:
            self.scope = msg.scope
        elif msg.scope != self.scope:
            # sequential-games policy: accept new scope and reset UI
            self.reset_for_new_game()
            self.scope = msg.scope

        payload = msg.payload

        match payload:
            case UpdStateGeneric():
                self.current_state_tag = payload.state_tag
                self.action_name_history = list(payload.action_name_history)

                if self.adapter is None or self.adapter_kind != msg.game_kind:
                    self.adapter = make_svg_adapter(
                        game_kind=msg.game_kind,
                        board_factory=self.board_factory,
                    )
                    self.adapter.reset_interaction()
                    self.adapter_kind = msg.game_kind

                self.current_pos = self.adapter.position_from_update(
                    state_tag=payload.state_tag,
                    adapter_payload=payload.adapter_payload,
                )

                render = self.adapter.render_svg(
                    self.current_pos,
                    size=int(self.board_size),
                    margin=int(self.margin),
                )
                self.widget_svg.load(render.svg_bytes)
                self._apply_render_info(render.info)

                self.display_action_name_history()

            case UpdParticipantProgress():
                self.update_participant_progress(payload.role, payload.progress_percent)

            case UpdEvaluation():
                self.update_evaluation(
                    evaluation_oracle=payload.oracle,
                    evaluation_chipiron=payload.chipiron,
                    evaluation_white=payload.white,
                    evaluation_black=payload.black,
                )

            case UpdParticipantsInfo():
                self.update_participants_info(payload.participants)

            case UpdMatchResults():
                self.update_match_stats(payload)

            case UpdGameStatus():
                self.update_game_play_status(payload.status)

            case UpdNeedHumanAction():
                self.pending_human_ctx = payload.ctx
                self.pending_human_state_tag = payload.state_tag
                self._refresh_participant_progress_widgets()

            case UpdNoHumanActionPending():
                self.pending_human_ctx = None
                self.pending_human_state_tag = None
                self._refresh_participant_progress_widgets()

            case _:
                raise GuiUpdateError(payload)

    def display_action_name_history(self) -> None:
        """Display action history as a chronological generic table."""
        self.tablewidget.setRowCount(len(self.action_name_history))
        self.tablewidget.setHorizontalHeaderLabels(["Ply", "Action"])
        for half_move, action_name in enumerate(self.action_name_history, start=1):
            self.tablewidget.setItem(half_move - 1, 0, QTableWidgetItem(str(half_move)))
            self.tablewidget.setItem(half_move - 1, 1, QTableWidgetItem(str(action_name)))

    def _apply_render_info(self, info: dict[str, str]) -> None:
        """Update generic side panel labels using adapter render metadata."""
        self.round_button.setText("🎲 Round: " + info.get("round", "-"))
        self.fen_button.setText("🔧 <b>fen:</b> " + info.get("fen", "-"))
        self.legal_moves_button.setWordWrap(True)
        self.legal_moves_button.setMinimumHeight(100)
        self.legal_moves_button.setText(
            f"📋 <b>legal moves:</b><pre>{info.get('legal_moves', '')}</pre>"
        )

    def update_participants_info(
        self, participants: typing.Sequence[ParticipantUiInfo]
    ) -> None:
        """Update the participant side panel from a role-driven payload."""
        self._ensure_participant_rows(len(participants))
        self.participant_rows_by_role.clear()

        for index, participant in enumerate(participants):
            row = self.participant_rows[index]
            row.role = participant.role
            row.is_human = participant.is_human
            self.participant_rows_by_role[participant.role] = row
            self._apply_participant_style(row.button, participant.role)
            row.button.setText(f" {participant.role_label}: {participant.label}")
            row.progress.reset()
            row.progress.setValue(0)
            row.progress.setTextVisible(False)

        for row in self.participant_rows[len(participants) :]:
            row.role = None
            row.is_human = False
            row.button.setVisible(False)
            row.progress.setVisible(False)
            row.progress.reset()
            row.progress.setValue(0)
            row.progress.setTextVisible(False)

        self._refresh_participant_progress_widgets()

    def update_participant_progress(
        self, role: GameRole, progress_percent: int | None
    ) -> None:
        """Update the progress bar for one participant role."""
        row = self.participant_rows_by_role.get(role)
        if row is None or row.is_human:
            return
        row.progress.setTextVisible(True)
        if progress_percent is None:
            row.progress.reset()
            row.progress.setValue(0)
        else:
            row.progress.setValue(progress_percent)
            row.progress.setFormat("%p%")

    def update_evaluation(
        self,
        evaluation_oracle: Value | None,
        evaluation_chipiron: Value | None,
        evaluation_white: Value | None,
        evaluation_black: Value | None,
    ) -> None:
        """Update the evaluation values displayed on the GUI.

        Args:
            evaluation_oracle (Value | None): The evaluation value for the oracle.
            evaluation_chipiron (Value | None): The evaluation value for chipiron.
            evaluation_white (Value | None): The evaluation value for white.
            evaluation_black (Value | None): The evaluation value for black.

        Returns:
            None

        """
        self.eval_button.setText("📊 Eval 🐟: " + format_state_eval(evaluation_oracle))
        self.eval_button_chi.setText(
            "🧮 Eval 🐙: " + format_state_eval(evaluation_chipiron)
        )
        self.eval_button_white.setText(
            "🧠 Eval White: " + format_state_eval(evaluation_white)
        )
        self.eval_button_black.setText(
            "🧠 Eval Black: " + format_state_eval(evaluation_black)
        )

    def update_game_play_status(self, play_status: PlayingStatus) -> None:
        """Update the game play status.

        Args:
            play_status (PlayingStatus): The new playing status.

        """
        chipiron_logger.info("update_game_play_status %s", play_status)

        if self.playing_status != play_status:
            self.playing_status = play_status
            if play_status == PlayingStatus.PAUSE:
                self.pause_button.setText("Play")  # text
                self._check_and_set_icon(
                    self.pause_button, os.path.join(GUI_DIR, "play.png")
                )  # icon
                self.pause_button.clicked.connect(self.play_button_clicked)  # pylint: disable=no-member
                self.pause_button.setToolTip("play the game")  # Tool tip
                self.pause_button.move(700, 100)
            elif play_status == PlayingStatus.PLAY:
                self.pause_button.setText("Pause")  # text
                self._check_and_set_icon(
                    self.pause_button, os.path.join(GUI_DIR, "pause.png")
                )  # icon
                self.pause_button.clicked.connect(self.pause_button_clicked)  # pylint: disable=no-member
                self.pause_button.setToolTip("pause the game")  # Tool tip
                self.pause_button.move(700, 100)

    def update_match_stats(self, upd: UpdMatchResults) -> None:
        """Update match stats."""
        self.score_button.setText(
            f"⚖ Score: {upd.wins_white}-{upd.wins_black}-{upd.draws}"
        )

        chipiron_logger.info("update match_finished=%s", upd.match_finished)
        if upd.match_finished:
            chipiron_logger.info("finishing the widget")
            self.close()
