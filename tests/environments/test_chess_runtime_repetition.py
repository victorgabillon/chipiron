"""Regression tests for chess runtime history and repetition handling."""

from __future__ import annotations

import queue
from types import SimpleNamespace

import chess
import pytest
from atomheart import ChessDynamics
from atomheart.games.chess.board import create_board_factory
from atomheart.games.chess.board.utils import FenPlusHistory
from atomheart.games.chess.state import ChessState
from valanga import Color

from chipiron.displays.gui_protocol import Scope, UpdStateGeneric
from chipiron.displays.gui_publisher import GuiPublisher
from chipiron.environments.chess.chess_gui_encoder import ChessGuiEncoder
from chipiron.environments.chess.chess_rules import ChessRules
from chipiron.games.domain.game.game import Game, ObservableGame
from chipiron.games.domain.game.game_manager import GameManager
from chipiron.games.domain.game.game_playing_status import GamePlayingStatus
from chipiron.games.domain.game.game_rules import OutcomeKind
from chipiron.games.runtime.orchestrator.domain_events import NeedAction
from chipiron.players.communications.player_request_encoder import (
    ChessPlayerRequestEncoder,
)

REPETITION_SEQUENCE: tuple[str, ...] = (
    "g1f3",
    "g8f6",
    "f3g1",
    "f6g8",
    "g1f3",
    "g8f6",
    "f3g1",
    "f6g8",
)
REPETITION_PREFIX: tuple[str, ...] = REPETITION_SEQUENCE[:4]
TEST_SCOPE = Scope(
    session_id="repetition-session",
    match_id="repetition-match",
    game_id="repetition-game",
)


class NoOpEvaluator:
    """Minimal evaluator surface required by GameManager."""

    def evaluate(self, state: ChessState) -> tuple[None, float]:
        """Return a neutral evaluation."""
        _ = state
        return None, 0.0

    def add_evaluation(self, player_color: Color, evaluation: float) -> None:
        """Ignore externally reported evaluations."""
        _ = player_color
        _ = evaluation


class NoOpProgressCollector:
    """Minimal progress collector surface required by GameManager."""

    def progress(self, role: object, value: int | None) -> None:
        """Ignore generic progress updates."""
        _ = role
        _ = value

    def progress_white(self, value: int | None) -> None:
        """Ignore legacy white progress updates."""
        _ = value

    def progress_black(self, value: int | None) -> None:
        """Ignore legacy black progress updates."""
        _ = value


def _build_game_manager(
    *, use_rust_boards: bool
) -> tuple[GameManager[ChessState], queue.Queue[object]]:
    board_factory = create_board_factory(use_rust_boards=use_rust_boards)
    board = board_factory(
        fen_with_history=FenPlusHistory(current_fen=chess.STARTING_FEN)
    )
    state = ChessState(board)
    game = Game(
        state=state,
        dynamics=ChessDynamics(),
        playing_status=GamePlayingStatus(),
        seed_=0,
    )
    observable_game = ObservableGame(
        game=game,
        gui_encoder=ChessGuiEncoder(),
        player_encoder=ChessPlayerRequestEncoder(),
        scope=TEST_SCOPE,
    )
    display_queue: queue.Queue[object] = queue.Queue()
    observable_game.register_display(
        GuiPublisher(
            out=display_queue,
            schema_version=1,
            game_kind="chess",
            scope=TEST_SCOPE,
        )
    )
    return (
        GameManager(
            game=observable_game,
            display_state_evaluator=NoOpEvaluator(),
            output_folder_path=None,
            args=SimpleNamespace(max_half_moves=None),
            participant_id_by_role={Color.WHITE: "white", Color.BLACK: "black"},
            main_thread_mailbox=queue.Queue(),
            players=[],
            move_factory=object(),
            progress_collector=NoOpProgressCollector(),
            rules=ChessRules(syzygy=None),
        ),
        display_queue,
    )


def _play_runtime_repetition_sequence(
    *, use_rust_boards: bool
) -> tuple[GameManager[ChessState], list[tuple[str, ...]]]:
    game_manager, display_queue = _build_game_manager(use_rust_boards=use_rust_boards)
    start_event = game_manager.start_match_sync(TEST_SCOPE)[0]
    assert isinstance(start_event, NeedAction)
    request_id = start_event.request_id

    gui_histories: list[tuple[str, ...]] = []
    for move_uci in REPETITION_SEQUENCE:
        state = game_manager.game.state
        action = game_manager.game.dynamics.action_from_name(state, move_uci)
        events = game_manager.propose_action_sync(
            TEST_SCOPE,
            state.turn,
            request_id,
            action,
        )
        game_manager.game.notify_display()
        gui_update = display_queue.get_nowait()
        payload = gui_update.payload
        assert isinstance(payload, UpdStateGeneric)
        gui_histories.append(tuple(payload.action_name_history))

        next_need_actions = [event for event in events if isinstance(event, NeedAction)]
        if next_need_actions:
            request_id = next_need_actions[-1].request_id

    return game_manager, gui_histories


@pytest.mark.parametrize(("use_rust_boards"), (False, True))
def test_runtime_repetition_sequence_keeps_full_gui_history_and_draws(
    use_rust_boards: bool,
) -> None:
    """The real runtime path should retain history and terminate repeated games."""
    game_manager, gui_histories = _play_runtime_repetition_sequence(
        use_rust_boards=use_rust_boards
    )

    assert [len(history) for history in gui_histories] == list(
        range(1, len(REPETITION_SEQUENCE) + 1)
    )
    assert gui_histories[-1] == REPETITION_SEQUENCE
    assert tuple(str(move) for move in game_manager.game.action_history) == REPETITION_SEQUENCE

    final_state = game_manager.game.state
    final_outcome = game_manager.rules.outcome(final_state)

    assert final_state.board.is_game_over()
    assert final_state.board.result(claim_draw=True) == "1/2-1/2"
    assert final_outcome is not None
    assert final_outcome.kind is OutcomeKind.DRAW


def test_rust_board_copy_preserves_current_repetition_count() -> None:
    """Copying a Rust board must preserve the current repetition count."""
    board_factory = create_board_factory(use_rust_boards=True)
    board = board_factory(
        fen_with_history=FenPlusHistory(current_fen=chess.STARTING_FEN)
    )

    for move_uci in REPETITION_PREFIX:
        board.play_move_uci(move_uci)

    source_key = board.fast_representation_without_counters
    copied_board = board.copy(stack=True)
    copied_key = copied_board.fast_representation_without_counters

    assert copied_key == source_key
    assert board.move_stack == copied_board.move_stack
    assert board.rep_to_count[source_key] == 2
    assert copied_board.rep_to_count[copied_key] == 2
