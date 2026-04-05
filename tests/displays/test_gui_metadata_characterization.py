"""Characterization tests for current GUI metadata payload behavior."""

from __future__ import annotations

import queue

from anemone import TreeAndValuePlayerArgs
from anemone.node_selector.composed.args import ComposedNodeSelectorArgs
from anemone.node_selector.node_selector_types import NodeSelectorType
from anemone.node_selector.opening_instructions import OpeningType
from anemone.node_selector.priority_check.noop_args import NoPriorityCheckArgs
from anemone.node_selector.uniform.uniform import UniformArgs
from anemone.progress_monitor.progress_monitor import TreeBranchLimitArgs
from anemone.recommender_rule.recommender_rule import AlmostEqualLogistic
from valanga import Color

from chipiron.displays.gui_protocol import Scope, UpdMatchResults, UpdPlayerProgress
from chipiron.displays.gui_publisher import GuiPublisher
from chipiron.environments.chess.players.evaluators.boardevaluators.board_evaluator_type import (
    BoardEvalTypes,
)
from chipiron.games.domain.game.final_game_result import FinalGameResult
from chipiron.games.domain.game.progress_collector import PlayerProgressCollectorObservable
from chipiron.games.domain.match.match_results import MatchResults
from chipiron.games.domain.match.observable_match_result import ObservableMatchResults
from chipiron.players import PlayerArgs, PlayerFactoryArgs
from chipiron.players.boardevaluators.all_board_evaluator_args import (
    BasicEvaluationBoardEvaluatorArgs,
)
from chipiron.players.boardevaluators.master_board_evaluator_args import (
    MasterBoardEvaluatorArgs,
)
from chipiron.players.move_selector.human import GuiHumanPlayerArgs
from chipiron.players.move_selector.move_selector_types import MoveSelectorTypes
from chipiron.players.move_selector.tree_and_value_args import TreeAndValueAppArgs
from chipiron.utils.communication.player_ui_info import make_players_info_payload


def make_publisher() -> tuple[queue.Queue[object], GuiPublisher]:
    """Create a publisher and its backing queue for GUI payload assertions."""
    out: queue.Queue[object] = queue.Queue()
    scope = Scope(session_id="session-1", match_id="match-1", game_id="game-1")
    publisher = GuiPublisher(
        out=out,
        schema_version=1,
        game_kind="checkers",
        scope=scope,
    )
    return out, publisher


def make_tree_and_value_selector() -> TreeAndValueAppArgs:
    """Build a minimal real tree-and-value selector args object for label tests."""
    return TreeAndValueAppArgs(
        anemone_args=TreeAndValuePlayerArgs(
            node_selector=ComposedNodeSelectorArgs(
                type=NodeSelectorType.COMPOSED,
                priority=NoPriorityCheckArgs(type=NodeSelectorType.PRIORITY_NOOP),
                base=UniformArgs(type=NodeSelectorType.UNIFORM),
            ),
            opening_type=OpeningType.ALL_CHILDREN,
            stopping_criterion=TreeBranchLimitArgs(
                type="tree_branch_limit",
                tree_branch_limit=64,
            ),
            recommender_rule=AlmostEqualLogistic(
                type="almost_equal_logistic",
                temperature=1.0,
            ),
        ),
        evaluator_args=MasterBoardEvaluatorArgs(
            board_evaluator=BasicEvaluationBoardEvaluatorArgs(
                type=BoardEvalTypes.BASIC_EVALUATION_EVAL.value
            ),
            oracle_evaluation=False,
        ),
    )


def test_make_players_info_payload_keeps_white_black_mapping_and_current_labels() -> None:
    """Freeze the current white/black payload structure and label formatting."""
    white_player = PlayerFactoryArgs(
        player_args=PlayerArgs(
            name="GuiHuman",
            main_move_selector=GuiHumanPlayerArgs(type=MoveSelectorTypes.GUI_HUMAN),
            oracle_play=False,
        ),
        seed=1,
    )
    black_player = PlayerFactoryArgs(
        player_args=PlayerArgs(
            name="TreeBot",
            main_move_selector=make_tree_and_value_selector(),
            oracle_play=False,
        ),
        seed=1,
    )

    payload = make_players_info_payload(
        {
            Color.WHITE: white_player,
            Color.BLACK: black_player,
        }
    )

    assert payload.white.label == "GuiHuman ()"
    assert payload.white.is_human is True
    assert payload.black.label == "TreeBot (64)"
    assert payload.black.is_human is False


def test_player_progress_collector_observable_publishes_white_and_black_progress() -> None:
    """Freeze the current progress payload shape and per-color mapping."""
    out, publisher = make_publisher()
    progress = PlayerProgressCollectorObservable(publishers=[publisher])

    progress.progress_white(12)
    progress.progress_black(88)

    first_payload = out.get_nowait().payload
    second_payload = out.get_nowait().payload
    assert isinstance(first_payload, UpdPlayerProgress)
    assert isinstance(second_payload, UpdPlayerProgress)
    assert (first_payload.player_color, first_payload.progress_percent) == (
        Color.WHITE,
        12,
    )
    assert (second_payload.player_color, second_payload.progress_percent) == (
        Color.BLACK,
        88,
    )


def test_observable_match_results_fields_track_players_not_literal_colors() -> None:
    """Freeze the current `wins_white`/`wins_black` field semantics."""
    out, publisher = make_publisher()
    observable = ObservableMatchResults(
        match_results=MatchResults(
            player_one_name_id="player-one",
            player_two_name_id="player-two",
        ),
        publishers=[publisher],
    )

    observable.add_result_one_game(
        white_player_name_id="player-two",
        game_result=FinalGameResult.WIN_FOR_WHITE,
    )
    first_payload = out.get_nowait().payload
    assert isinstance(first_payload, UpdMatchResults)
    assert first_payload.wins_white == 0
    assert first_payload.wins_black == 1
    assert first_payload.draws == 0
    assert first_payload.games_played == 1
    assert first_payload.match_finished is False

    observable.finish()
    second_payload = out.get_nowait().payload
    assert isinstance(second_payload, UpdMatchResults)
    assert second_payload.wins_white == 0
    assert second_payload.wins_black == 1
    assert second_payload.games_played == 1
    assert second_payload.match_finished is True
