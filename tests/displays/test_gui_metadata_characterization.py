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

from chipiron.displays.action_history_table import build_action_history_table
from chipiron.displays.gui_protocol import (
    Scope,
    UpdMatchResults,
    UpdParticipantProgress,
    UpdParticipantsInfo,
)
from chipiron.displays.gui_publisher import GuiPublisher
from chipiron.environments.chess.players.evaluators.boardevaluators.board_evaluator_type import (
    BoardEvalTypes,
)
from chipiron.games.domain.game.final_game_result import (
    FinalGameResult,
    GameReport,
    RoleOutcome,
)
from chipiron.games.domain.game.progress_collector import (
    PlayerProgressCollectorObservable,
)
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
from chipiron.utils.communication.player_ui_info import make_participants_info_payload


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


def test_make_participants_info_payload_keeps_role_order_and_current_labels() -> None:
    """Participant metadata should preserve role order and current label formatting."""
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

    payload = make_participants_info_payload(
        participant_factory_args_by_role={
            Color.WHITE: white_player,
            Color.BLACK: black_player,
        },
        role_order=(Color.WHITE, Color.BLACK),
    )

    assert isinstance(payload, UpdParticipantsInfo)
    assert [participant.role for participant in payload.participants] == [
        Color.WHITE,
        Color.BLACK,
    ]
    assert payload.participants[0].role_label == "White"
    assert payload.participants[0].label == "GuiHuman ()"
    assert payload.participants[0].is_human is True
    assert payload.participants[1].role_label == "Black"
    assert payload.participants[1].label == "TreeBot (64)"
    assert payload.participants[1].is_human is False


def test_make_participants_info_payload_supports_single_and_three_role_cases() -> None:
    """The generic payload must represent non-white/black role sets cleanly."""
    human_player = PlayerFactoryArgs(
        player_args=PlayerArgs(
            name="SoloHuman",
            main_move_selector=GuiHumanPlayerArgs(type=MoveSelectorTypes.GUI_HUMAN),
            oracle_play=False,
        ),
        seed=7,
    )
    engine_player = PlayerFactoryArgs(
        player_args=PlayerArgs(
            name="TriBot",
            main_move_selector=make_tree_and_value_selector(),
            oracle_play=False,
        ),
        seed=8,
    )

    solo_payload = make_participants_info_payload(
        participant_factory_args_by_role={"solo": human_player},
        role_order=("solo",),
    )
    tri_payload = make_participants_info_payload(
        participant_factory_args_by_role={
            "alpha": engine_player,
            "beta": human_player,
            "gamma": engine_player,
        },
        role_order=("alpha", "beta", "gamma"),
    )

    assert [participant.role for participant in solo_payload.participants] == ["solo"]
    assert solo_payload.participants[0].label == "SoloHuman ()"
    assert [participant.role_label for participant in tri_payload.participants] == [
        "alpha",
        "beta",
        "gamma",
    ]
    assert [participant.is_human for participant in tri_payload.participants] == [
        False,
        True,
        False,
    ]


def test_build_action_history_table_groups_two_roles_side_by_side() -> None:
    """Two-role history should be displayed by round with one column per role."""
    headers, rows = build_action_history_table(
        action_name_history=("e4", "e5", "Nf3", "Nc6", "Bb5"),
        participant_labels=("White", "Black"),
    )

    assert headers == ["White", "Black"]
    assert rows == [
        ["e4", "e5"],
        ["Nf3", "Nc6"],
        ["Bb5", ""],
    ]


def test_build_action_history_table_supports_solo_and_preserves_order() -> None:
    """Solo and multi-role layouts should respect the participant display order."""
    solo_headers, solo_rows = build_action_history_table(
        action_name_history=("x", "y", "z"),
        participant_labels=("Solo",),
    )
    tri_headers, tri_rows = build_action_history_table(
        action_name_history=("a1", "b1", "c1", "a2"),
        participant_labels=("Gamma", "Alpha", "Beta"),
    )

    assert solo_headers == ["Solo"]
    assert solo_rows == [["x"], ["y"], ["z"]]
    assert tri_headers == ["Gamma", "Alpha", "Beta"]
    assert tri_rows == [["a1", "b1", "c1"], ["a2", "", ""]]


def test_player_progress_collector_observable_publishes_role_keyed_progress() -> None:
    """Progress payloads should be keyed by arbitrary roles, not only colors."""
    out, publisher = make_publisher()
    progress = PlayerProgressCollectorObservable(publishers=[publisher])

    progress.progress("solo", 12)
    progress.progress(Color.BLACK, 88)

    first_payload = out.get_nowait().payload
    second_payload = out.get_nowait().payload
    assert isinstance(first_payload, UpdParticipantProgress)
    assert isinstance(second_payload, UpdParticipantProgress)
    assert (first_payload.role, first_payload.progress_percent) == (
        "solo",
        12,
    )
    assert (second_payload.role, second_payload.progress_percent) == (
        Color.BLACK,
        88,
    )


def test_observable_match_results_publish_participant_stats_with_legacy_aliases() -> None:
    """Match-result payloads should be participant-based with two-player aliases intact."""
    out, publisher = make_publisher()
    observable = ObservableMatchResults(
        match_results=MatchResults(
            player_one_name_id="player-one",
            player_two_name_id="player-two",
        ),
        publishers=[publisher],
    )

    observable.add_result_one_game(
        game_report=GameReport(
            final_game_result=FinalGameResult.WIN_FOR_WHITE,
            action_history=[],
            state_tag_history=[],
            participant_id_by_role={"White": "player-two", "Black": "player-one"},
            result_by_role={"White": RoleOutcome.WIN, "Black": RoleOutcome.LOSS},
            winner_roles=["White"],
        ),
    )
    first_payload = out.get_nowait().payload
    assert isinstance(first_payload, UpdMatchResults)
    assert [
        (participant.participant_id, participant.wins, participant.losses)
        for participant in first_payload.participant_stats
    ] == [
        ("player-one", 0, 1),
        ("player-two", 1, 0),
    ]
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
