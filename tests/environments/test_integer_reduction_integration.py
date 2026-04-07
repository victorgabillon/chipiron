"""Focused integration tests for Chipiron integer reduction support."""

from __future__ import annotations

import json
import queue
from dataclasses import asdict
from typing import TYPE_CHECKING, cast

import yaml
from anemone import TreeAndValuePlayerArgs
from anemone.node_selector.composed.args import ComposedNodeSelectorArgs
from anemone.node_selector.node_selector_types import NodeSelectorType
from anemone.node_selector.opening_instructions import OpeningType
from anemone.node_selector.priority_check.noop_args import NoPriorityCheckArgs
from anemone.node_selector.uniform.uniform import UniformArgs
from anemone.progress_monitor.progress_monitor import (
    StoppingCriterionTypes,
    TreeBranchLimitArgs,
)
from anemone.recommender_rule.recommender_rule import AlmostEqualLogistic
from valanga import SOLO
from valanga.evaluations import Certainty

from chipiron.displays.gui_protocol import (
    GuiUpdate,
    HumanActionChosen,
    UpdNeedHumanAction,
    UpdNoHumanActionPending,
    UpdParticipantsInfo,
    UpdStateGeneric,
)
from chipiron.displays.integer_reduction_svg_adapter import IntegerReductionSvgAdapter
from chipiron.environments.chess.players.evaluators.boardevaluators.factory import (
    create_game_board_evaluator_for_game_kind,
)
from chipiron.environments.deps import IntegerReductionEnvironmentDeps
from chipiron.environments.environment import make_environment
from chipiron.environments.integer_reduction.integer_reduction_gui_encoder import (
    IntegerReductionDisplayPayload,
)
from chipiron.environments.integer_reduction.players.evaluators.integer_reduction_state_evaluator import (
    IntegerReductionStateEvaluator,
)
from chipiron.environments.integer_reduction.players.wiring.integer_reduction_wiring import (
    BuildIntegerReductionGamePlayerArgs,
    build_integer_reduction_game_player,
)
from chipiron.environments.integer_reduction.starting_position_args import (
    IntegerReductionValueStartingPositionArgs,
)
from chipiron.environments.integer_reduction.types import IntegerReductionState
from chipiron.environments.types import GameKind
from chipiron.games.domain.game.game_args import GameArgs
from chipiron.games.domain.game.game_args_factory import GameArgsFactory
from chipiron.games.domain.game.game_manager_factory import GameManagerFactory
from chipiron.games.domain.match.match_manager import MatchManager
from chipiron.games.domain.match.match_results import MatchResults
from chipiron.games.domain.match.match_results_factory import MatchResultsFactory
from chipiron.games.domain.match.match_role_schedule import (
    SoloMatchSchedule,
    build_validated_match_plan,
)
from chipiron.players import PlayerArgs, PlayerFactoryArgs
from chipiron.players.boardevaluators.all_board_evaluator_args import (
    BasicEvaluationBoardEvaluatorArgs,
)
from chipiron.players.boardevaluators.master_board_evaluator_args import (
    MasterBoardEvaluatorArgs,
)
from chipiron.players.move_selector.human import GuiHumanPlayerArgs
from chipiron.players.move_selector.move_selector_types import MoveSelectorTypes
from chipiron.players.move_selector.random_args import RandomSelectorArgs
from chipiron.players.move_selector.tree_and_value_args import (
    NodeEvaluatorArgs,
    TreeAndValueAppArgs,
)
from chipiron.players.player_ids import PlayerConfigTag
from chipiron.scripts.chipiron_args import ImplementationArgs

if TYPE_CHECKING:
    from pathlib import Path

    import pytest
    from atomheart.games.chess.move.move_factory import MoveFactory

    from chipiron.utils.communication.mailbox import MainMailboxMessage


def make_player_args(*, name: str, human: bool) -> PlayerArgs:
    """Build simple player args for integer reduction tests."""
    selector = (
        GuiHumanPlayerArgs(type=MoveSelectorTypes.GUI_HUMAN)
        if human
        else RandomSelectorArgs()
    )
    return PlayerArgs(
        name=name,
        main_move_selector=selector,
        oracle_play=False,
    )


def make_game_args(*, value: int) -> GameArgs:
    """Build integer-reduction game args with a concrete starting value."""
    return GameArgs(
        game_kind=GameKind.INTEGER_REDUCTION,
        starting_position=IntegerReductionValueStartingPositionArgs(value=value),
        max_half_moves=None,
        each_player_has_its_own_thread=False,
    )


def make_tree_and_value_selector() -> TreeAndValueAppArgs:
    """Build a minimal tree selector to assert unsupported integer-reduction wiring."""
    return TreeAndValueAppArgs(
        anemone_args=TreeAndValuePlayerArgs(
            node_selector=ComposedNodeSelectorArgs(
                type=NodeSelectorType.COMPOSED,
                priority=NoPriorityCheckArgs(type=NodeSelectorType.PRIORITY_NOOP),
                base=UniformArgs(type=NodeSelectorType.UNIFORM),
            ),
            opening_type=OpeningType.ALL_CHILDREN,
            stopping_criterion=TreeBranchLimitArgs(
                type=StoppingCriterionTypes.TREE_BRANCH_LIMIT,
                tree_branch_limit=8,
            ),
            recommender_rule=AlmostEqualLogistic(
                type="almost_equal_logistic",
                temperature=1.0,
            ),
        ),
        evaluator_args=NodeEvaluatorArgs(
            master_board_evaluator=MasterBoardEvaluatorArgs(
                board_evaluator=BasicEvaluationBoardEvaluatorArgs(
                    type="basic_evaluation"
                ),
                oracle_evaluation=False,
            )
        ),
    )


def make_game_manager_factory(
    *,
    gui_queue: queue.Queue[GuiUpdate] | None = None,
) -> GameManagerFactory:
    """Build a real game-manager factory for integer reduction tests."""
    main_thread_mailbox: queue.Queue[MainMailboxMessage] = queue.Queue()
    factory = GameManagerFactory(
        env_deps=IntegerReductionEnvironmentDeps(),
        game_manager_state_evaluator=create_game_board_evaluator_for_game_kind(
            game_kind=GameKind.INTEGER_REDUCTION,
            gui=gui_queue is not None,
            can_oracle=True,
        ),
        output_folder_path=None,
        main_thread_mailbox=main_thread_mailbox,
        implementation_args=ImplementationArgs(),
        move_factory=cast("MoveFactory", object()),
        universal_behavior=False,
        session_id="session-integer-reduction",
        match_id="match-integer-reduction",
    )
    if gui_queue is not None:
        factory.subscribe(gui_queue)
    return factory


def drain_payloads(gui_queue: queue.Queue[GuiUpdate]) -> list[object]:
    """Drain GUI update payloads from a queue for assertions."""
    payloads: list[object] = []
    while not gui_queue.empty():
        payloads.append(gui_queue.get_nowait().payload)
    return payloads


def test_integer_reduction_environment_declares_solo_role_and_readable_payloads() -> (
    None
):
    """Environment assembly should expose one real solo role and readable GUI data."""
    environment = make_environment(
        game_kind=GameKind.INTEGER_REDUCTION,
        deps=IntegerReductionEnvironmentDeps(),
    )
    state = environment.make_initial_state(
        environment.normalize_start_tag(
            IntegerReductionValueStartingPositionArgs(value=8).get_start_tag()
        )
    )
    payload = environment.gui_encoder.make_state_payload(state=state, seed=13)
    match_plan = build_validated_match_plan(
        participant_ids=("SoloHuman",),
        environment_roles=environment.roles,
        schedule=SoloMatchSchedule(number_of_games=1),
    )

    assert environment.roles == (SOLO,)
    assert match_plan.scheduled_roles == (SOLO,)
    assert match_plan.is_solo is True
    assert state.value == 8
    assert state.steps == 0
    assert state.turn == SOLO
    assert isinstance(payload, UpdStateGeneric)
    assert payload.state_tag == (8, 0)
    assert isinstance(payload.adapter_payload, IntegerReductionDisplayPayload)
    assert payload.adapter_payload.steps == 0
    assert payload.adapter_payload.legal_actions == ("dec1", "half")

    adapter = IntegerReductionSvgAdapter()
    pos = adapter.position_from_update(
        state_tag=payload.state_tag,
        adapter_payload=payload.adapter_payload,
    )
    render = adapter.render_svg(pos, size=600, margin=0)
    first_button = adapter._buttons[0]
    click = adapter.handle_click(
        pos,
        x=int(first_button.x + first_button.width / 2),
        y=int(first_button.y + first_button.height / 2),
        board_size=600,
        margin=0,
    )

    assert b"Integer Reduction" in render.svg_bytes
    assert render.info["fen"] == "value=8 steps=0"
    assert render.info["legal_moves"] == "dec1, half"
    assert click.action_name == payload.adapter_payload.legal_actions[0]


def test_integer_reduction_solo_schedule_reports_one_game() -> None:
    """Solo integer-reduction scheduling should be represented directly."""
    assert SoloMatchSchedule(number_of_games=1).total_games == 1


def test_integer_reduction_debug_tree_player_records_move_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The debug tree player should emit one live-debug session per recommendation."""
    monkeypatch.setenv("CHIPIRON_OUTPUT_DIR", str(tmp_path))

    player = build_integer_reduction_game_player(
        BuildIntegerReductionGamePlayerArgs(
            player_factory_args=PlayerFactoryArgs(
                player_args=PlayerArgs(
                    name=PlayerConfigTag.INTEGER_REDUCTION_TREE_BASIC_DEBUG.value,
                    main_move_selector=make_tree_and_value_selector(),
                    oracle_play=False,
                ),
                seed=11,
            ),
            player_role=SOLO,
            implementation_args=None,
            universal_behavior=False,
        )
    )

    recommendation = player.select_move_from_snapshot(
        snapshot=15,
        seed=23,
        notify_percent_function=lambda _progress: None,
    )

    debug_root = tmp_path / "runs" / "debug" / "integer_reduction"
    session_roots = sorted(path for path in debug_root.iterdir() if path.is_dir())
    assert len(session_roots) == 1

    move_directories = sorted(
        path for path in session_roots[0].iterdir() if path.is_dir()
    )
    assert len(move_directories) == 1
    move_directory = move_directories[0]

    summary = json.loads(
        (move_directory / "move_summary.json").read_text(encoding="utf-8")
    )
    session_payload = json.loads(
        (move_directory / "session.json").read_text(encoding="utf-8")
    )

    assert move_directory.name.startswith("move_000_state_15")
    assert summary["move_index"] == 0
    assert summary["state_debug"] == "15"
    assert summary["recommended_move_name"] == recommendation.recommended_name
    assert (move_directory / "index.html").exists()
    assert (move_directory / "snapshots").is_dir()
    assert session_payload["is_live"] is True
    assert session_payload["entry_count"] >= 1


def test_integer_reduction_human_session_emits_need_action_and_advances_state() -> None:
    """The solo human role should get a standard need-action request and advance cleanly."""
    gui_queue: queue.Queue[GuiUpdate] = queue.Queue()
    factory = make_game_manager_factory(gui_queue=gui_queue)
    session = factory.create(
        args_game_manager=make_game_args(value=8),
        participant_factory_args_by_role={
            SOLO: PlayerFactoryArgs(
                player_args=make_player_args(name="SoloHuman", human=True),
                seed=1,
            )
        },
        game_seed=17,
    )

    initial_payloads = drain_payloads(gui_queue)
    assert len(initial_payloads) == 1
    assert isinstance(initial_payloads[0], UpdParticipantsInfo)
    assert [participant.role for participant in initial_payloads[0].participants] == [
        SOLO
    ]

    session.manager.game.notify_display()
    display_payloads = drain_payloads(gui_queue)
    assert len(display_payloads) == 1
    assert isinstance(display_payloads[0], UpdStateGeneric)
    assert display_payloads[0].action_name_history == []
    assert isinstance(
        display_payloads[0].adapter_payload, IntegerReductionDisplayPayload
    )
    assert display_payloads[0].adapter_payload.value == 8
    assert display_payloads[0].adapter_payload.steps == 0

    session.controller.start()
    start_payloads = drain_payloads(gui_queue)
    assert len(start_payloads) == 1
    assert isinstance(start_payloads[0], UpdNeedHumanAction)
    assert start_payloads[0].ctx.role_to_play == SOLO

    session.controller.handle_human_action(
        HumanActionChosen(
            action_name="half",
            ctx=start_payloads[0].ctx,
            corresponding_state_tag=start_payloads[0].state_tag,
        )
    )

    assert session.manager.game.state.value == 4
    assert session.manager.game.state.steps == 1
    after_action_payloads = drain_payloads(gui_queue)
    assert isinstance(after_action_payloads[0], UpdNoHumanActionPending)
    assert isinstance(after_action_payloads[1], UpdStateGeneric)
    assert after_action_payloads[1].action_name_history == ["half"]
    assert after_action_payloads[1].adapter_payload.value == 4
    assert after_action_payloads[1].adapter_payload.steps == 1
    assert isinstance(after_action_payloads[2], UpdNeedHumanAction)
    assert after_action_payloads[2].ctx.role_to_play == SOLO


def test_integer_reduction_random_game_terminates_and_serializes_role_aware_report() -> (
    None
):
    """A random solo player should finish the game and produce a role-aware report."""
    factory = make_game_manager_factory()
    session = factory.create(
        args_game_manager=make_game_args(value=7),
        participant_factory_args_by_role={
            SOLO: PlayerFactoryArgs(
                player_args=make_player_args(name="SoloRandom", human=False),
                seed=3,
            )
        },
        game_seed=23,
    )

    game_report = session.manager.play_one_game(session.controller)

    assert game_report.action_history
    assert set(game_report.action_history) <= {"dec1", "half"}
    assert game_report.participant_id_by_role == {"Solo": "SoloRandom"}
    assert game_report.result_by_role == {"Solo": "win"}
    assert game_report.winner_roles == ["Solo"]
    assert game_report.result_reason == "reached_one"
    serialized_report = asdict(game_report)
    serialized_report["result_by_role"] = {
        role: str(outcome)
        for role, outcome in serialized_report["result_by_role"].items()
    }
    dumped = yaml.safe_dump(serialized_report, sort_keys=True)
    assert "SoloRandom" in dumped
    assert "reached_one" in dumped

    match_results = MatchResults(participant_ids=("SoloRandom",))
    match_results.add_result_one_game(game_report=game_report)
    simple = match_results.get_simple_result()
    assert simple.wins_by_participant == {"SoloRandom": 1}
    assert simple.stats_by_participant["SoloRandom"].losses == 0
    assert simple.games_played == 1


def test_integer_reduction_match_manager_plays_one_solo_match() -> None:
    """The match layer should support a one-participant integer-reduction run."""
    player_args = make_player_args(name="SoloRandom", human=False)
    game_args = make_game_args(value=9)
    match_plan = build_validated_match_plan(
        participant_ids=("SoloRandom",),
        environment_roles=(SOLO,),
        schedule=SoloMatchSchedule(number_of_games=1),
    )
    match_manager = MatchManager(
        game_manager_factory=make_game_manager_factory(),
        game_args_factory=GameArgsFactory(
            args_player_one=player_args,
            args_player_two=None,
            seed_=29,
            args_game=game_args,
            match_plan=match_plan,
        ),
        match_results_factory=MatchResultsFactory(match_plan=match_plan),
        output_folder_path=None,
    )

    match_report = match_manager.play_one_match()
    simple = match_report.match_results.get_simple_result()

    assert match_report.match_results.participant_ids == ("SoloRandom",)
    assert set(match_report.match_move_history[0]) <= {"dec1", "half"}
    assert simple.wins_by_participant == {"SoloRandom": 1}
    assert simple.draws == 0
    assert simple.games_played == 1


def test_integer_reduction_evaluator_prefers_fewer_steps_and_marks_terminal() -> None:
    """The tree-search heuristic should reward fewer steps and mark terminal certainty."""
    evaluator = IntegerReductionStateEvaluator()

    quick_state = evaluator.evaluate(IntegerReductionState(value=2, steps=2))
    slow_state = evaluator.evaluate(IntegerReductionState(value=2, steps=5))
    terminal_state = evaluator.evaluate(IntegerReductionState(value=1, steps=3))

    assert quick_state.score > slow_state.score
    assert quick_state.score == -2.0
    assert slow_state.score == -5.0
    assert terminal_state.score == -3.0
    assert terminal_state.certainty is Certainty.TERMINAL


def test_integer_reduction_tree_selector_builds_and_prefers_half_when_available() -> (
    None
):
    """Tree selectors should build successfully and pick the better smaller successor."""
    game_player = build_integer_reduction_game_player(
        BuildIntegerReductionGamePlayerArgs(
            player_factory_args=PlayerFactoryArgs(
                player_args=PlayerArgs(
                    name="SoloTree",
                    main_move_selector=make_tree_and_value_selector(),
                    oracle_play=False,
                ),
                seed=5,
            ),
            player_role=SOLO,
            implementation_args=ImplementationArgs(),
            universal_behavior=False,
        )
    )

    recommendation = game_player.select_move_from_snapshot(
        snapshot=8,
        seed=0,
        notify_percent_function=lambda _progress: None,
    )

    assert recommendation.recommended_name == "half"
