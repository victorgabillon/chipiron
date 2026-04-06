"""Focused integration tests for Chipiron integer reduction support."""

from __future__ import annotations

import queue
from dataclasses import asdict
from typing import TYPE_CHECKING, cast

import pytest
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
from chipiron.environments.integer_reduction.players.wiring.integer_reduction_wiring import (
    BuildIntegerReductionGamePlayerArgs,
    UnsupportedIntegerReductionTreeSelectorError,
    build_integer_reduction_game_player,
)
from chipiron.environments.integer_reduction.starting_position_args import (
    IntegerReductionValueStartingPositionArgs,
)
from chipiron.environments.types import GameKind
from chipiron.games.domain.game.game_args import GameArgs
from chipiron.games.domain.game.game_args_factory import GameArgsFactory
from chipiron.games.domain.game.game_manager_factory import GameManagerFactory
from chipiron.games.domain.match.match_factories import (
    validate_supported_match_topology,
)
from chipiron.games.domain.match.match_manager import MatchManager
from chipiron.games.domain.match.match_results import MatchResults
from chipiron.games.domain.match.match_results_factory import MatchResultsFactory
from chipiron.games.domain.match.match_settings_args import MatchSettingsArgs
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
from chipiron.scripts.chipiron_args import ImplementationArgs

if TYPE_CHECKING:
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


def test_integer_reduction_environment_declares_solo_role_and_readable_payloads() -> None:
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

    assert environment.roles == (SOLO,)
    assert validate_supported_match_topology(
        participant_ids=("SoloHuman",),
        environment_roles=environment.roles,
    ) == (SOLO,)
    assert state.value == 8
    assert state.turn == SOLO
    assert isinstance(payload, UpdStateGeneric)
    assert payload.state_tag == 8
    assert isinstance(payload.adapter_payload, IntegerReductionDisplayPayload)
    assert payload.adapter_payload.legal_actions == ("dec1", "half")

    adapter = IntegerReductionSvgAdapter()
    pos = adapter.position_from_update(
        state_tag=payload.state_tag,
        adapter_payload=payload.adapter_payload,
    )
    render = adapter.render_svg(pos, size=600, margin=0)
    click = adapter.handle_click(pos, x=100, y=200, board_size=600, margin=0)

    assert b"Integer Reduction" in render.svg_bytes
    assert render.info["fen"] == "value=8"
    assert render.info["legal_moves"] == "dec1, half"
    assert click.action_name == "dec1"


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
    assert isinstance(display_payloads[0].adapter_payload, IntegerReductionDisplayPayload)
    assert display_payloads[0].adapter_payload.value == 8

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
    after_action_payloads = drain_payloads(gui_queue)
    assert isinstance(after_action_payloads[0], UpdNoHumanActionPending)
    assert isinstance(after_action_payloads[1], UpdStateGeneric)
    assert after_action_payloads[1].action_name_history == ["half"]
    assert after_action_payloads[1].adapter_payload.value == 4
    assert isinstance(after_action_payloads[2], UpdNeedHumanAction)
    assert after_action_payloads[2].ctx.role_to_play == SOLO


def test_integer_reduction_random_game_terminates_and_serializes_role_aware_report() -> None:
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
    match_manager = MatchManager(
        participant_ids=("SoloRandom",),
        game_manager_factory=make_game_manager_factory(),
        game_args_factory=GameArgsFactory(
            args_match=MatchSettingsArgs(
                number_of_games_player_one_white=1,
                number_of_games_player_one_black=0,
                game_args=game_args,
            ),
            args_player_one=player_args,
            args_player_two=None,
            seed_=29,
            args_game=game_args,
            scheduled_roles=(SOLO,),
        ),
        match_results_factory=MatchResultsFactory(participant_ids=("SoloRandom",)),
        output_folder_path=None,
    )

    match_report = match_manager.play_one_match()
    simple = match_report.match_results.get_simple_result()

    assert match_report.match_results.participant_ids == ("SoloRandom",)
    assert set(match_report.match_move_history[0]) <= {"dec1", "half"}
    assert simple.wins_by_participant == {"SoloRandom": 1}
    assert simple.draws == 0
    assert simple.games_played == 1


def test_integer_reduction_tree_selector_remains_explicitly_unsupported() -> None:
    """Tree selectors should fail clearly instead of silently taking a wrong path."""
    with pytest.raises(UnsupportedIntegerReductionTreeSelectorError):
        build_integer_reduction_game_player(
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
