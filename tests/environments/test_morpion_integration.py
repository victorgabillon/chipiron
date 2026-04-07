"""Focused integration tests for Chipiron Morpion support."""

from __future__ import annotations

import queue
from typing import TYPE_CHECKING, cast

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
from atomheart.games.morpion import MorpionState as AtomMorpionState
from atomheart.games.morpion.state import Variant as MorpionVariant
from valanga import SOLO

from chipiron.displays.gui_protocol import (
    GuiUpdate,
    HumanActionChosen,
    UpdNeedHumanAction,
    UpdNoHumanActionPending,
    UpdParticipantsInfo,
    UpdStateGeneric,
)
from chipiron.displays.morpion_svg_adapter import MorpionSvgAdapter
from chipiron.environments.chess.players.evaluators.boardevaluators.factory import (
    create_game_board_evaluator_for_game_kind,
)
from chipiron.environments.deps import MorpionEnvironmentDeps
from chipiron.environments.environment import make_environment
from chipiron.environments.morpion.morpion_gui_encoder import MorpionDisplayPayload
from chipiron.environments.morpion.morpion_rules import MorpionRules
from chipiron.environments.morpion.players.evaluators.morpion_state_evaluator import (
    MorpionStateEvaluator,
)
from chipiron.environments.morpion.players.wiring.morpion_wiring import (
    BuildMorpionGamePlayerArgs,
    build_morpion_game_player,
)
from chipiron.environments.morpion.starting_position_args import (
    MorpionStandardStartingPositionArgs,
)
from chipiron.environments.morpion.types import MorpionDynamics, MorpionState
from chipiron.environments.types import GameKind
from chipiron.games.domain.game.game_args import GameArgs
from chipiron.games.domain.game.game_manager_factory import GameManagerFactory
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
from chipiron.scripts.chipiron_args import ImplementationArgs

if TYPE_CHECKING:
    from atomheart.games.chess.move.move_factory import MoveFactory

    from chipiron.utils.communication.mailbox import MainMailboxMessage


def make_player_args(*, name: str, human: bool) -> PlayerArgs:
    """Build simple player args for Morpion tests."""
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


def make_game_args() -> GameArgs:
    """Build standard Morpion game args."""
    return GameArgs(
        game_kind=GameKind.MORPION,
        starting_position=MorpionStandardStartingPositionArgs(),
        max_half_moves=None,
        each_player_has_its_own_thread=False,
    )


def make_tree_and_value_selector() -> TreeAndValueAppArgs:
    """Build a tree selector close to the shipped Morpion config."""
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
                tree_branch_limit=128,
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
    """Build a real game-manager factory for Morpion tests."""
    main_thread_mailbox: queue.Queue[MainMailboxMessage] = queue.Queue()
    factory = GameManagerFactory(
        env_deps=MorpionEnvironmentDeps(),
        game_manager_state_evaluator=create_game_board_evaluator_for_game_kind(
            game_kind=GameKind.MORPION,
            gui=gui_queue is not None,
            can_oracle=True,
        ),
        output_folder_path=None,
        main_thread_mailbox=main_thread_mailbox,
        implementation_args=ImplementationArgs(),
        move_factory=cast("MoveFactory", object()),
        universal_behavior=False,
        session_id="session-morpion",
        match_id="match-morpion",
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


def make_standard_state() -> MorpionState:
    """Build the standard Morpion runtime state."""
    environment = make_environment(
        game_kind=GameKind.MORPION,
        deps=MorpionEnvironmentDeps(),
    )
    return environment.make_initial_state(
        environment.normalize_start_tag(MorpionStandardStartingPositionArgs().get_start_tag())
    )


def test_morpion_environment_declares_solo_role_and_readable_payloads() -> None:
    """Environment assembly should expose one real solo role and readable GUI data."""
    environment = make_environment(
        game_kind=GameKind.MORPION,
        deps=MorpionEnvironmentDeps(),
    )
    state = environment.make_initial_state(
        environment.normalize_start_tag(MorpionStandardStartingPositionArgs().get_start_tag())
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
    assert state.moves == 0
    assert state.turn == SOLO
    assert state.variant is MorpionVariant.TOUCHING_5T
    assert state.is_game_over() is False
    assert len(state.points) == 36
    assert isinstance(payload, UpdStateGeneric)
    assert isinstance(payload.adapter_payload, MorpionDisplayPayload)
    assert payload.adapter_payload.variant == "5T"
    assert payload.adapter_payload.moves == 0
    assert payload.adapter_payload.point_count == 36
    assert len(payload.adapter_payload.points) == 36
    assert payload.adapter_payload.segments == ()
    assert len(payload.adapter_payload.legal_moves) == 20
    assert payload.adapter_payload.legal_moves[0].new_point not in payload.adapter_payload.points

    adapter = MorpionSvgAdapter()
    pos = adapter.position_from_update(
        state_tag=payload.state_tag,
        adapter_payload=payload.adapter_payload,
    )
    render = adapter.render_svg(pos, size=600, margin=0)
    first_target = adapter._click_targets[0]
    click = adapter.handle_click(
        pos,
        x=int(round(first_target[1])),
        y=int(round(first_target[2])),
        board_size=600,
        margin=0,
    )

    assert b"Morpion Solitaire" in render.svg_bytes
    assert b"<line" in render.svg_bytes
    assert b"<circle" in render.svg_bytes
    assert render.info["fen"] == "variant=5T moves=0 points=36"
    assert click.action_name == payload.adapter_payload.legal_moves[0].action_name


def test_morpion_wrapped_terminal_state_keeps_rules_and_gui_in_sync() -> None:
    """Wrapped terminal Morpion states should report terminality consistently."""
    dynamics = MorpionDynamics()
    rules = MorpionRules()
    atom_state = AtomMorpionState(
        points=frozenset(),
        used_unit_segments=frozenset(),
        dir_usage={},
        moves=7,
        variant=MorpionVariant.TOUCHING_5T,
    )

    state = dynamics.wrap_atomheart_state(atom_state)
    payload = make_environment(
        game_kind=GameKind.MORPION,
        deps=MorpionEnvironmentDeps(),
    ).gui_encoder.make_state_payload(state=state, seed=19)
    outcome = rules.outcome(state)

    assert state.is_game_over() is True
    assert isinstance(payload, UpdStateGeneric)
    assert payload.adapter_payload.is_terminal is True
    assert payload.adapter_payload.legal_moves == ()
    assert outcome is not None
    assert outcome.winner == SOLO
    assert outcome.reason == "no_legal_moves"


def test_morpion_evaluator_rewards_progress() -> None:
    """The simple Morpion evaluator should prefer states with more progress."""
    evaluator = MorpionStateEvaluator()
    dynamics = MorpionDynamics()

    start = make_standard_state()
    first_action = dynamics.legal_actions(start).get_all()[0]
    progressed = dynamics.step(start, first_action).next_state

    assert evaluator.evaluate(progressed).score > evaluator.evaluate(start).score


def test_morpion_human_session_emits_need_action_and_advances_state() -> None:
    """The solo human role should get a standard need-action request and advance cleanly."""
    gui_queue: queue.Queue[GuiUpdate] = queue.Queue()
    factory = make_game_manager_factory(gui_queue=gui_queue)
    session = factory.create(
        args_game_manager=make_game_args(),
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
    assert isinstance(display_payloads[0].adapter_payload, MorpionDisplayPayload)
    first_action_name = display_payloads[0].adapter_payload.legal_moves[0].action_name

    session.controller.start()
    start_payloads = drain_payloads(gui_queue)
    assert len(start_payloads) == 1
    assert isinstance(start_payloads[0], UpdNeedHumanAction)
    assert start_payloads[0].ctx.role_to_play == SOLO

    session.controller.handle_human_action(
        HumanActionChosen(
            action_name=first_action_name,
            ctx=start_payloads[0].ctx,
            corresponding_state_tag=start_payloads[0].state_tag,
        )
    )

    assert session.manager.game.state.moves == 1
    assert len(session.manager.game.state.points) == 37
    after_action_payloads = drain_payloads(gui_queue)
    assert isinstance(after_action_payloads[0], UpdNoHumanActionPending)
    assert isinstance(after_action_payloads[1], UpdStateGeneric)
    assert after_action_payloads[1].adapter_payload.moves == 1
    assert after_action_payloads[1].adapter_payload.point_count == 37
    assert len(after_action_payloads[1].adapter_payload.segments) == 4
    assert isinstance(after_action_payloads[2], UpdNeedHumanAction)
    assert after_action_payloads[2].ctx.role_to_play == SOLO


def test_morpion_random_player_builds_and_recommends_a_legal_move() -> None:
    """Random Morpion players should build and return a legal move."""
    state = make_standard_state()
    dynamics = MorpionDynamics()
    legal_names = {
        dynamics.action_name(state, action)
        for action in dynamics.legal_actions(state).get_all()
    }

    player = build_morpion_game_player(
        BuildMorpionGamePlayerArgs(
            player_factory_args=PlayerFactoryArgs(
                player_args=PlayerArgs(
                    name="SoloRandom",
                    main_move_selector=RandomSelectorArgs(),
                    oracle_play=False,
                ),
                seed=3,
            ),
            player_role=SOLO,
            implementation_args=ImplementationArgs(),
            universal_behavior=False,
        )
    )

    recommendation = player.select_move_from_snapshot(
        snapshot=state,
        seed=0,
        notify_percent_function=lambda _progress: None,
    )

    assert recommendation.recommended_name in legal_names


def test_morpion_tree_selector_builds_and_uses_single_agent_search_semantics() -> None:
    """Tree selectors should succeed on SOLO Morpion states via the single-agent max family."""
    state = make_standard_state()
    dynamics = MorpionDynamics()
    legal_names = {
        dynamics.action_name(state, action)
        for action in dynamics.legal_actions(state).get_all()
    }

    game_player = build_morpion_game_player(
        BuildMorpionGamePlayerArgs(
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
        snapshot=state,
        seed=0,
        notify_percent_function=lambda _progress: None,
    )

    assert recommendation.recommended_name in legal_names
