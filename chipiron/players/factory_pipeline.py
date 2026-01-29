"""Generic player construction pipeline with game-specific builders."""

from __future__ import annotations

import random
from typing import Callable, TypeVar

from anemone.node_evaluation.node_direct_evaluation.node_direct_evaluator import (
    MasterStateEvaluator,
)
from valanga import TurnState
from valanga.policy import BranchSelector

from chipiron.players.boardevaluators.master_board_evaluator import (
    MasterBoardEvaluatorArgs,
)
from chipiron.players.move_selector import factory as move_selector_factory
from chipiron.players.move_selector.human import GuiHumanPlayerArgs
from chipiron.players.move_selector.move_selector_args import AnyMoveSelectorArgs
from chipiron.players.move_selector.tree_and_value_args import TreeAndValueAppArgs
from chipiron.players.oracles import PolicyOracle, TerminalOracle, ValueOracle
from chipiron.players.player import GameAdapter, Player
from chipiron.players.player_args import HasMoveSelectorType
from chipiron.utils.communication.mailbox import MainMailboxMessage

from ..utils.queue_protocols import PutQueue

SnapT = TypeVar("SnapT")
StateT = TypeVar("StateT", bound=TurnState)
EvalArgsT = TypeVar("EvalArgsT")
NonTreeArgsT = TypeVar("NonTreeArgsT", bound=HasMoveSelectorType)


def create_player_with_pipeline(
    *,
    name: str,
    main_selector_args: AnyMoveSelectorArgs,
    state_type: type[StateT],
    policy_oracle: PolicyOracle[StateT] | None,
    value_oracle: ValueOracle[StateT] | None,
    terminal_oracle: TerminalOracle[StateT] | None,
    master_evaluator_from_args: Callable[
        [
            MasterBoardEvaluatorArgs,
            ValueOracle[StateT] | None,
            TerminalOracle[StateT] | None,
        ],
        "MasterStateEvaluator",
    ],
    adapter_builder: Callable[
        [BranchSelector[StateT], PolicyOracle[StateT] | None],
        GameAdapter[SnapT, StateT],
    ],
    create_non_tree_selector: Callable[
        [move_selector_factory.NonTreeMoveSelectorArgs], BranchSelector[StateT]
    ],
    random_generator: random.Random,
    queue_progress_player: PutQueue[MainMailboxMessage] | None,
) -> Player[SnapT, StateT]:
    """Create a player using a generic selection pipeline with game-specific builders."""

    if isinstance(main_selector_args, TreeAndValueAppArgs):
        master_state_evaluator = master_evaluator_from_args(
            main_selector_args.evaluator_args.master_board_evaluator,
            value_oracle,
            terminal_oracle,
        )
        main_move_selector = move_selector_factory.create_tree_and_value_move_selector(
            args=main_selector_args.anemone_args,
            state_type=state_type,
            master_state_evaluator=master_state_evaluator,
            state_representation_factory=None,
            random_generator=random_generator,
            queue_progress_player=queue_progress_player,
        )
    elif isinstance(main_selector_args, GuiHumanPlayerArgs):
        # Minimal behavior: fail fast with a clear message (or implement GUI path)
        raise NotImplementedError(
            "GuiHumanPlayerArgs is handled by the GUI->GameManager command path, "
            "not by create_main_move_selector."
        )
    else:
        main_move_selector = create_non_tree_selector(main_selector_args)

    adapter = adapter_builder(main_move_selector, policy_oracle)
    return Player(name=name, adapter=adapter)
