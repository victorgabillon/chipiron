"""Generic player construction pipeline with game-specific builders."""

from __future__ import annotations

import random
from typing import Callable, TypeVar

from valanga import TurnState
from valanga.policy import BranchSelector

from chipiron.players.move_selector import factory as move_selector_factory
from chipiron.players.move_selector.tree_and_value_args import TreeAndValueAppArgs
from chipiron.players.oracles import PolicyOracle, TerminalOracle, ValueOracle
from chipiron.players.player import GameAdapter, Player

from ..utils.dataclass import IsDataclass
from ..utils.queue_protocols import PutQueue

SnapT = TypeVar("SnapT")
StateT = TypeVar("StateT", bound=TurnState)
EvalArgsT = TypeVar("EvalArgsT")
MasterEvalT = TypeVar("MasterEvalT")
NonTreeArgsT = TypeVar("NonTreeArgsT")


def create_player_with_pipeline(
    *,
    name: str,
    main_selector_args: TreeAndValueAppArgs[StateT, EvalArgsT] | NonTreeArgsT,
    state_type: type[StateT],
    policy_oracle: PolicyOracle[StateT] | None,
    value_oracle: ValueOracle[StateT] | None,
    terminal_oracle: TerminalOracle[StateT] | None,
    master_evaluator_from_args: Callable[
        [EvalArgsT, ValueOracle[StateT] | None, TerminalOracle[StateT] | None],
        MasterEvalT,
    ],
    adapter_builder: Callable[
        [BranchSelector[StateT], PolicyOracle[StateT] | None],
        GameAdapter[SnapT, StateT],
    ],
    create_non_tree_selector: Callable[[NonTreeArgsT], BranchSelector[StateT]],
    random_generator: random.Random,
    queue_progress_player: PutQueue[IsDataclass] | None,
) -> Player[SnapT, StateT]:
    """Create a player using a generic selection pipeline with game-specific builders."""
    match main_selector_args:
        case TreeAndValueAppArgs() as tree_args:
            master_state_evaluator = master_evaluator_from_args(
                tree_args.evaluator_args,
                value_oracle,
                terminal_oracle,
            )
            main_move_selector: BranchSelector[StateT] = (
                move_selector_factory.create_tree_and_value_move_selector(
                    args=tree_args.anemone_args,
                    state_type=state_type,
                    master_state_evaluator=master_state_evaluator,
                    state_representation_factory=None,
                    random_generator=random_generator,
                    queue_progress_player=queue_progress_player,
                )
            )
        case _:
            main_move_selector = create_non_tree_selector(
                main_selector_args,
            )

    adapter = adapter_builder(main_move_selector, policy_oracle)
    return Player(name=name, adapter=adapter)
