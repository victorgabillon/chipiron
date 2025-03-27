"""
This module provides functions for creating a TreeAndValueMoveSelector object.

The TreeAndValueMoveSelector is a player that uses a tree-based approach to select moves in a game. It evaluates the
game tree using a node evaluator and selects moves based on a set of criteria defined by the node selector. The player
uses a stopping criterion to determine when to stop the search and a recommender rule to recommend a move after
exploration.

This module also provides functions for creating the necessary components of the TreeAndValueMoveSelector, such as the
node evaluator, node selector, tree factory, and tree manager.

"""

import queue
import random
from dataclasses import dataclass
from typing import Any, Literal

import chipiron.players.move_selector.treevalue.search_factory as search_factories
from chipiron.players.boardevaluators.neural_networks.input_converters.factory import (
    RepresentationFactory,
)
from chipiron.players.boardevaluators.neural_networks.input_converters.representation_factory_factory import (
    create_board_representation_factory,
)
from chipiron.players.boardevaluators.table_base.syzygy_table import SyzygyTable
from chipiron.players.move_selector.move_selector_types import MoveSelectorTypes
from chipiron.players.move_selector.treevalue import node_factory
from chipiron.players.move_selector.treevalue.progress_monitor.progress_monitor import (
    AllStoppingCriterionArgs,
)
from chipiron.utils.dataclass import IsDataclass

from . import node_evaluator as node_eval
from . import node_selector as node_selector_m
from . import recommender_rule
from . import tree_manager as tree_man
from .indices.node_indices.index_types import IndexComputationType
from .tree_and_value_move_selector import TreeAndValueMoveSelector
from .trees.factory import MoveAndValueTreeFactory


@dataclass
class TreeAndValuePlayerArgs:
    """
    Data class for the arguments of a TreeAndValueMoveSelector.
    """

    type: Literal[MoveSelectorTypes.TreeAndValue]  # for serialization
    node_selector: node_selector_m.AllNodeSelectorArgs
    opening_type: node_selector_m.OpeningType
    board_evaluator: node_eval.AllNodeEvaluatorArgs
    stopping_criterion: AllStoppingCriterionArgs
    recommender_rule: recommender_rule.AllRecommendFunctionsArgs
    index_computation: IndexComputationType | None = None


def create_tree_and_value_builders(
    args: TreeAndValuePlayerArgs,
    syzygy: SyzygyTable[Any] | None,
    random_generator: random.Random,
    queue_progress_player: queue.Queue[IsDataclass] | None,
) -> TreeAndValueMoveSelector:
    """
    Create a TreeAndValueMoveSelector object with the given arguments.

    Args:
        args (TreeAndValuePlayerArgs): The arguments for creating the TreeAndValueMoveSelector.
        syzygy (SyzygyTable | None): The SyzygyTable object for tablebase endgame evaluation.
        random_generator (random.Random): The random number generator.

    Returns:
        TreeAndValueMoveSelector: The created TreeAndValueMoveSelector object.

    """

    node_evaluator: node_eval.NodeEvaluator = node_eval.create_node_evaluator(
        arg_board_evaluator=args.board_evaluator, syzygy=syzygy
    )

    # node_factory_name: str = args['node_factory_name'] if 'node_factory_name' in args else 'Base'
    node_factory_name: str = "Base_with_algorithm_tree_node"

    tree_node_factory: node_factory.Base[Any] = node_factory.create_node_factory(
        node_factory_name=node_factory_name
    )

    board_representation_factory: RepresentationFactory[Any] | None
    board_representation_factory = create_board_representation_factory(
        internal_tensor_representation_type=args.board_evaluator.internal_representation_type
    )

    search_factory: search_factories.SearchFactoryP = search_factories.SearchFactory(
        node_selector_args=args.node_selector,
        opening_type=args.opening_type,
        random_generator=random_generator,
        index_computation=args.index_computation,
    )

    algorithm_node_factory: node_factory.AlgorithmNodeFactory
    algorithm_node_factory = node_factory.AlgorithmNodeFactory(
        tree_node_factory=tree_node_factory,
        board_representation_factory=board_representation_factory,
        exploration_index_data_create=search_factory.node_index_create,
    )

    tree_factory = MoveAndValueTreeFactory(
        node_factory=algorithm_node_factory, node_evaluator=node_evaluator
    )

    tree_manager: tree_man.AlgorithmNodeTreeManager
    tree_manager = tree_man.create_algorithm_node_tree_manager(
        algorithm_node_factory=algorithm_node_factory,
        node_evaluator=node_evaluator,
        index_computation=args.index_computation,
        index_updater=search_factory.create_node_index_updater(),
    )

    tree_move_selector: TreeAndValueMoveSelector = TreeAndValueMoveSelector(
        tree_manager=tree_manager,
        random_generator=random_generator,
        tree_factory=tree_factory,
        node_selector_create=search_factory.create_node_selector_factory(),
        stopping_criterion_args=args.stopping_criterion,
        recommend_move_after_exploration=args.recommender_rule,
        queue_progress_player=queue_progress_player,
    )
    return tree_move_selector
