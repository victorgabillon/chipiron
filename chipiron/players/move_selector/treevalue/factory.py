import random

from .tree_and_value_player import TreeAndValueMoveSelector
from . import tree_manager as tree_man
from . import node_evaluator as node_eval
from chipiron.players.move_selector.treevalue import node_factory

from .trees.factory import MoveAndValueTreeFactory

from chipiron.players.boardevaluators.neural_networks.input_converters.factory import create_board_representation

from . import node_selector
from .node_indices.types import IndexComputationType
from .stopping_criterion import AllStoppingCriterionArgs
from . import recommender_rule
from typing import Literal
from dataclasses import dataclass

Tree_Value_Name_Literal: str = 'TreeAndValue'


@dataclass
class TreeAndValuePlayerArgs:
    type: Literal[Tree_Value_Name_Literal]  # for serialization
    node_selector: node_selector.AllNodeSelectorArgs
    opening_type: node_selector.OpeningType
    board_evaluator: node_eval.AllNodeEvaluatorArgs
    stopping_criterion: AllStoppingCriterionArgs
    recommender_rule: recommender_rule.AllRecommendFunctionsArgs
    index_computation: IndexComputationType | None = None


def create_tree_and_value_builders(args: TreeAndValuePlayerArgs,
                                   syzygy,
                                   random_generator: random.Random) -> TreeAndValueMoveSelector:
    node_evaluator: node_eval.NodeEvaluator = node_eval.create_node_evaluator(
        arg_board_evaluator=args.board_evaluator,
        syzygy=syzygy
    )

    # node_factory_name: str = args['node_factory_name'] if 'node_factory_name' in args else 'Base'
    node_factory_name: str = 'Base'

    tree_node_factory: node_factory.Base = node_factory.create_node_factory(
        node_factory_name=node_factory_name
    )

    board_representation_factory: object | None = None
    board_representation_factory = create_board_representation(
        board_representation=args.board_evaluator.representation
    )

    algorithm_node_factory: node_factory.AlgorithmNodeFactory
    algorithm_node_factory = node_factory.AlgorithmNodeFactory(
        tree_node_factory=tree_node_factory,
        board_representation_factory=board_representation_factory,
        index_computation=args.index_computation
    )

    tree_factory = MoveAndValueTreeFactory(
        node_factory=algorithm_node_factory,
        node_evaluator=node_evaluator
    )

    tree_manager: tree_man.AlgorithmNodeTreeManager
    tree_manager = tree_man.create_algorithm_node_tree_manager(
        algorithm_node_factory=algorithm_node_factory,
        node_evaluator=node_evaluator,
        index_computation=args.index_computation
    )

    tree_move_selector: TreeAndValueMoveSelector = TreeAndValueMoveSelector(opening_type=args.opening_type,
                                                                            tree_manager=tree_manager,
                                                                            random_generator=random_generator,
                                                                            tree_factory=tree_factory,
                                                                            node_selector_args=args.node_selector,
                                                                            stopping_criterion_args=args.stopping_criterion,
                                                                            recommend_move_after_exploration=args.recommender_rule)
    return tree_move_selector
