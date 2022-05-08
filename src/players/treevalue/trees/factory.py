from src.players.treevalue.trees.move_and_value_tree import MoveAndValueTree
from src.players.treevalue.trees.imove_and_value_tree import IMoveAndValueTree
from src.players.treevalue.trees.move_tree_value_tree_observable import MoveAndValueTreeObservable as Observable
from src.players.treevalue.node_factory.RecurZipfBase import RecurZipfBase as RecurZipfBaseNodeFactory
from src.chessenvironment.board import iboard
from src.players.treevalue.node_selector.node_selector import NodeSelector
from src.players.treevalue.node_factory.factory import TreeNodeFactory
from typing import List


def create_move_and_value_tree(args: dict,
                               board_evaluators_wrapper,
                               board: iboard,
                               subscribers: List[NodeSelector]) -> IMoveAndValueTree:
    tree_builder_type: str = args['type']

    node_factory: TreeNodeFactory
    match tree_builder_type:
        case 'RecurZipfBase':
            node_factory = RecurZipfBaseNodeFactory()
        case other:
            raise 'please implement your node factory!! no {}'.format(other)

    move_and_value_tree: IMoveAndValueTree = MoveAndValueTree(board_evaluator=board_evaluators_wrapper,
                                                              starting_board=board,
                                                              node_factory=node_factory)

    if subscribers:
        move_and_value_tree: IMoveAndValueTree = Observable(move_and_vale_tree=move_and_value_tree,
                                                            subscribers=subscribers)

    return move_and_value_tree
