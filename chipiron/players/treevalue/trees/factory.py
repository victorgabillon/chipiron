import chipiron as ch

from chipiron.players.treevalue.trees.move_and_value_tree import MoveAndValueTree
from chipiron.players.treevalue.node_factory.RecurZipfBase import RecurZipfBase as RecurZipfBaseNodeFactory
from chipiron.players.treevalue.node_selector.node_selector import NodeSelector
from chipiron.players.treevalue.node_factory.factory import TreeNodeFactory
from typing import List


def create_tree_manager(args: dict,
                        board_evaluators_wrapper,
                        board: ch.chess.IBoard,
                        expander_subscribers: List[NodeSelector]) -> MoveAndValueTree:
    tree_builder_type: str = args['type']

    node_factory: TreeNodeFactory
    match tree_builder_type:
        case 'RecurZipfBase':
            node_factory = RecurZipfBaseNodeFactory()
        case other:
            raise 'please implement your node factory!! no {}'.format(other)

    move_and_value_tree: MoveAndValueTree = MoveAndValueTree(board_evaluator=board_evaluators_wrapper,
                                                             starting_board=board,
                                                             node_factory=node_factory)

    return move_and_value_tree
