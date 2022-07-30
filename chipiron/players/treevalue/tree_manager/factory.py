import chipiron as ch
from typing import List
from chipiron.players.treevalue.tree_manager.tree_manager import TreeManager
from chipiron.players.treevalue.tree_manager.tree_expander import TreeExpander
import chipiron.players.treevalue.node_selector as nodeselector
from chipiron.players.treevalue.trees.move_and_value_tree import MoveAndValueTree
from chipiron.players.treevalue import node_factory


def create_tree_manager(args: dict,
                        board_evaluators_wrapper,
                        board: ch.chess.IBoard,
                        expander_subscribers: List[nodeselector.NodeSelector]) -> TreeManager:
    tree_builder_type: str = args['type']

    tree_node_factory: node_factory.TreeNodeFactory
    match tree_builder_type:
        case 'RecurZipfBase':
            tree_node_factory = node_factory.RecurZipfBase()
        case other:
            raise 'please implement your node factory!! no {}'.format(other)

    move_and_value_tree: MoveAndValueTree = MoveAndValueTree(board_evaluator=board_evaluators_wrapper,
                                                             starting_board=board)

    tree_expander: TreeExpander = TreeExpander(tree=move_and_value_tree, node_factory=tree_node_factory)

    tree_manager: TreeManager = TreeManager(tree=move_and_value_tree, tree_expander=tree_expander)

    return tree_manager
