import chipiron as ch
from typing import List
from chipiron.players.treevalue.tree_manager.tree_manager import TreeManager
from chipiron.players.treevalue.tree_manager.tree_expander import TreeExpander, TreeExpansionHistory
import chipiron.players.treevalue.node_selector as nodeselector
from chipiron.players.treevalue.trees.move_and_value_tree import MoveAndValueTree
from chipiron.players.treevalue import node_factory


def create_tree_manager(args: dict,
                        board_evaluators_wrapper,
                        board: ch.chess.BoardChi,
                        expander_subscribers: List[nodeselector.NodeSelector]) -> TreeManager:
    node_factory_name: str = args['node_factory_name'] if 'node_factory_name' in args else 'Base'

    tree_node_factory: node_factory.TreeNodeFactory
    match node_factory_name:
        case 'Base':
            tree_node_factory = node_factory.Base()
        case other:
            raise f'please implement your node factory!! no {other}'

    root_node = node_factory.create_root_node(board, board.ply(), board_evaluators_wrapper)
    tree_expansion_history = TreeExpansionHistory(root_node=root_node)

    move_and_value_tree: MoveAndValueTree = MoveAndValueTree(root_node=root_node)

    tree_expander: TreeExpander = TreeExpander(tree=move_and_value_tree,
                                               node_factory=tree_node_factory,
                                               tree_expansion_history=tree_expansion_history,
                                               board_evaluator=board_evaluators_wrapper)

    tree_manager: TreeManager = TreeManager(tree=move_and_value_tree, tree_expander=tree_expander)

    return tree_manager
