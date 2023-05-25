from chipiron.players.treevalue.tree_manager.tree_manager import TreeManager
from chipiron.players.treevalue.tree_manager.tree_expander import TreeExpander
from chipiron.players.treevalue import node_factory


def create_tree_manager(args: dict,
                        board_evaluators_wrapper) -> TreeManager:
    node_factory_name: str = args['node_factory_name'] if 'node_factory_name' in args else 'Base'

    tree_node_factory: node_factory.TreeNodeFactory
    match node_factory_name:
        case 'Base':
            tree_node_factory = node_factory.Base()
        case other:
            raise f'please implement your node factory!! no {other}'

    tree_expander: TreeExpander = TreeExpander(node_factory=tree_node_factory,
                                               board_evaluator=board_evaluators_wrapper)

    tree_manager: TreeManager = TreeManager(tree_expander=tree_expander)

    return tree_manager
