from src.players.treevalue.trees.move_and_value_tree import MoveAndValueTreeBuilder
from src.players.treevalue.node_factory.RecurZipfBase import RecurZipfBase as RecurZipfBaseNodeFactory
from src.players.treevalue.node_factory.factory import TreeNodeFactory


def create_move_and_value_tree_builder(args: dict) -> MoveAndValueTreeBuilder:
    tree_builder_type: str = args['type']

    node_factory: TreeNodeFactory
    match tree_builder_type:
        case 'RecurZipfBase':
            node_factory = RecurZipfBaseNodeFactory()
        case other:
            raise 'please implement your node factory!! no {}'.format(other)

    tree_builder: MoveAndValueTreeBuilder = MoveAndValueTreeBuilder(node_factory=node_factory)

    return tree_builder
