from .node_factory import TreeNodeFactory
from .base import Base
def create_node_factory(node_factory_name: str) -> TreeNodeFactory:
    tree_node_factory: TreeNodeFactory

    match node_factory_name:
        case 'Base':
            tree_node_factory = Base()
        case other:
            raise f'please implement your node factory!! no {other}'

    return tree_node_factory
