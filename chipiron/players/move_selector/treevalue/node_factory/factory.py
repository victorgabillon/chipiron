from .base import Base


def create_node_factory(
        node_factory_name: str
) -> Base:
    tree_node_factory: Base

    match node_factory_name:
        case 'Base':
            tree_node_factory = Base()
        case other:
            raise ValueError(f'please implement your node factory!! no {other} in {__name__}')

    return tree_node_factory
