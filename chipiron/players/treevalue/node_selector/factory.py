from . import NodeSelector
from .uniform import Uniform
from .recur_zipf_base import RecurZipfBase


def create(arg: dict,
           opening_instructor,
           random_generator
           ) -> NodeSelector:
    node_move_opening_selector: NodeSelector

    tree_builder_type: str = arg['type']

    match tree_builder_type:
        case 'Uniform':
            node_move_opening_selector = Uniform(arg,
                                                 opening_instructor=opening_instructor)
        case 'RecurZipfBase':
            node_move_opening_selector = RecurZipfBase(arg=arg,
                                                       random_generator=random_generator,
                                                       opening_instructor=opening_instructor)
        case other:
            raise Exception('tree builder: can not find ' + other)

    return node_move_opening_selector
