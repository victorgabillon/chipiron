from src.players.treevaluebuilders.trees.move_and_value_tree import MoveAndValueTree
from src.players.treevaluebuilders.node_factory.RecurZipfBase import RecurZipfBase as RecurZipfBaseNodeFactory


def create_move_and_value_tree(args,
                               board_evaluators_wrapper,
                               board) -> MoveAndValueTree:

    tree_builder_type = args['type']

    if tree_builder_type == 'RecurZipfBase':
        node_factory = RecurZipfBaseNodeFactory()
    else:
        raise('please implement your node factory!!')

    move_and_value_tree: MoveAndValueTree = MoveAndValueTree(board_evaluator=board_evaluators_wrapper,
                                                             starting_board=board,
                                                             node_factory=node_factory,
                                                             updater=None
                                                             )

    return move_and_value_tree
