"""
stopping criterion
"""

from .node_selector import NodeSelector
from .trees import MoveAndValueTree


class StoppingCriterion:
    """
    The general stopping criterion
    """

    def should_we_continue(self, tree: MoveAndValueTree) -> bool:
        """
        Asking should we continue

        Returns:
            boolean of should we continue
        """
        if tree.root_node.is_over():
            return False
        return True


class TreeMoveLimit(StoppingCriterion):
    """
    The stopping criterion based on a tree move limit
    """

    tree_move_limit: int

    def __init__(self, tree_move_limit: int):
        self.tree_move_limit = tree_move_limit

    def should_we_continue(self, tree: MoveAndValueTree):
        continue_base = super().should_we_continue(tree=tree)
        if not continue_base:
            return continue_base
        return tree.move_count < self.tree_move_limit

    def get_string_of_progress(self, tree: MoveAndValueTree) -> str:
        """
        compute the string that display the progress in the terminal
        Returns:
            a string that display the progress in the terminal
        """
        return f'========= tree move counting: {tree.move_count} out of {self.tree_move_limit}' \
               f' |  {tree.move_count / self.tree_move_limit:.0%}'


class DepthLimit(StoppingCriterion):
    """
    The stopping criterion based on a depth limit
    """
    depth_limit: int
    node_selector: NodeSelector

    def __init__(self, depth_limit: int,
                 node_selector: NodeSelector):
        self.depth_limit = depth_limit
        self.node_selector = node_selector

    def should_we_continue(self, tree: MoveAndValueTree):
        continue_base = super().should_we_continue(tree=tree)
        if not continue_base:
            return continue_base
        return self.node_selector.current_depth_to_expand < self.depth_limit

    def get_string_of_progress(self):
        """
        compute the string that display the progress in the terminal
        Returns:
            a string that display the progress in the terminal
        """
        return '========= tree move counting: ' + str(self.tree.move_count) + ' | Depth: ' + str(
            self.node_selector.current_depth_to_expand) + ' out of ' + str(self.depth_limit)


def create_stopping_criterion(arg: dict,
                              node_selector:NodeSelector) -> StoppingCriterion:
    """
    creating the stopping criterion
    Args:
        arg: arguments

    Returns:
        A stopping criterion

    """
    stopping_criterion: StoppingCriterion

    match arg['name']:
        case 'depth_limit':
            stopping_criterion = DepthLimit(depth_limit=arg['depth_limit'], node_selector=node_selector)
        case 'tree_move_limit':
            stopping_criterion = TreeMoveLimit(tree_move_limit=arg['tree_move_limit'])
        case other:
            raise Exception(f'stopping criterion builder: can not find {other}')

    return stopping_criterion
