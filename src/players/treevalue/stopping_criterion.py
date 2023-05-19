"""
stopping criterion
"""

from .node_selector.node_selector import NodeSelector
from .trees.imove_and_value_tree import ITreeBuilder


class StoppingCriterion:
    """
    The general stopping criterion
    """
    node_selector: NodeSelector

    def should_we_continue(self, tree: ITreeBuilder) -> bool:
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

    def __init__(self,
                 tree_move_limit: int):
        self.tree_move_limit = tree_move_limit


    def should_we_continue(self, tree):
        continue_base = super().should_we_continue(tree=tree)
        if not continue_base:
            return continue_base
        return tree.move_count < self.tree_move_limit

    def get_string_of_progress(self) -> str:
        """
        compute the string that display the progress in the terminal
        Returns:
            a string that display the progress in the terminal
        """
        return f'========= tree move counting: {self.player.tree.move_count} out of {self.tree_move_limit}' \
               f' |  {self.player.tree.move_count / self.tree_move_limit:.0%}'


class DepthLimit(StoppingCriterion):
    """
    The stopping criterion based on a depth limit
    """
    node_selector: NodeSelector

    def __init__(self, depth_limit: int, node_selector: NodeSelector):
        self.depth_limit: int = depth_limit
        self.node_selector: NodeSelector = node_selector

    def should_we_continue(self):
        continue_base: bool = super().should_we_continue()
        if not continue_base:
            return continue_base
        return self.node_selector.current_depth_to_expand < self.depth_limit

    def get_string_of_progress(self):
        """
        compute the string that display the progress in the terminal
        Returns:
            a string that display the progress in the terminal
        """
        return '========= tree move counting: ' + str(self.player.tree.move_count) + ' | Depth: ' + str(
            self.player.current_depth_to_expand) + ' out of ' + str(self.depth_limit)


def create_stopping_criterion(node_selector: NodeSelector,
                              arg: dict) -> StoppingCriterion:
    """
    creating the stopping criterion
    Args:
        node_selector: the node_selector
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
