"""
stopping criterion
"""

from . import tree_manager as tree_man


class StoppingCriterion:
    """
    The general stopping criterion
    """

    def should_we_continue(self, tree_manager: tree_man.TreeManager) -> bool:
        """
        Asking should we continue

        Returns:
            boolean of should we continue
        """
        if tree_manager.tree.root_node.is_over():
            return False
        return True


class TreeMoveLimit(StoppingCriterion):
    """
    The stopping criterion based on a tree move limit
    """

    def __init__(self, tree_move_limit):
        self.tree_move_limit = tree_move_limit

    def should_we_continue(self, tree_manager: tree_man.TreeManager):
        continue_base = super().should_we_continue(tree_manager)
        if not continue_base:
            return continue_base
        return tree_manager.tree.move_count < self.tree_move_limit

    def get_string_of_progress(self, tree_manager: tree_man.TreeManager) -> str:
        """
        compute the string that display the progress in the terminal
        Returns:
            a string that display the progress in the terminal
        """
        return f'========= tree move counting: {tree_manager.tree.move_count} out of {self.tree_move_limit}' \
               f' |  {tree_manager.tree.move_count / self.tree_move_limit:.0%}'


class DepthLimit(StoppingCriterion):
    """
    The stopping criterion based on a depth limit
    """

    def __init__(self, depth_limit):
        self.depth_limit = depth_limit

    def should_we_continue(self, tree_manager: tree_man.TreeManager):
        continue_base = super().should_we_continue()
        if not continue_base:
            return continue_base
        return self.player.current_depth_to_expand < self.depth_limit

    def get_string_of_progress(self):
        """
        compute the string that display the progress in the terminal
        Returns:
            a string that display the progress in the terminal
        """
        return '========= tree move counting: ' + str(self.player.tree.move_count) + ' | Depth: ' + str(
            self.player.current_depth_to_expand) + ' out of ' + str(self.depth_limit)


def create_stopping_criterion(arg: dict) -> StoppingCriterion:
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
            stopping_criterion = DepthLimit(depth_limit=arg['depth_limit'])
        case 'tree_move_limit':
            stopping_criterion = TreeMoveLimit(tree_move_limit=arg['tree_move_limit'])
        case other:
            raise Exception(f'stopping criterion builder: can not find {other}')

    return stopping_criterion
