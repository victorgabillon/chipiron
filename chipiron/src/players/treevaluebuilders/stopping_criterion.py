import sys


def create_stopping_criterion(player, arg, ):
    stopping_criterion_type = arg['name']
    if stopping_criterion_type == 'depth_limit':
        stopping_criterion = DepthLimitCriterion(arg['depth_limit'])
    elif stopping_criterion_type == 'tree_move_limit':
        stopping_criterion = TreeMoveLimitCriterion(arg['tree_move_limit'])
    else:
        sys.exit('stopping criterion builder: can not find ' + arg['tree_builder']['type'])
    stopping_criterion.player = player
    return stopping_criterion


class StoppingCriterion:
    def should_we_continue(self):
        if self.player.tree.root_node.is_over():
            return False
        else:
            return True


class TreeMoveLimitCriterion(StoppingCriterion):
    def __init__(self, tree_move_limit):
        self.tree_move_limit = tree_move_limit

    def should_we_continue(self):
        continue_base = super().should_we_continue()
        if not continue_base:
            return continue_base
        return self.player.tree.move_count < self.tree_move_limit

    def get_string_of_progress(self):
        return '========= tree move counting: ' + str(self.player.tree.move_count) + ' out of ' + str(
            self.tree_move_limit) + ' | ' + "{0:.0%}".format(self.player.tree.move_count / self.tree_move_limit)


class DepthLimitCriterion(StoppingCriterion):
    def __init__(self, depth_limit):
        self.depth_limit = depth_limit

    def should_we_continue(self):
        continue_base = super().should_we_continue()
        if not continue_base:
            return continue_base
        return self.player.current_depth_to_expand < self.depth_limit

    def get_string_of_progress(self):
        return '========= tree move counting: ' + str(self.player.tree.move_count) + ' | Depth: ' + str(
            self.player.current_depth_to_expand) + ' out of ' + str(self.depth_limit)
