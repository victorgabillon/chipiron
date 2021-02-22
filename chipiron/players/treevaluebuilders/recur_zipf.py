from players.treevaluebuilders.trees.move_and_value_tree import MoveAndValueTree
from players.treevaluebuilders.tree_and_value import TreeAndValue
from players.treevaluebuilders.trees.nodes.proportion_tree_node import VisitsAndProportionsNode


class RecurZipfTree(MoveAndValueTree):
    def create_tree_node(self, board, half_move, count, father_node):
        return VisitsAndProportionsNode(board, half_move, count, father_node, self.arg['zipf_style'])


class RecurZipf(TreeAndValue):

    def __init__(self, arg):
        super().__init__(arg)

    def create_tree(self,board):
        return RecurZipfTree(self.environment, self.board_evaluator, self.color_player, self.arg, board)

    def choose_node_and_move_to_open(self):
        # todo maybe proportions and proportions can be valuesorted dict with smart updates

        node = self.tree.root_node

        while node.children_not_over:
            assert (not node.is_over())
            # print('I am at node',node.id)

            node = node.choose_child_with_visits_and_proportions()

        # print('I choose to open node',node.id,last_move)
        return self.opening_instructor.instructions_to_open_all_moves(node)

    def print_info(self):
        super().print_info()
        print('RecurZipf')
