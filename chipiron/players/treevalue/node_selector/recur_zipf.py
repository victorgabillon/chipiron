from chipiron.players.treevalue.trees.move_and_value_tree import MoveAndValueTree
from chipiron.players.treevalue.tree_and_value_player import TreeAndValuePlayer
from chipiron.players.treevalue.nodes.tree_node_with_proportions import ProportionsNode
from chipiron.players.treevalue.node_selector.move_explorer import VisitProportionMoveExplorer


class RecurZipfTree(MoveAndValueTree):
    def create_tree_node(self, board, half_move, count, father_node):
        return ProportionsNode(board, half_move, count, father_node)


class RecurZipf(TreeAndValuePlayer):

    def __init__(self, arg):
        super().__init__(arg)
        self.move_explorer = VisitProportionMoveExplorer(arg['move_explorer_priority'])

    def create_tree(self, board):
        return RecurZipfTree(self.environment, self.board_evaluators_wrapper,  board)

    def choose_node_and_move_to_open(self):
        # todo maybe proportions and proportions can be valuesorted dict with smart updates

        wandering_node = self.tree.root_node

        while wandering_node.children_not_over:
            assert (not wandering_node.is_over())
            # print('I am at node',node.id)

            wandering_node = self.move_explorer.sample_child_to_explore(tree_node_to_sample_from=wandering_node)

        # print('I choose to open node',node.id,last_move)
        return self.opening_instructor.instructions_to_open_all_moves(wandering_node)

    def print_info(self):
        super().print_info()
        print('RecurZipf')
