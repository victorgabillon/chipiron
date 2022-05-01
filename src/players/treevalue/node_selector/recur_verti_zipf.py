from src.players.treevalue.trees.move_and_value_tree import MoveAndValueTree
from src.players.treevalue.tree_and_value_player import TreeAndValuePlayer
from src.players.treevalue.node_selector.notations_and_statics import zipf_picks
from src.players.treevalue.nodes.tree_node_with_proportions import ProportionsNode


class RecurVertiZipfTree(MoveAndValueTree):
    def create_tree_node(self, board, half_move, count, father_node):
        return ProportionsNode(board, half_move, count, father_node, self.zipf_style)


class RecurVertiZipf(TreeAndValuePlayer):

    def __init__(self, arg):
        super().__init__(arg)

    def create_tree(self, board):
        return RecurVertiZipfTree(self.environment, self.board_evaluator, self.color_player, self.arg, board)

    def choose_node_and_move_to_open(self):
        # print('----------------')
        node = self.tree.root_node

        if node.moves_children == {}:
            return self.opening_instructor.instructions_to_open_all_moves(node)

        node = node.choose_child_with_visits_and_proportions()

        while True:
            # print('ssss----------------')
            # choose a depth a which to test an alternative with zipf
            node = self.choose_node_in_best_line(node)

            if node.moves_children == {}:
                assert (not node.is_over())
                assert (node.moves_children == {})
                break
            elif len(node.children_not_over) == 1:
                # assert (node.best_child() in node.children_not_over)
                node = next(iter(node.children_not_over))
            else:
                # choose an alternative to the best line
                node = node.choose_child_with_visits_and_proportions(children_exception_set={node.best_child()})
        return self.opening_instructor.instructions_to_open_all_moves(node)

    def choose_node_in_best_line(self, node):
        best_line = node.best_node_sequence_filtered_from_over()
        best_k, best_value = zipf_picks(list_elements=best_line,
                                        value_of_element=lambda index: self.visits_depth[
                                            index] if index in self.visits_depth else 1)

        self.visits_depth[best_k] = self.visits_depth[best_k] + 1 if best_k in self.visits_depth else 1

        chosen_node = best_line[best_k]
        assert (not chosen_node.is_over())
        return chosen_node

    def get_move_from_player(self, board):

        print(board.chess_board)
        self.visits_depth = {0: 1}

        return super().get_move_from_player(board)

    def print_info(self):
        super().print_info()
        print('Recur+VertiZipf')
