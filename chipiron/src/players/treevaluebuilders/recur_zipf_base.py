import time

from src.players.treevaluebuilders.trees.move_and_value_tree import MoveAndValueTree
from src.players.treevaluebuilders.tree_and_value_player import TreeAndValuePlayer
from src.players.treevaluebuilders.trees.nodes.tree_node_with_values import TreeNodeWithValue
from src.players.treevaluebuilders.trees.nodes.tree_node_with_descendants import NodeWithDescendants
from src.players.treevaluebuilders.move_explorer import ZipfMoveExplorer
import chess
import random


# todo merge with the other recurzipf it seems we need to have otomatic tree node initiation

class RecurZipfBaseTree(MoveAndValueTree):
    def create_tree_node(self, board, half_move, count, father_node):
        board_depth = half_move - self.tree_root_half_move
        if board_depth == 0:
            return NodeWithDescendants(board, half_move, count, father_node)
        else:
            return TreeNodeWithValue(board, half_move, count, father_node)

    def update_after_node_creation(self, node, parent_node):
        node_depth = node.half_move - self.tree_root_half_move
        if node_depth >= 1:
            self.root_node.descendants.add_descendant(node)


class RecurZipfBase(TreeAndValuePlayer):

    def __init__(self, arg):
        super().__init__(arg)
        self.move_explorer = ZipfMoveExplorer(arg['move_explorer_priority'])

    def create_tree(self, board):
        return RecurZipfBaseTree(self.environment, self.board_evaluators_wrapper, board)

    def choose_node_and_move_to_open(self):
        # todo maybe proportions and proportions can be valuesorted dict with smart updates


        if self.tree.root_node.best_node_sequence:
            last_node_in_best_line = self.tree.root_node.best_node_sequence[-1]
            if last_node_in_best_line.board.is_attacked(not last_node_in_best_line.player_to_move) and not last_node_in_best_line.is_over():
                # print('best line is underattacked')
                if random.random() > .5:
                    return self.opening_instructor.instructions_to_open_all_moves(last_node_in_best_line)

        wandering_node = self.tree.root_node

        while wandering_node.children_not_over:
            assert (not wandering_node.is_over())
            #print('I am at node', wandering_node.id)

            wandering_node = self.move_explorer.sample_child_to_explore(tree_node_to_sample_from=wandering_node)

       # print('I choose to open node', wandering_node.id, wandering_node.is_over())
        opening_instructions = self.opening_instructor.instructions_to_open_all_moves(wandering_node)
        return opening_instructions

    def print_info(self):
        super().print_info()
        print('RecurZipfBase')
