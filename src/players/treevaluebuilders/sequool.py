from src.players.treevaluebuilders.tree_and_value_player import TreeAndValuePlayer
from src.players.treevaluebuilders.trees.move_and_value_tree import MoveAndValueTree
from src.players.treevaluebuilders.trees.nodes.index_tree_node import IndexTreeNode
from src.players.treevaluebuilders.notations_and_statics import zipf_picks
import math


class SequoolTree(MoveAndValueTree):

    def __init__(self, environment, board_evaluator, color, arg, board):
        self.all_nodes_2_not_opened = []
        self.count_visits_at_depth = []
        super().__init__(environment, board_evaluator, color, arg, board)
        self.root_node.index = 0

    def create_tree_node(self, board, half_move, count, father_node):
        return IndexTreeNode(board, half_move, count, father_node)

    def add_to_dic(self, depth, fen, node): #todo to be changed!! to descendants no?
        super().add_to_dic(depth, fen, node)
        if depth < len(self.all_nodes_2_not_opened):
            self.all_nodes_2_not_opened[depth].add(node)
        #    self.count_to_open_at_depth[node.depth] += 1
        else:  # depth == len(self.all_nodes_2_not_opened)
            assert (depth == len(self.all_nodes_2_not_opened))
            self.all_nodes_2_not_opened.append({node})
            if depth == len(self.count_visits_at_depth):
                self.count_visits_at_depth.append(1)

        #   self.count_to_open_at_depth.append(1)

    def update_all_indices(self):
      #  print('hhhhhhhhhhhhhhhhhhhhhhhhh')
        for depth in range(self.get_max_depth()):
            for node in self.all_nodes[depth].values():
                for child in node.moves_children.values():
                    child.index = None

        for depth in range(self.get_max_depth()):
            #print('depth',depth)
            for node in self.all_nodes[depth].values():
             #   print('node',node.id)
                for child in node.moves_children.values():
              #      print('child', child.id)

                    index = max(abs(child.value_white - node.value_white) / 2, node.index)
       #             print('----', index)

                    if child.index is None:
                        child.index = index
         #               print('#',child.index)
                    else:
        #                print('mm', child.index,index)
                        child.index = min(child.index, index)
          #              print('dmm', child.index)

        assert(self.get_max_depth() == len(self.all_nodes))
        for depth in range(self.get_max_depth()):
            #print('eee',self.get_max_depth())
            for node in self.all_nodes[depth].values():
                for child in node.moves_children.values():
                    assert(child.index is not None)

     #   self.print_some_stats()

class Sequool(TreeAndValuePlayer):

    def __init__(self, arg):
        super().__init__(arg)

    def create_tree(self, board):
        return SequoolTree(self.environment, self.board_evaluator, self.color_player, self.arg, board)

    def choose_node_and_move_to_open(self):
        self.tree.update_all_indices()

        # todo remove the depths that are fully explored

        depth_picked, best_value = zipf_picks(list_elements=self.tree.all_nodes_2_not_opened,
                                              value_of_element=lambda depth: self.tree.count_visits_at_depth[
                                                  depth] if self.tree.all_nodes_2_not_opened[depth] else math.inf)
        #print('depthpicked', depth_picked)

        self.tree.count_visits_at_depth[depth_picked] += 1
        #print('visit_at_depth', self.tree.count_visits_at_depth)
        #for i in self.tree.all_nodes_2_not_opened:
         # print('tr', len(i))

        nodes_to_consider = list(self.tree.all_nodes_2_not_opened[depth_picked])

        nodes_to_consider = []
        for depth in range(depth_picked+1):
            nodes_to_consider += list(self.tree.all_nodes_2_not_opened[depth])

        # for i, el in enumerate(self.tree.all_nodes_2_not_opened[depth_picked]):
        #     print('#',i)
        #     print('[]//', el)

        best_node = nodes_to_consider[0]
        best_value = (best_node.index, best_node.half_move)
        for node in nodes_to_consider:
          #  print('~',node.index , best_value,node.id,node.depth)

            if (node.index , node.half_move) < best_value:
                best_node = node
                best_value = (node.index , node.half_move)

        #print('best_value',best_value)
        #print('best_node',best_node.id)

        self.tree.all_nodes_2_not_opened[best_node.half_move].remove(best_node)
        #   self.tree.count_to_open_at_depth=[node.depth] -=1
        return self.opening_instructor.instructions_to_open_all_moves(best_node)

    def get_move_from_player(self, board, timetoMove):
        print(board.chess_board)
        return super().get_move_from_player(board, timetoMove)

    def print_info(self):
        super().print_info()
        print('Sequool')
