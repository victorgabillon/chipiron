from src.players.treevalue.tree_and_value_player import TreeAndValuePlayer
from src.players.treevalue.trees.move_and_value_tree import MoveAndValueTree
from src.players.treevalue.nodes.index_tree_node import IndexTreeNode
from src.players.treevalue.node_selector.notations_and_statics import zipf_picks
import math



class SequoolTree2(MoveAndValueTree):

    def __init__(self, environment, board_evaluator, color, arg, board):
        self.all_nodes_2_not_opened = []
        #   self.count_to_open_at_depth=[1]
        self.count_visits_at_depth = []
        super().__init__(environment, board_evaluator, color, arg, board)
        self.root_node.index = 0

    def create_tree_node(self, board, half_move, count, father_node):
        return IndexTreeNode(board, half_move, count, father_node)

    def add_to_dic(self, depth, fen, node):  #todo to be changed!! to descendants no?
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
                    child.index = None  # rootnode is not set to zero haha

        root_node_value_white = self.root_node.value_white
        root_node_second_value_white = self.root_node.second_best_child().value_white

        for depth in range(self.get_max_depth()):
            # print('depth',depth)
            for node in self.all_nodes[depth].values():
                #   print('node',node.id)
                for child in node.moves_children.values():
                    #      print('child', child.id)
                    if node.index is None:
                        index = None
                    else:
                        if depth % 2 == 0:
                            if self.root_node.best_child() in child.first_moves:  # todo what if it is inboth at the sam time
                                if child == node.best_child():
                                    index = abs(child.value_white - root_node_second_value_white) / 2
                                else:
                                    index = None
                            else:
                                index = abs(child.value_white - root_node_value_white) / 2
                        else:  # depth %2 ==1
                            if self.root_node.best_child() in child.first_moves:
                                index = abs(child.value_white - root_node_second_value_white) / 2
                            else:  # not the best line
                                if child == node.best_child():
                                    index = abs(child.value_white - root_node_value_white) / 2
                                else:  # not the best child response
                                    index = None
                        #print('@@@', child.id, index, child.index)
                    if index is not None:
                        if child.index is None:  # if the index has beene initiated already by another parent node
                            child.index = index
                            # print(']]]',child.id, self.root_node.best_node_sequence[-1].id)
                            if child.id == self.root_node.best_node_sequence[-1].id:
                                assert (self.root_node.best_node_sequence[-1].index is not None)

                        else:
                            child.index = min(child.index, index)
                            if child.id == self.root_node.best_node_sequence[-1].id:
                                assert (self.root_node.best_node_sequence[-1].index is not None)
                    #print('===', child.id, index, child.index)

#        print('fggggggg', self.root_node.best_node_sequence[-1].id, self.root_node.best_node_sequence[-1].index)

        assert (self.root_node.best_node_sequence[-1].index is not None)
        # assert (self.get_max_depth() == len(self.all_nodes))
        # for depth in range(self.get_max_depth()):
        #     # print('eee',self.get_max_depth())
        #     for node in self.all_nodes[depth].values():
        #         for child in node.moves_children.values():
        #             assert (child.index is not None)

    #   self.print_some_stats()


class Sequool2(TreeAndValuePlayer):

    def __init__(self, arg):
        super().__init__(arg)

    def create_tree(self, board):
        return SequoolTree2(self.environment, self.board_evaluator, self.color_player, self.arg, board)

    def is_there_smth_to_open(self, depth):
        res = False
        for node in self.tree.all_nodes_2_not_opened[depth]:
            if node.index is not None:
                res = True
                break
        return res

    def choose_node_and_move_to_open(self):
        if self.tree.root_node.are_all_moves_and_children_opened():
 #           print('fg')
            self.tree.update_all_indices()

        # todo remove the depths that are fully explored
     #   self.tree.save_raw_data_to_file(self.count)
      #  self.count += 1
  #      print('###', self.count)

        depth_picked, best_value = zipf_picks(list_elements=self.tree.all_nodes_2_not_opened,
                                              value_of_element=lambda depth_: self.tree.count_visits_at_depth[
                                                  depth_] if self.is_there_smth_to_open(depth_) else math.inf)
   #     if self.tree.root_node.best_node_sequence:
    #        print('jklj', self.tree.root_node.best_node_sequence[-1].depth)
     #       print('jklj', self.tree.root_node.best_node_sequence[-1].index)
      #      print('jklj', self.tree.root_node.best_node_sequence[-1].are_all_moves_and_children_opened())

       #     dd = self.tree.root_node.best_node_sequence[-1].depth
        #    print('....', self.is_there_smth_to_open(dd))

        #print('depthpicked', depth_picked)
        #print([self.is_there_smth_to_open(d) for d, i in enumerate(self.tree.all_nodes_2_not_opened)])
        #print(self.tree.root_node.is_over())

        self.tree.count_visits_at_depth[depth_picked] += 1
        # print('visit_at_depth', self.tree.count_visits_at_depth)
        # for i in self.tree.all_nodes_2_not_opened:
        # print('tr', len(i))

        nodes_to_consider = list(self.tree.all_nodes_2_not_opened[depth_picked])

        nodes_to_consider = []
        for depth in range(depth_picked + 1):
            nodes_to_consider += list(self.tree.all_nodes_2_not_opened[depth])

        # for i, el in enumerate(self.tree.all_nodes_2_not_opened[depth_picked]):
        #     print('#',i)
        #     print('[]//', el)
        #print('###]][]', len(nodes_to_consider))
        best_node = nodes_to_consider[0]
        best_value = (best_node.index, best_node.half_move)
        for node in nodes_to_consider:
            # print('~',node.index , best_value,node.id,node.depth)

            if node.index is not None:
                if best_node.index is None or (node.index, node.half_move) < best_value:
                    best_node = node
                    best_value = (node.index, node.half_move)

        # print('best_value',best_value)
        # print('best_node',best_node.id)

        self.tree.all_nodes_2_not_opened[best_node.half_move].remove(best_node)
        #   self.tree.count_to_open_at_depth=[node.depth] -=1
        assert (best_node.index is not None)
        return self.opening_instructor.instructions_to_open_all_moves(best_node)

    def get_move_from_player(self, board, timetoMove):
        print(board.chess_board)
        return super().get_move_from_player(board, timetoMove)

    def print_info(self):
        super().print_info()
        print('Sequool2')
