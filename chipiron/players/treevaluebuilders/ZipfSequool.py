from players.treevaluebuilders.tree_and_value import TreeAndValue
from players.treevaluebuilders.Trees.move_and_value_tree import MoveAndValueTree
from players.treevaluebuilders.Trees.Nodes.index_tree_node import IndexTreeNode
from players.treevaluebuilders.notations_and_statics import zipf_picks
from players.treevaluebuilders.Trees.Nodes.proportion_tree_node import VisitsAndProportionsNode
from players.treevaluebuilders.Trees.descendants import SortedDescendants
from players.treevaluebuilders.Trees.Nodes.tree_node_with_descendants import NodeWithDescendantsNoUpdate
from players.treevaluebuilders.Trees.first_moves import FirstMoves
import math


class ZipfSequoolTreeNode(VisitsAndProportionsNode, IndexTreeNode):
    pass


class IndexDescendantsTreeNode(NodeWithDescendantsNoUpdate, IndexTreeNode):
    pass


class ZipfSequoolTree(MoveAndValueTree):

    def __init__(self, environment, board_evaluator, arg, board):
        super().__init__(environment, board_evaluator, arg, board)
        self.count_visits_at_depth = []
        self.first_moves = FirstMoves()

    def create_tree_node(self, board, half_move, count, father_node):
        if half_move == self.tree_root_half_move:
            new_node = ZipfSequoolTreeNode(board, half_move, count, father_node, self.arg['zipf_style'])
        elif half_move == self.tree_root_half_move + 1:
            new_node = IndexDescendantsTreeNode(board, half_move, count, father_node)
        else:
            new_node = IndexTreeNode(board, half_move, count, father_node)
        node_depth = half_move - self.tree_root_half_move
        if node_depth > 1:
            self.descendants.add_descendant(new_node)
        return new_node

    def update_after_either_node_or_link_creation(self, node, parent_node):
        pass

    def update_after_link_creation(self, node, parent_node):
        #  print('#sdsdsd')
        previous_first_move = self.first_moves[node].copy()

        self.first_moves.add_first_move(node, parent_node)
        new_first_moves = self.first_moves[node].difference(previous_first_move)
        if new_first_moves:
            node_descendants = node.get_descendants()
        for new_first_move in new_first_moves:
            for descendant in node_descendants:
                if not new_first_move.descendants.contains_node(descendant):
                    new_first_move.descendants.add_descendant(descendant)
            node_descendants_not_opened = node.get_not_opened_descendants()
            for descendant_not_opened in node_descendants_not_opened:
                if not new_first_move.descendants_not_opened.contains_node(descendant_not_opened):
                    index = descendant_not_opened.index if descendant_not_opened.index is not None else math.inf
                    value = (index, descendant_not_opened.half_move, descendant_not_opened.id)
                    new_first_move.descendants_not_opened.add_descendant(descendant_not_opened, value)

        # if self.root_node is not None:
        #     first_moves = self.root_node.moves_children.values()
        #     for f in first_moves:
        #         f.descendants.test_2(f)

    def update_after_node_creation(self, node, parent_node):
        # print('############',node.id,node.half_move,parent_node)
        # if parent_node is not None:
        #   print('<>',parent_node.id)
        #  for i in self.first_moves[parent_node]:
        #     print('[]',i.id)
        self.first_moves.add_first_move(node, parent_node)

        root_node_half_move = self.tree_root_half_move
        if node.half_move - root_node_half_move - 1 == len(self.count_visits_at_depth):
            self.count_visits_at_depth.append(1)

        node_depth = node.half_move - self.tree_root_half_move

        if node_depth == 1:
            node.descendants_not_opened = SortedDescendants()

        if node_depth >= 1:
            for first_move in self.first_moves[node]:
                #    print(';',node.id,node.half_move)
                first_move.descendants_not_opened.add_descendant(node, (math.inf, node.half_move, node.id))

        if node_depth >= 2:  # todo can we do this strategy without the descendnats at each first child?
            for first_move in self.first_moves[node]:
                first_move.descendants.add_descendant(node)

        # if self.root_node is not None:
        #     first_moves = self.root_node.moves_children.values()
        #     for f in first_moves:
        #         f.descendants.test_2(f)

    def update_all_indices(self):
        for half_move in self.descendants:
            for node in self.descendants[half_move].values():
                for child in node.moves_children.values():
                    child.index = None  # rootnode is not set to zero haha

        self.root_node.index = 0
        for half_move in self.descendants:
            for node in self.descendants[half_move].values():
                for child in node.moves_children.values():
                    index = max(abs(child.value_white - node.value_white) / 2, node.index)
                    if child.index is None:
                        child.index = index
                    else:
                        child.index = min(child.index, index)
                    for fm in self.first_moves[child]:
                        if not child.all_legal_moves_generated:
                            fm.descendants_not_opened.update_value(child, (child.index, child.half_move, child.id))

        for half_move in self.descendants:
            # print('eee',self.get_max_depth())
            for node in self.descendants[half_move].values():
                for child in node.moves_children.values():
                    assert (child.index is not None)

    #   self.print_some_stats()


class ZipfSequool(TreeAndValue):

    def __init__(self, arg):
        super().__init__(arg)

    def create_tree(self, board):
        return ZipfSequoolTree(self.environment, self.board_evaluators_wrapper, self.arg, board)

    def who_is_there_smth_to_open(self, node, half_move):
        if half_move not in node.descendants_not_opened:
            #  print('~~')
            return set()
        else:
            # print('~s~', type(node.descendants_not_opened),
            #      type(node.descendants_not_opened.sorted_descendants_at_half_move[half_move]))
            return node.descendants_not_opened.sorted_descendants_at_half_move[half_move]

    def candidates_at_depth(self, node, depth_):
        root_node_half_move = self.tree.root_node.half_move
        return self.who_is_there_smth_to_open(node, 1 + root_node_half_move + depth_)

    def choose_node_and_move_to_open(self):
        # self.tree.update_all_indices()

        self.count_refresh_index += 1
        if self.tree.root_node.are_all_moves_and_children_opened():
            print('fg', self.count_refresh_index,self.next_update)
            if self.count_refresh_index < 10 or self.count_refresh_index == self.next_update:
                if self.count_refresh_index == self.next_update:
                     self.next_update = 2*self.next_update +1
                print('fgeeee', self.count_refresh_index)

                self.tree.update_all_indices()

        node = self.tree.root_node
        if node.moves_children == {}:
            return self.opening_instructor.instructions_to_open_all_moves(node)

        node_first_move = node.choose_child_with_visits_and_proportions()

        def func(depth_):
            candidates = len(self.candidates_at_depth(node_first_move, depth_))
            if candidates:
                return self.tree.count_visits_at_depth[depth_]
            else:
                return math.inf

        depth_picked, best_value = zipf_picks(list_elements=list(node_first_move.descendants.keys()),
                                              value_of_element=func)
        #print('depthpicked', depth_picked)
        #print('no_id', node_first_move.id)
        #  node_first_move.descendants.print_info()
        # node_first_move.descendants_not_opened.print_info()

        #print('values', [func(i) for i in range(len(list(node_first_move.descendants.keys())))])

        self.tree.count_visits_at_depth[depth_picked] += 1
        #print('####', type(self.candidates_at_depth(node_first_move, depth_picked)))

        nodes_to_consider_dict = self.candidates_at_depth(node_first_move, depth_picked)
        nodes_to_consider_list = list(nodes_to_consider_dict.keys())  # todo loosing time in the list conversion?

        #print('####', type(nodes_to_consider_list))
        # for i in nodes_to_consider:
        #    print('nodes', i.id)
        # todo chack if mixing depth improves results

        best_node = nodes_to_consider_list[0]
        #print('op', type(best_node), best_node, type(nodes_to_consider_list[0]))
        index = best_node.index if best_node.index is not None else math.inf
        best_value = (index, best_node.half_move, best_node.id)
        for node_to_consider in nodes_to_consider_list:
            index = node_to_consider.index if node_to_consider.index is not None else math.inf
          #  print('~',  node_to_consider.index, (index, node_to_consider.half_move, node_to_consider.id) , best_value)

            if (index, node_to_consider.half_move, node_to_consider.id) < best_value:
                best_node = node_to_consider
                best_value = (index, node_to_consider.half_move, node_to_consider.id)

        # print('best_value',best_value)
        best_node_2 = min(nodes_to_consider_dict, key=nodes_to_consider_dict.get)  # next(iter(nodes_to_consider_dict))
        # print('~', nodes_to_consider_dict)
       # print('best_node', best_node.id, best_node.index)
        assert (best_node == best_node_2)
        assert (not best_node.all_legal_moves_generated)
        #  self.tree.descendants.print_info()
        for nfm in self.tree.first_moves[best_node]:
            nfm.descendants_not_opened.remove_descendant(best_node)

        return self.opening_instructor.instructions_to_open_all_moves(best_node)

    def get_move_from_player(self, board, timetoMove):
        print(board.chess_board)
        self.count_refresh_index = 0
        self.next_update =10

        return super().get_move_from_player(board, timetoMove)

    def print_info(self):
        super().print_info()
        print('ZipfSequool')
