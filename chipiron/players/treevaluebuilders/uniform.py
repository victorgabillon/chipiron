from players.treevaluebuilders.tree_and_value import TreeAndValue
from players.treevaluebuilders.trees.move_and_value_tree import MoveAndValueTree
from sortedcollections import ValueSortedDict
from players.treevaluebuilders.trees.opening_instructions import OpeningInstructionsBatch


class Uniform(TreeAndValue):

    def __init__(self, arg):
        super().__init__(arg)
        self.max_depth = int(arg['depth_tree']) if 'depth_tree' in arg else None
        self.current_depth_to_expand = None

    def continue_exploring(self):
        bool_one = super().continue_exploring()
        bool_two = True
        if self.max_depth is not None:
            bool_two = self.max_depth > self.current_depth_to_expand
        return bool_one and bool_two

    def choose_node_and_move_to_open(self):
        opening_instructions_batch = OpeningInstructionsBatch()

        # generate the nodes to expand
        nodes_to_consider = list(self.tree.all_nodes[self.current_depth_to_expand].values())
        # sort them by order of importance for the player
        if self.current_depth_to_expand == 0:
            nodes_to_consider_sorted_by_value = nodes_to_consider
        else:
            nodes_to_consider_sorted_by_value = sorted(nodes_to_consider,
                                                       key=lambda x: self.tree.root_node.subjective_value_of(x)) #best last

        for node in nodes_to_consider_sorted_by_value:
            new_opening_instructions = self.opening_instructor.instructions_to_open_all_moves(node)
            opening_instructions_batch.merge(new_opening_instructions)

        self.current_depth_to_expand += 1
        return opening_instructions_batch

    def create_tree(self, board):
        return MoveAndValueTree(self.environment, self.board_evaluator, self.color_player, self.arg, board)

    def get_move_from_player(self, board, timetoMove):
        self.current_depth_to_expand = 0
        return super().get_move_from_player(board, timetoMove)

    def print_info(self):
        super().print_info()
        print('Uniform, depth = ', self.max_depth, 'self.tree_move_limit :', self.tree_move_limit)
