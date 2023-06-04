from chipiron.players.treevalue.nodes.tree_node_with_values import TreeNodeWithValue
from chipiron.players.treevalue.nodes.tree_node_with_descendants import NodeWithDescendants
from .. import trees
from chipiron.players.treevalue.node_selector.opening_instructions import OpeningInstructions, OpeningInstructor


class UniformTree:  # todo probably this is overkill as the tree s nothing special or does it?
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


class Uniform:
    """ The Uniform Node selector """

    opening_instructor: OpeningInstructor

    def __init__(self,
                 arg: dict,
                 opening_instructor: OpeningInstructor):
        self.opening_instructor = opening_instructor
        self.current_depth_to_expand = 0

    def choose_node_and_move_to_open(self, tree: trees.MoveAndValueTree) -> OpeningInstructions:

        if tree.nodes_count <= 1:
            self.current_depth_to_expand = 0

        opening_instructions_batch = OpeningInstructions()

        # generate the nodes to expand
        current_half_move_to_expand = tree.tree_root_half_move + self.current_depth_to_expand

        # self.tree.descendants.print_info()
        # print('self.root_half_move ',self.root_half_move,self.current_depth_to_expand)
        nodes_to_consider = list(tree.descendants[current_half_move_to_expand].values())

        # filter the game-over ones
        nodes_to_consider = [node for node in nodes_to_consider if not node.is_over()]

        # sort them by order of importance for the player
        nodes_to_consider_sorted_by_value = sorted(nodes_to_consider,
                                                   key=lambda x: tree.root_node.subjective_value_of(
                                                       x))  # best last

        for node in nodes_to_consider_sorted_by_value:
            new_opening_instructions = self.opening_instructor.instructions_to_open_all_moves(node)
            opening_instructions_batch.merge(new_opening_instructions)

        self.current_depth_to_expand += 1
        return opening_instructions_batch

    def print_info(self):
        super().print_info()
        print('Uniform')
