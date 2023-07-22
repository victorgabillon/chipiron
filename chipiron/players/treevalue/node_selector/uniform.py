from .. import trees
from chipiron.players.treevalue.node_selector.opening_instructions import OpeningInstructions, OpeningInstructor, \
    create_instructions_to_open_all_moves
from .. import tree_manager as tree_man


class Uniform:
    """ The Uniform Node selector """

    opening_instructor: OpeningInstructor

    def __init__(
            self,
            opening_instructor: OpeningInstructor
    ) -> None:
        self.opening_instructor = opening_instructor
        self.current_depth_to_expand = 0

    def choose_node_and_move_to_open(
            self,
            tree: trees.MoveAndValueTree,
            latest_tree_expansions: tree_man.TreeExpansions

    ) -> OpeningInstructions:
        opening_instructions_batch: OpeningInstructions = OpeningInstructions()

        # generate the nodes to expand
        current_half_move_to_expand = tree.tree_root_half_move + self.current_depth_to_expand

        # self.tree.descendants.print_info()
        nodes_to_consider = list(tree.descendants[current_half_move_to_expand].values())

        # filter the game-over ones
        nodes_to_consider = [node for node in nodes_to_consider if not node.is_over()]

        # sort them by order of importance for the player
        nodes_to_consider_sorted_by_value = sorted(nodes_to_consider,
                                                   key=lambda x: tree.root_node.minmax_evaluation.subjective_value_of(
                                                       x))  # best last

        for node in nodes_to_consider_sorted_by_value:
            all_moves_to_open = self.opening_instructor.all_moves_to_open(node_to_open=node.tree_node)
            opening_instructions: OpeningInstructions = create_instructions_to_open_all_moves(
                moves_to_play=all_moves_to_open,
                node_to_open=node)
            opening_instructions_batch.merge(opening_instructions)

        self.current_depth_to_expand += 1
        return opening_instructions_batch

    def print_info(self):
        super().print_info()
        print('Uniform')
