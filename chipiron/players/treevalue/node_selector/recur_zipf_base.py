from chipiron.players.treevalue.node_selector.move_explorer import ZipfMoveExplorer
from chipiron.players.treevalue.node_selector.opening_instructions import OpeningInstructions, OpeningInstructor, create_instructions_to_open_all_moves
from .. import trees


# todo merge with the other recurzipf it seems we need to have automatic tree node initiation


class RecurZipfBase:
    """ The RecurZipfBase Node selector """
    opening_instructor: OpeningInstructor

    def __init__(self,
                 arg: dict,
                 random_generator,
                 opening_instructor: OpeningInstructor):
        self.opening_instructor = opening_instructor
        self.move_explorer = ZipfMoveExplorer(arg['move_explorer_priority'], random_generator)
        self.random_generator = random_generator

    def choose_node_and_move_to_open(self, tree: trees.MoveAndValueTree) -> OpeningInstructions:
        # todo maybe proportions and proportions can be valuesorted dict with smart updates

        if tree.root_node.minmax_evaluation.best_node_sequence:
            last_node_in_best_line = tree.root_node.minmax_evaluation.best_node_sequence[-1]
            if last_node_in_best_line.board.is_attacked(
                    not last_node_in_best_line.tree_node.player_to_move) and not last_node_in_best_line.minmax_evaluation.is_over():
                # print('best line is underattacked')
                if self.random_generator.random() > .5:
                    # print('best line is underattacked and i do')
                    all_moves_to_open = self.opening_instructor.all_moves_to_open(node_to_open=last_node_in_best_line.tree_node)
                    opening_instructions: OpeningInstructions = create_instructions_to_open_all_moves(
                        moves_to_play=all_moves_to_open,
                        node_to_open=last_node_in_best_line)
                    return opening_instructions

        wandering_node = tree.root_node

        while wandering_node.minmax_evaluation.children_not_over:
            assert (not wandering_node.is_over())
            wandering_node = self.move_explorer.sample_child_to_explore(tree_node_to_sample_from=wandering_node)

        all_moves_to_open = self.opening_instructor.all_moves_to_open(node_to_open=wandering_node.tree_node)
        opening_instructions: OpeningInstructions = create_instructions_to_open_all_moves(moves_to_play=all_moves_to_open,
                                                                                          node_to_open=wandering_node)

        return opening_instructions

    def __str__(self):
        return 'RecurZipfBase'
