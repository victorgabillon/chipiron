from chipiron.players.treevalue.node_selector.move_explorer import ZipfMoveExplorer
from chipiron.players.treevalue.node_selector.opening_instructions import OpeningInstructionsBatch, OpeningInstructor
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

    def choose_node_and_move_to_open(self, tree: trees.MoveAndValueTree) -> OpeningInstructionsBatch:
        # todo maybe proportions and proportions can be valuesorted dict with smart updates

        if tree.root_node.best_node_sequence:
            last_node_in_best_line = tree.root_node.best_node_sequence[-1]
            if last_node_in_best_line.board.is_attacked(
                    not last_node_in_best_line.player_to_move) and not last_node_in_best_line.is_over():
                # print('best line is underattacked')
                if self.random_generator.random() > .5:
                    # print('best line is underattacked and i do')
                    return self.opening_instructor.instructions_to_open_all_moves(last_node_in_best_line)

        wandering_node = tree.root_node

        while wandering_node.children_not_over:
            assert (not wandering_node.is_over())
            wandering_node = self.move_explorer.sample_child_to_explore(tree_node_to_sample_from=wandering_node)

        opening_instructions: OpeningInstructionsBatch = self.opening_instructor.instructions_to_open_all_moves(
            wandering_node)
        return opening_instructions

    def __str__(self):
        return 'RecurZipfBase'
