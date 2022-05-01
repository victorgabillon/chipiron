from src.players.treevalue.node_selector.move_explorer import ZipfMoveExplorer
from src.players.treevalue.node_selector.node_selector import NodeSelector
from src.players.treevalue.trees.opening_instructions import OpeningInstructionsBatch


# todo merge with the other recurzipf it seems we need to have automatic tree node initiation


class RecurZipfBase(NodeSelector):

    def __init__(self, arg :dict, random_generator, opening_instructor):
        self.opening_instructor = opening_instructor
        self.move_explorer = ZipfMoveExplorer(arg['move_explorer_priority'], random_generator)
        self.random_generator = random_generator

    def choose_node_and_move_to_open(self)-> OpeningInstructionsBatch:
        # todo maybe proportions and proportions can be valuesorted dict with smart updates

        if self.tree.root_node.best_node_sequence:
            last_node_in_best_line = self.tree.root_node.best_node_sequence[-1]
            if last_node_in_best_line.board.is_attacked(
                    not last_node_in_best_line.player_to_move) and not last_node_in_best_line.is_over():
                # print('best line is underattacked')
                if self.random_generator.random() > .5:
                    # print('best line is underattacked and i do')
                    return self.opening_instructor.instructions_to_open_all_moves(last_node_in_best_line)

        wandering_node = self.tree.root_node

        while wandering_node.children_not_over:
            assert (not wandering_node.is_over())
            wandering_node = self.move_explorer.sample_child_to_explore(tree_node_to_sample_from=wandering_node)

        # print('I choose to open node', wandering_node.id, wandering_node.is_over())
        opening_instructions = self.opening_instructor.instructions_to_open_all_moves(wandering_node)
        return opening_instructions

    def update_after_node_creation(self, node, parent_node):
        node_depth = node.half_move - self.tree_root_half_move
        if node_depth >= 1:
            self.root_node.descendants.add_descendant(node)

    def print_info(self):
        super().print_info()
        print('RecurZipfBase')
