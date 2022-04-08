

from src.players.treevaluebuilders.move_explorer import ZipfMoveExplorer



# todo merge with the other recurzipf it seems we need to have otomatic tree node initiation



class RecurZipfBase:

    def __init__(self, arg, random_generator):
        self.move_explorer = ZipfMoveExplorer(arg['move_explorer_priority'], random_generator)
        self.random_generator = random_generator

    def create_tree(self, board):
        return RecurZipfBaseTree(self.board_evaluators_wrapper, board)

    def choose_node_and_move_to_open(self):
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
            #  print('I am at node', wandering_node.id)
            # wandering_node.print_a_move_sequence_from_root()
            # wandering_node.print_children_sorted_by_value()

            wandering_node = self.move_explorer.sample_child_to_explore(tree_node_to_sample_from=wandering_node)

        # print('I choose to open node', wandering_node.id, wandering_node.is_over())
        opening_instructions = self.opening_instructor.instructions_to_open_all_moves(wandering_node)
        return opening_instructions

    def print_info(self):
        super().print_info()
        print('RecurZipfBase')
