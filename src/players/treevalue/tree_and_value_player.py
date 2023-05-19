import chess
from src.players.treevalue.node_selector.notations_and_statics import softmax
from src.players.boardevaluators.over_event import OverEvent
from src.players.treevalue.node_selector.node_selector import NodeSelector
from src.chessenvironment.board.iboard import IBoard
from src.players.treevalue.stopping_criterion import StoppingCriterion
from .trees.move_and_value_tree import MoveAndValueTree, MoveAndValueTreeBuilder
from src.players.treevalue.trees.opening_instructions import OpeningInstructionsBatch


class TreeAndValuePlayer:
    # at the moment it looks like i do not need this class i could go directly
    # for the tree builder no? think later bout that? maybe one is for multi round and the other is not?

    node_move_opening_selector: NodeSelector
    arg: dict
    stopping_criterion: StoppingCriterion
    tree_builder: MoveAndValueTreeBuilder

    def __init__(self,
                 arg: dict,
                 random_generator,
                 node_move_opening_selector: NodeSelector,
                 stopping_criterion: StoppingCriterion,
                 board_evaluators_wrapper,
                 tree_builder: MoveAndValueTreeBuilder):
        self.node_move_opening_selector = node_move_opening_selector
        self.board_evaluators_wrapper = board_evaluators_wrapper
        self.arg: dict = arg
        self.stopping_criterion = stopping_criterion
        self.random_generator = random_generator
        self.tree_builder = tree_builder

    def continue_exploring(self, tree):
        return self.stopping_criterion.should_we_continue(tree=tree)

    def print_info_during_move_computation(self,
                                           tree):
        if tree.root_node.best_node_sequence:
            current_best_child = tree.root_node.best_node_sequence[0]
            current_best_move = tree.root_node.moves_children.inverse[current_best_child]
            assert (tree.root_node.get_value_white() == current_best_child.get_value_white())

        else:
            current_best_move = '?'
        if self.random_generator.random() < .05:
            str_progress = self.stopping_criterion.get_string_of_progress()
            print(str_progress,
                  '| current best move:', current_best_move, '| current white value:',
                  tree.root_node.value_white_minmax)  # ,end='\r')
            tree.root_node.print_children_sorted_by_value_and_exploration()
            tree.print_best_line()

    def recommend_move_after_exploration(self):
        # todo the preference for action that have been explored more is not super clear, is it weel implemented, ven for debug?

        # for debug we fix the choice in the next lines
        # if global_variables.deterministic_behavior:
        #     print(' FIXED CHOICE FOR DEBUG')
        #     best_child = self.tree.root_node.get_all_of_the_best_moves(how_equal='considered_equal')[-1]
        #     print('We have as best: ', self.tree.root_node.moves_children.inverse[best_child])
        #     best_move = self.tree.root_node.moves_children.inverse[best_child]

        if True:  # normal behavior
            selection_rule = self.arg['move_selection_rule']['type']
            if selection_rule == 'softmax':
                temperature = self.arg['move_selection_rule']['temperature']
                values = [self.tree.root_node.subjective_value_of(node) for node in
                          self.tree.root_node.moves_children.values()]

                softmax_ = softmax(values, temperature)
                print(values)
                print('SOFTMAX', temperature, [i / sum(softmax_) for i in softmax_],
                      sum([i / sum(softmax_) for i in softmax_]))

                move_as_list = self.random_generator.choices(list(self.tree.root_node.moves_children.keys()),
                                                             weights=softmax_, k=1)
                best_move = move_as_list[0]
            elif selection_rule == 'almost_equal' or selection_rule == 'almost_equal_logistic':
                # find the best first move allowing for random choice for almost equally valued moves.
                best_root_children = self.tree.root_node.get_all_of_the_best_moves(how_equal=selection_rule)
                print('We have as bests: ',
                      [self.tree.root_node.moves_children.inverse[best] for best in best_root_children])
                best_child = self.random_generator.choice(best_root_children)
                if self.tree.root_node.over_event.how_over == OverEvent.WIN:
                    assert (best_child.over_event.how_over == OverEvent.WIN)
                best_move = self.tree.root_node.moves_children.inverse[best_child]
            else:
                raise (Exception('move_selection_rule is not valid it seems'))
        return best_move

    def explore(self, board: IBoard):

        tree = MoveAndValueTree(board_evaluator=self.board_evaluators_wrapper,
                                starting_board=board)

        self.tree_builder.add_root_node(tree=tree, board=board)

        while self.continue_exploring(tree=tree):
            assert (not tree.root_node.is_over())
            self.print_info_during_move_computation(tree=tree)

            opening_instructions_batch: OpeningInstructionsBatch \
                = self.node_move_opening_selector.choose_node_and_move_to_open(tree=tree)

            if self.arg['stopping_criterion']['name'] == 'tree_move_limit':
                tree_move_limit = self.arg['stopping_criterion']['tree_move_limit']
                opening_instructions_subset = opening_instructions_batch.pop_items(
                    tree_move_limit - tree.move_count)
            else:
                opening_instructions_subset = opening_instructions_batch

            self.tree_builder.open_and_update(tree=tree,
                                              opening_instructions_batch=opening_instructions_subset,
                                              board_evaluator=self.board_evaluators_wrapper)

        #  self.tree.save_raw_data_to_file()
        tree.print_some_stats()
        for move, child in tree.root_node.moves_children.items():
            print(move, tree.root_node.moves_children[move].get_value_white(), child.over_event.get_over_tag())
        print('evaluation for white: ', tree.root_node.get_value_white())

    def select_move(self, board: IBoard) -> chess.Move:
        self.explore(board)
        best_move: chess.Move = self.recommend_move_after_exploration()
        self.tree.print_best_line()  # todo maybe almost best chosen line no?

        return best_move

    def print_info(self):
        super().print_info()
        print('type: Tree and Value')
