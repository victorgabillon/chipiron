import chipiron as ch
from chipiron.players.treevalue.stopping_criterion import StoppingCriterion
from chipiron.players.treevalue.node_selector.notations_and_statics import softmax
from chipiron.players.boardevaluators.over_event import OverEvent

from . import tree_manager as tree_man
from . import node_selector
from . import trees


class TreeAndValuePlayer:
    # at the moment it looks like i do not need this class i could go directly
    # for the tree builder no? think later bout that? maybe one is for multi round and the other is not?

    node_move_opening_selector: node_selector.NodeSelector
    stopping_criterion: StoppingCriterion

    def __init__(self,
                 arg: dict,
                 random_generator,
                 node_move_opening_selector: node_selector.NodeSelector,
                 board_evaluators_wrapper,
                 stopping_criterion: StoppingCriterion):
        self.node_move_opening_selector = node_move_opening_selector
        self.board_evaluators_wrapper = board_evaluators_wrapper
        self.arg = arg
        self.tree_move_limit = arg['tree_move_limit'] if 'tree_move_limit' in arg else None
        self.stopping_criterion = stopping_criterion
        self.random_generator = random_generator

    def print_info_during_move_computation(self, tree_manager: tree_man.TreeManager):
        if tree_manager.tree.root_node.best_node_sequence:
            current_best_child = tree_manager.tree.root_node.best_node_sequence[0]
            current_best_move = tree_manager.tree.root_node.moves_children.inverse[current_best_child]
            assert (tree_manager.tree.root_node.get_value_white() == current_best_child.get_value_white())

        else:
            current_best_move = '?'
        if self.random_generator.random() < .05:
            str_progress = self.stopping_criterion.get_string_of_progress(tree_manager)
            print(str_progress,
                  '| current best move:', current_best_move, '| current white value:',
                  tree_manager.tree.root_node.value_white_minmax)  # ,end='\r')
            tree_manager.tree.root_node.print_children_sorted_by_value_and_exploration()
            tree_manager.print_best_line()

    def recommend_move_after_exploration(self, tree: trees.MoveAndValueTree):
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
                values = [tree.root_node.subjective_value_of(node) for node in
                          tree.root_node.moves_children.values()]

                softmax_ = softmax(values, temperature)
                print(values)
                print('SOFTMAX', temperature, [i / sum(softmax_) for i in softmax_],
                      sum([i / sum(softmax_) for i in softmax_]))

                move_as_list = self.random_generator.choices(
                    list(tree.root_node.moves_children.keys()),
                    weights=softmax_, k=1)
                best_move = move_as_list[0]
            elif selection_rule == 'almost_equal' or selection_rule == 'almost_equal_logistic':
                # find the best first move allowing for random choice for almost equally valued moves.
                best_root_children = tree.root_node.get_all_of_the_best_moves(
                    how_equal=selection_rule)
                print('We have as bests: ',
                      [tree.root_node.moves_children.inverse[best] for best in best_root_children])
                best_child = self.random_generator.choice(best_root_children)
                if tree.root_node.over_event.how_over == OverEvent.WIN:
                    assert (best_child.over_event.how_over == OverEvent.WIN)
                best_move = tree.root_node.moves_children.inverse[best_child]
            else:
                raise (Exception('move_selection_rule is not valid it seems'))
        return best_move

    def explore(self, tree_manager: tree_man.TreeManager):

        while self.stopping_criterion.should_we_continue(tree_manager):
            assert (not tree_manager.tree.root_node.is_over())
            self.print_info_during_move_computation(tree_manager=tree_manager)

            opening_instructions_batch: node_selector.OpeningInstructionsBatch
            opening_instructions_batch = self.node_move_opening_selector.choose_node_and_move_to_open(tree_manager.tree)

            if self.arg['stopping_criterion']['name'] == 'tree_move_limit':
                tree_move_limit = self.arg['stopping_criterion']['tree_move_limit']
                opening_instructions_subset = opening_instructions_batch.pop_items(
                    tree_move_limit - tree_manager.tree.move_count)
            else:
                opening_instructions_subset = opening_instructions_batch

            tree_manager.open_and_update(opening_instructions_subset)

        if self.tree_move_limit is not None:
            assert self.tree_move_limit == tree_manager.tree.move_count or tree_manager.tree.root_node.is_over()
        #  self.tree.save_raw_data_to_file()
        tree_manager.print_some_stats()
        for move, child in tree_manager.tree.root_node.moves_children.items():
            print(move, tree_manager.tree.root_node.moves_children[move].get_value_white(),
                  child.over_event.get_over_tag())
        print('evaluation for white: ', tree_manager.tree.root_node.get_value_white())

    def select_move(self, board: ch.chess.IBoard):

        tree_manager: tree_man.TreeManager
        tree_manager = tree_man.create_tree_manager(args=self.arg,
                                                    board_evaluators_wrapper=self.board_evaluators_wrapper,
                                                    board=board,
                                                    expander_subscribers=[
                                                        self.node_move_opening_selector])

        self.explore(tree_manager)
        best_move = self.recommend_move_after_exploration(tree_manager.tree)
        tree_manager.print_best_line()  # todo maybe almost best chosen line no?

        return best_move

    def print_info(self):
        super().print_info()
        print('type: Tree and Value')
