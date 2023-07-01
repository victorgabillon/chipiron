from chipiron.extra_tools.small_tools import softmax
from chipiron.players.boardevaluators.over_event import OverEvent
from chipiron.players.treevalue.trees.factory import MoveAndValueTreeFactory
from . import trees
from . import tree_manager as tree_man
from chipiron.players.treevalue.tree_manager.tree_expander import TreeExpansions
from chipiron.players.treevalue.node_selector.opening_instructions import OpeningInstructor
from chipiron.players.treevalue.stopping_criterion import StoppingCriterion, create_stopping_criterion
from . import node_selector as node_sel



class TreeExploration:
    tree: trees.MoveAndValueTree
    tree_manager: tree_man.AlgorithmNodeTreeManager
    node_selector: node_sel.NodeSelector
    args: dict

    def __init__(
            self,
            tree: trees.MoveAndValueTree,
            tree_manager: tree_man.AlgorithmNodeTreeManager,
            stopping_criterion: StoppingCriterion,
            node_selector: node_sel.NodeSelector,
            random_generator,
            args: dict):
        self.args = args
        self.tree = tree
        self.tree_manager = tree_manager
        self.stopping_criterion = stopping_criterion
        self.node_selector = node_selector
        self.random_generator = random_generator

    def print_info_during_move_computation(self):
        if self.tree.root_node.minmax_evaluation.best_node_sequence:
            current_best_child = self.tree.root_node.minmax_evaluation.best_node_sequence[0]
            current_best_move = self.tree.root_node.moves_children.inverse[current_best_child]
            assert (
                    self.tree.root_node.minmax_evaluation.get_value_white() == current_best_child.minmax_evaluation.get_value_white())

        else:
            current_best_move = '?'
        if self.random_generator.random() < 5:
            str_progress = self.stopping_criterion.get_string_of_progress(self.tree)
            print(str_progress,
                  '| current best move:', current_best_move, '| current white value:',
                  self.tree.root_node.minmax_evaluation.value_white_minmax)  # ,end='\r')
            self.tree.root_node.minmax_evaluation.print_children_sorted_by_value_and_exploration()
            self.tree_manager.print_best_line(tree=self.tree)

    def recommend_move_after_exploration(
            self,
            tree: trees.MoveAndValueTree):
        # todo the preference for action that have been explored more is not super clear, is it weel implemented, ven for debug?

        # for debug we fix the choice in the next lines
        # if global_variables.deterministic_behavior:
        #     print(' FIXED CHOICE FOR DEBUG')
        #     best_child = self.tree.root_node.get_all_of_the_best_moves(how_equal='considered_equal')[-1]
        #     print('We have as best: ', self.tree.root_node.moves_children.inverse[best_child])
        #     best_move = self.tree.root_node.moves_children.inverse[best_child]

        if True:  # normal behavior
            selection_rule = self.args['move_selection_rule']['type']
            if selection_rule == 'softmax':
                temperature = self.args['move_selection_rule']['temperature']
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
                best_root_children = tree.root_node.minmax_evaluation.get_all_of_the_best_moves(
                    how_equal=selection_rule)
                print('We have as bests: ',
                      [tree.root_node.moves_children.inverse[best] for best in best_root_children])
                best_child = self.random_generator.choice(best_root_children)
                if tree.root_node.minmax_evaluation.over_event.how_over == OverEvent.WIN:
                    assert (best_child.minmax_evaluation.over_event.how_over == OverEvent.WIN)
                best_move = tree.root_node.moves_children.inverse[best_child]
            else:
                raise (Exception('move_selection_rule is not valid it seems'))
        return best_move

    def explore(self):

        while self.stopping_criterion.should_we_continue(tree=self.tree):
            assert (not self.tree.root_node.is_over())
            # print info
            # self.print_info_during_move_computation()

            # choose the moves and nodes to open
            opening_instructions: node_sel.OpeningInstructions
            opening_instructions = self.node_selector.choose_node_and_move_to_open(self.tree)

            # make sure we do not break the stopping criterion
            opening_instructions_subset: node_sel.OpeningInstructions
            opening_instructions_subset = self.stopping_criterion.respectful_opening_instructions(
                opening_instructions=opening_instructions,
                tree=self.tree)
            # open the nodes
            tree_expansions: TreeExpansions = self.tree_manager.open(tree=self.tree,
                                                                     opening_instructions=opening_instructions_subset)
            # self.node_selector.communicate_expansions()
            self.tree_manager.update_backward(tree_expansions=tree_expansions)

        # trees.save_raw_data_to_file(tree=self.tree,
        #                            args=self.args)
        # self.tree_manager.print_some_stats(tree=self.tree)
        # for move, child in self.tree.root_node.moves_children.items():
        #    print(f'{move} {self.tree.root_node.moves_children[move].minmax_evaluation.get_value_white()}'
        #          f' {child.minmax_evaluation.over_event.get_over_tag()}')
        # print(f'evaluation for white: {self.tree.root_node.minmax_evaluation.get_value_white()}')

        best_move = self.recommend_move_after_exploration(self.tree)
        self.tree_manager.print_best_line(tree=self.tree)  # todo maybe almost best chosen line no?

        return best_move


def create_tree_exploration(
        args: dict,
        random_generator,
        board,
        tree_manager: tree_man.AlgorithmNodeTreeManager,
        tree_factory: MoveAndValueTreeFactory) -> TreeExploration:
    opening_instructor: OpeningInstructor \
        = OpeningInstructor(args['opening_type'], random_generator) if 'opening_type' in args else None

    move_and_value_tree: trees.MoveAndValueTree = tree_factory.create(board=board)

    node_selector: node_sel.NodeSelector = node_sel.create(
        arg=args,
        opening_instructor=opening_instructor,
        random_generator=random_generator)

    stopping_criterion: StoppingCriterion = create_stopping_criterion(arg=args['stopping_criterion'],
                                                                      node_selector=node_selector)

    tree_exploration: TreeExploration = TreeExploration(
        tree=move_and_value_tree,
        tree_manager=tree_manager,
        stopping_criterion=stopping_criterion,
        node_selector=node_selector,
        random_generator=random_generator,
        args=args
    )
    return tree_exploration
