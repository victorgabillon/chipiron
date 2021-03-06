from random import choice
from src.players.treevaluebuilders.trees.nodes.tree_node import TreeNode
from sortedcollections import ValueSortedDict, SortedList
import chess
from src.players.boardevaluators.over_event import OverEvent
from src.players.treevaluebuilders.trees.updates import UpdateInstructions, BaseUpdateInstructionsBlock
from src.players.treevaluebuilders.notations_and_statics import nth_key
import math
from src.extra_tools.my_value_sorted_dict import MyValueSortedDict
import pandas as pd


# todo maybe further split values from over?


class TreeNodeWithValue(TreeNode):

    def __init__(self, board, half_move, id_number, parent_node):
        super().__init__(board, half_move, id_number, parent_node)

        # absolute value wrt to white player as estimated by an evaluator
        self.value_white_evaluator = None

        # absolute value wrt to white player as computed from the value_white_* of the descendants
        # of this node (self) by a minmax procedure.
        self.value_white_minmax = None

        # self.best_move_sequence = []
        self.best_node_sequence = []

        # children_sorted_by_value records subjective values of children by descending order
        # subjective value means the values is from the point of view of player_to_move
        # careful, i have hard coded in the self.best_child() function the descending order for
        # fast access to best element, so please do not change!
        self.iii()

        # self.children_sorted_by_value = {}

        # convention of descending order, careful if changing read above!!
        self.best_index_for_value = 0

        # the list of children that have not yet be found to be over
        # using atm a list instaed of set as atm python set are not insertion ordered which adds randomness
        # and makes debug harder
        self.children_not_over = []

        # creating a base Over event that is set to None
        self.over_event = OverEvent()

    def iii(self):
        # self.children_sorted_by_value_vsd = ValueSortedDict({})
        self.children_sorted_by_value = MyValueSortedDict()

    def get_value_white(self):
        """ returns the best estimation of the value white in this node."""
        return self.value_white_minmax

    def set_evaluation(self, evaluation):
        """ sets the evaluation from the board evaluator"""
        self.value_white_evaluator = evaluation
        self.value_white_minmax = evaluation  # base value before knowing values of the children

    def subjective_value(self, value_white):
        """ return the value_white from the point of view of the self.player_to_move"""
        # subjective value is so that from the point of view of the self.player_to_move if value_1> value_2 then value_1
        # is preferable for self.player_to_move
        subjective_value = value_white if self.player_to_move == chess.WHITE else -value_white
        return subjective_value

    def subjective_value(self):
        """ return the self.value_white from the point of view of the self.player_to_move"""
        subjective_value = self.get_value_white() if self.player_to_move == chess.WHITE else -self.get_value_white()
        return subjective_value

    def subjective_value_of(self, another_node):
        # return the value from the point of view of the self.player_to_move of the value of another_node
        subjective_value = another_node.get_value_white() if self.player_to_move == chess.WHITE else -another_node.get_value_white()
        return subjective_value

    def best_child(self):
        # fast way to access first key with highest subjective value
        if self.children_sorted_by_value.dic:
            best_child = next(iter(self.children_sorted_by_value.dic))
        else:
            best_child = None
        return best_child

    def best_child_value(self):
        # fast way to access first key with highest subjective value
        if self.children_sorted_by_value:
            best_value = next(iter(self.children_sorted_by_value.dic.values()))
        else:
            best_value = None
        return best_value

    def second_best_child(self):
        assert (len(self.children_sorted_by_value) >= 2)
        # fast way to access second first key with highest subjective value
        second_best_child = nth_key(self.children_sorted_by_value, 1)
        return second_best_child

    def is_over(self):
        return self.over_event.is_over()

    def is_win(self):
        return self.over_event.is_win()

    def is_draw(self):
        return self.over_event.is_draw()

    def is_winner(self, player):
        return self.over_event.is_winner(player)

    def print_children_sorted_by_value(self):
        print('here are the ', len(self.children_sorted_by_value), ' children sorted by value: ')
        for child_node, subjective_sort_value in self.children_sorted_by_value.items():
            print(self.moves_children.inverse[child_node], subjective_sort_value[0], end=' $$ ')
        print('')

    def print_children_not_over(self):
        print('here are the ', len(self.children_not_over), ' children not over: ', end=' ')
        for child in self.children_not_over:
            print(child.id, end=' ')
        print(' ')

    def print_info(self):
        print('Soy el Node', self.id)
        self.print_moves_children()
        self.print_children_sorted_by_value()
        self.print_children_not_over()
        # todo probably more to print...

    def record_sort_value_of_child(self, child):
        """ stores the subjective value of the child in the self.children_sorted_by_value (automatically sorted)"""
        # - children_sorted_by_value records subjective value of children by descending order
        # therefore we have to convert the value_white of children into a subjective value that depends
        # on the player to move
        # - subjective best move/children is at index 0 however sortedValueDict are sorted ascending (best index: -1),
        # therefore for white we have negative values
        child_value_white = child.get_value_white()
        subjective_value_of_child = -child_value_white if self.player_to_move == chess.WHITE else child_value_white
        if self.is_over():
            # the shorter the check the better now
            self.children_sorted_by_value[child] = (subjective_value_of_child, -len(child.best_node_sequence), child.id)
        #   self.children_sorted_by_value_vsd[child] = (subjective_value_of_child, -len(child.best_node_sequence), child.id)

        else:
            # the longer the line the better now
            self.children_sorted_by_value[child] = (subjective_value_of_child, len(child.best_node_sequence), child.id)
        #  self.children_sorted_by_value_vsd[child] = (subjective_value_of_child, len(child.best_node_sequence), child.id)

    def are_equal_values(self, value_1, value_2):
        return value_1 == value_2

    def are_considered_equal_values(self, value_1, value_2):
        return value_1[:2] == value_2[:2]

    def are_almost_equal_values(self, value_1, value_2):
        # expect floats
        epsilon = 0.01
        return value_1 > value_2 - epsilon and value_2 > value_1 - epsilon

    def becoming_over_from_children(self):
        """ this nodes is asked to switch to over status"""
        assert (not self.is_over())

        # becoming over triggers a full update record_sort_value_of_child
        # where ties are now broken to reach over as fast as possible
        # todo we should reach it asap if we are winning and think about what to ddo in other scenarios....
        for child in self.moves_children.values():
            self.record_sort_value_of_child(child)

        # fast way to access first key with highest subjective value
        best_child = self.best_child()

        self.over_event.becomes_over(how_over=best_child.over_event.how_over,
                                     who_is_winner=best_child.over_event.who_is_winner)

    def update_over(self, children_with_updated_over):
        """ updating the over_event of the node based on notification of change of over_event in children"""

        is_newly_over = False
        # two cases can make this node (self) become over:
        # 1- one of the children of this node is over and is a win for the node.player_to_move: then do that!
        # 2- all children are now over then choose your best over event (choose draw if you can avoid a loss)

        for child in children_with_updated_over:
            assert child.is_over()
            if child in self.children_not_over:
                self.children_not_over.remove(child)
            # atm, it happens that child is already  not in children_not_over so we check,
            if not self.is_over() and child.is_winner(self.player_to_move):
                self.becoming_over_from_children()
                is_newly_over = True

        # check if all children are over but not winning for self.player_to_move
        if not self.is_over() and not self.children_not_over:
            self.becoming_over_from_children()
            is_newly_over = True

        return is_newly_over

    def update_children_values(self, children_nodes_to_consider):
        for child in children_nodes_to_consider:
            self.record_sort_value_of_child(child)
        self.children_sorted_by_value.sort_dic()
        # print('############',self.children_sorted_by_value.dic)
        # print('-###########',self.children_sorted_by_value_vsd)
        # for i in self.children_sorted_by_value_vsd:
        #     assert i

    def sort_children_not_over(self):
        # todo: looks like the deterministism of the sort induces some determinisin the play like always playing the same actions when a lot of them have equal value: introduce some randomness?

        return [child for child in self.children_sorted_by_value if
                child in self.children_not_over]  # todo is this a fast way to do it?

    def update_value_minmax(self):
        best_child = self.best_child()
        # not only best_child.value_white is considered. best_child.value_white is the best minmax values from
        # the children of self. In the case when not all the children of self have been evaluated for safety
        # reason we take a min or a max with self.value_white_evaluator. if all the children have been evaluated
        # we only use self.value_white_evaluator as the children evaluations are expected give a finer evaluation.
        if self.all_legal_moves_generated:
            self.value_white_minmax = best_child.get_value_white()
        elif self.player_to_move == chess.WHITE:
            self.value_white_minmax = max(best_child.get_value_white(), self.value_white_evaluator)
        else:
            self.value_white_minmax = min(best_child.gat_value_white(), self.value_white_evaluator)

    def update_best_move_sequence(self,
                                  children_nodes_with_updated_best_move_seq):
        """ triggered if a children notifies an updated best node sequence
        returns boolean: True if self.best_node_sequence is modified, False otherwise"""
        has_best_node_seq_changed = False
        best_node = self.best_node_sequence[0]

        if best_node in children_nodes_with_updated_best_move_seq:
            self.best_node_sequence = [best_node] + best_node.best_node_sequence
            has_best_node_seq_changed = True

        return has_best_node_seq_changed

    def one_of_best_children_becomes_best_next_node(self):
        """ triggered when the value of the current best move does not match the best value"""
        how_equal_ = 'equal'
        best_children = self.get_all_of_the_best_moves(how_equal=how_equal_)
        if how_equal_ == 'equal':
            assert (len(best_children) == 1)
        best_child = choice(best_children)
        # best_child = best_children[len(best_children) - 1]  # for debug!!

        self.best_node_sequence = [best_child] + best_child.best_node_sequence
        assert self.best_node_sequence

    def is_value_subjectively_better_than_evaluation(self, value_white):
        subjective_value = self.subjective_value(value_white)
        return subjective_value >= self.value_white_evaluator

    def minmax_value_update_from_children(self, children_with_updated_value):
        """ updates the values of children in self.sortedchildren
        updates value minmax and updates the best move"""

        # todo to be tested!!

        # updates value
        value_white_before_update = self.get_value_white()
        best_child_before_update = self.best_child()

        self.update_children_values(children_with_updated_value)
        self.update_value_minmax()
        value_white_after_update = self.get_value_white()
        has_value_changed = value_white_before_update != value_white_after_update

        # # updates best_move #todo maybe split in two functions but be careful one has to be done oft the other
        if best_child_before_update is None:
            best_child_before_update_not_the_best_anymore = True
        else:
            # here we compare the values in the self.children_sorted_by_value which might include more
            # than just the basic values #todo make that more clear at some point maybe even createing a value object
            updated_value_of_best_child_before_update = self.children_sorted_by_value[best_child_before_update]
            best_value_children_after = self.best_child_value()
            best_child_before_update_not_the_best_anymore = not self.are_equal_values(
                updated_value_of_best_child_before_update,
                best_value_children_after)

        best_node_seq_before_update = self.best_node_sequence.copy()
        if self.all_legal_moves_generated:
            if best_child_before_update_not_the_best_anymore:
                self.one_of_best_children_becomes_best_next_node()
        else:
            # we only consider a child as best if it is more promising than the evaluation of self
            # in self.value_white_evaluator
            if self.is_value_subjectively_better_than_evaluation(best_child_before_update.get_value_white()):
                self.one_of_best_children_becomes_best_next_node()
            else:
                self.best_node_sequence = []
        best_node_seq_after_update = self.best_node_sequence
        has_best_node_seq_changed = best_node_seq_before_update != best_node_seq_after_update

        return has_value_changed, has_best_node_seq_changed

    def dot_description(self):
        value_mm = "{:.3f}".format(self.value_white_minmax) if self.value_white_minmax is not None else 'None'
        value_eval = "{:.3f}".format(self.value_white_evaluator) if self.value_white_evaluator is not None else 'None'
        return super().dot_description() + '\n wh_val_mm: ' + value_mm + '\n wh_val_eval: ' \
               + value_eval + '\n moves*' + \
               self.description_best_move_sequence() + '\nover: ' + self.over_event.get_over_tag()

    def description_best_move_sequence(self):
        res = ''
        parent_node = self
        for child_node in self.best_node_sequence:
            move = parent_node.moves_children.inverse[child_node]
            parent_node = child_node
            res += '_' + str(move)
        return res

    def description_tree_visualizer_move(self, child):
        return ''

    def test(self):
        super().test()
        # print('testing node', self.id)
        self.test_values()
        self.test_over()
        self.test_children_not_over()
        self.test_best_node_sequence()

    def test_children_not_over(self):
        for move, child in self.moves_children.items():
            if child.is_over():
                assert (child not in self.children_not_over)
            else:
                assert (child in self.children_not_over)

    def test_over(self):
        if self.are_all_moves_and_children_opened() and self.children_not_over == set():
            assert (self.is_over())
        # todo assert its the good self.over

        self.over_event.test()

        for move, child in self.moves_children.items():
            if child.is_winner(self.player_to_move):
                assert (self.over_event.how_over == self.over_event.WIN)
                assert (self.over_event.who_is_winner == self.player_to_move)

        # todo test the contrary is its over is it the right over

    def test_values(self):
        # print('testvalues')
        # todo test valuewhiteminmax is none iif not all children opened
        value_children = []
        for move, child in self.moves_children.items():
            assert (self.children_sorted_by_value[child][0] * (1 - 2 * self.player_to_move) == child.get_value_white())
            value_children.append(child.get_value_white())
        if self.moves_children:
            if self.player_to_move == chess.WHITE:
                assert (max(value_children) == self.get_value_white())
            if self.player_to_move == chess.BLACK:
                assert (min(value_children) == self.get_value_white())
            for ind, child in enumerate(self.children_sorted_by_value):
                if ind == 0:
                    assert (child.get_value_white() == self.get_value_white())
                    before = child
                else:
                    if self.player_to_move == chess.WHITE:
                        assert (before.get_value_white() >= child.get_value_white())
                    if self.player_to_move == chess.BLACK:
                        assert (before.get_value_white() <= child.get_value_white())
        else:
            pass
            # todo test board value

    def test_best_node_sequence(self):
        # print ('testbestseq')
        # todo test if the sequence is empty, does it make sense? now yes and test if it is when the opened children
        #  ahve a value less promissing than the self.value_white_evaluator
        # todo test better for the weird value that are tuples!! with length and id
        if self.best_node_sequence:
            best_child = self.best_child()
            assert (self.best_node_sequence[0].get_value_white() ==
                    self.children_sorted_by_value[best_child][0] * (
                            1 - 2 * self.player_to_move))  # todo check the best index actually works=]

            old_best_node = self.best_node_sequence[0]
            assert (self.best_child() == self.best_node_sequence[0])
        for node in self.best_node_sequence[1:]:
            assert (isinstance(node, TreeNode))
            # print('@ss@',old_best_node, ':ddd' ,old_best_node.best_node_sequence[0].id,':ddd' ,node.id,':ddd' ,self.best_node_sequence)
            assert (old_best_node.best_node_sequence[0] == node)

            assert (old_best_node.best_node_sequence[0] == old_best_node.best_child())
            old_best_node = node

    def print_best_line(self):
        print('Best line from node ' + str(self.id) + ':', end=' ')
        parent_node = self
        for child in self.best_node_sequence:
            print(parent_node.moves_children.inverse[child], '(' + str(child.id) + ')', end=' ')
            parent_node = child
        print(' ')

    def my_logit(self, x):  # todo look out for numerical problem with utmatic rounding to 0 or especillay to 1
        y = min(max(x, .000000000000000000000001), .9999999999999999)
        return math.log(y / (1 - y))

    def get_all_of_the_best_moves(self, how_equal=None):
        # todo make it faster
        best_children = []
        best_child = self.best_child()
        best_value = self.children_sorted_by_value[best_child]
        for child in self.children_sorted_by_value:  # todo here faster...
            if how_equal == 'equal':
                if self.are_equal_values(self.children_sorted_by_value[child], best_value):
                    best_children.append(child)
                    assert (len(best_children) == 1)
            elif how_equal == 'considered_equal':
                if self.are_considered_equal_values(self.children_sorted_by_value[child], best_value):
                    best_children.append(child)
            elif how_equal == 'almost_equal':
                if self.are_almost_equal_values(self.children_sorted_by_value[child][0], best_value[0]):
                    best_children.append(child)
            elif how_equal == 'almost_equal_logistic':
                best_value_logit = self.my_logit(best_value[0] * .5 + .5)  # from [-1,1] to [0,1]
                child_value_logit = self.my_logit(self.children_sorted_by_value[child][0] * .5 + .5)
                if self.are_almost_equal_values(child_value_logit, best_value_logit):
                    best_children.append(child)
        return best_children

    def best_node_sequence_not_over(self):
        # todo investigate the case of having over in the best lines? does it make sense? what does it mean?
        res = [self]
        for best_child in self.best_node_sequence:
            if not best_child.is_over():
                res.append(best_child)
        return res

    def create_update_instructions_after_node_birth(self):
        update_instructions = UpdateInstructions()
        base_update_instructions_block = BaseUpdateInstructionsBlock(node_sending_update=self,
                                                                     is_node_newly_over=self.over_event.is_over(),
                                                                     new_value_for_node=True,
                                                                     new_best_move_for_node=False)
        update_instructions.all_instructions_blocks['base'] = base_update_instructions_block
        return update_instructions

    def perform_updates(self, updates_instructions):
        # get the base block
        updates_instructions_block = updates_instructions['base']  # todo create a variable for the tag

        # UPDATE VALUE
        has_value_changed, has_best_node_seq_changed_1 = self.minmax_value_update_from_children(
            updates_instructions_block['children_with_updated_value'])

        # UPDATE BEST MOVE
        has_best_node_seq_changed_2 = self.update_best_move_sequence(
            updates_instructions_block['children_with_updated_best_move'])
        # print('^&',has_best_node_seq_changed_2)
        has_best_node_seq_changed = has_best_node_seq_changed_1 or has_best_node_seq_changed_2

        # UPDATE OVER
        is_newly_over = self.update_over(updates_instructions_block['children_with_updated_over'])
        assert (is_newly_over is not None)

        # create the new instructions for the parents
        new_instructions = UpdateInstructions()
        base_update_instructions_block = BaseUpdateInstructionsBlock(node_sending_update=self,
                                                                     is_node_newly_over=is_newly_over,
                                                                     new_value_for_node=has_value_changed,
                                                                     new_best_move_for_node=has_best_node_seq_changed)

        new_instructions.all_instructions_blocks['base'] = base_update_instructions_block

        #        self.test()

        return new_instructions

        # todo i dont understand anymore when the instructions stops beeing propagated back
