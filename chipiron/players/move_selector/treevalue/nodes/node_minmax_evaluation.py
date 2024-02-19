from __future__ import annotations
import chess
import math
from random import choice
from .tree_node import TreeNode
from chipiron.players.boardevaluators.over_event import OverEvent
from chipiron.utils.small_tools import nth_key
from chipiron.utils.my_value_sorted_dict import sort_dic
from typing import List
from typing import Protocol
from dataclasses import dataclass, field


# todo maybe further split values from over?

# Class created to avoid circular import and defines what is seen and needed by the NodeMinmaxEvaluation class
class NodeWithValue(Protocol):
    minmax_evaluation: NodeMinmaxEvaluation


@dataclass(slots=True)
class NodeMinmaxEvaluation:
    # a reference to the original tree node that is evaluated
    tree_node: TreeNode

    # absolute value wrt to white player as estimated by an evaluator
    value_white_evaluator: float | None = None

    # absolute value wrt to white player as computed from the value_white_* of the descendants
    # of this node (self) by a minmax procedure.
    value_white_minmax: float | None = None

    # self.best_move_sequence = []
    best_node_sequence: list = field(default_factory=list)

    # the children of the tree node are kept in a dictionary that can be sorted by their evaluations ()

    # children_sorted_by_value records subjective values of children by descending order
    # subjective value means the values is from the point of view of player_to_move
    # careful, I have hard coded in the self.best_child() function the descending order for
    # fast access to the best element, so please do not change!
    # self.children_sorted_by_value_vsd = ValueSortedDict({})
    children_sorted_by_value_: dict[NodeWithValue, tuple] = field(default_factory=dict)

    # self.children_sorted_by_value = {}

    # convention of descending order, careful if changing read above!!
    best_index_for_value: int = 0

    # the list of children that have not yet be found to be over
    # using atm a list instead of set as atm python set are not insertion ordered which adds randomness
    # and makes debug harder
    children_not_over: list = field(default_factory=list)

    # creating a base Over event that is set to None
    over_event: OverEvent = field(default_factory=OverEvent)

    @property
    def children_sorted_by_value(self):
        return self.children_sorted_by_value_

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
        subjective_value = self.get_value_white() if self.tree_node.player_to_move == chess.WHITE else -self.get_value_white()
        return subjective_value

    def subjective_value_of(self, another_node):
        # return the value from the point of view of the self.player_to_move of the value of another_node
        if self.tree_node.player_to_move == chess.WHITE:
            subjective_value = another_node.minmax_evaluation.get_value_white()
        else:
            subjective_value = -another_node.minmax_evaluation.get_value_white()
        return subjective_value

    def best_child(self):
        # fast way to access first key with the highest subjective value
        if self.children_sorted_by_value:
            best_child = next(iter(self.children_sorted_by_value))
        else:
            best_child = None
        return best_child

    def best_child_not_over(self):
        for child in self.children_sorted_by_value:
            if not child.is_over():
                return child
        assert (1 == 0)

    def best_child_value(self):
        # fast way to access first key with the highest subjective value
        if self.children_sorted_by_value:
            best_value = next(iter(self.children_sorted_by_value.values()))
        else:
            best_value = None
        return best_value

    def second_best_child(self) -> NodeWithValue:
        assert (len(self.children_sorted_by_value) >= 2)
        # fast way to access second key with the highest subjective value
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

    def print_children_sorted_by_value_and_exploration(self):
        print('here are the ', len(self.children_sorted_by_value), ' children sorted by value: ')
        for child_node, subjective_sort_value in self.children_sorted_by_value.items():
            print(self.tree_node.moves_children.inverse[child_node], subjective_sort_value[0], end=' $$ ')
            pass  # print(self.moves_children.inverse[child_node], subjective_sort_value[0],'('+str(child_node.descendants.number_of_descendants)+')', end=' $$ ')
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
        child_value_white = child.minmax_evaluation.get_value_white()
        subjective_value_of_child = -child_value_white if self.tree_node.player_to_move == chess.WHITE else child_value_white
        if self.is_over():
            # the shorter the check the better now
            self.children_sorted_by_value_[child] = (
                subjective_value_of_child, -len(child.minmax_evaluation.best_node_sequence), child.tree_node.id)

        else:
            # the longer the line the better now
            self.children_sorted_by_value_[child] = (
                subjective_value_of_child, len(child.minmax_evaluation.best_node_sequence), child.tree_node.id)

    def are_equal_values(self, value_1, value_2):
        return value_1 == value_2

    def are_considered_equal_values(self, value_1, value_2):
        return value_1[:2] == value_2[:2]

    def are_almost_equal_values(self, value_1, value_2):
        # expect floats
        epsilon = 0.01
        return value_1 > value_2 - epsilon and value_2 > value_1 - epsilon

    def becoming_over_from_children(self):
        """ this node is asked to switch to over status"""
        assert (not self.is_over())

        # becoming over triggers a full update record_sort_value_of_child
        # where ties are now broken to reach over as fast as possible
        # todo we should reach it asap if we are winning and think about what to ddo in other scenarios....
        for child in self.tree_node.moves_children.values():
            self.record_sort_value_of_child(child)

        # fast way to access first key with the highest subjective value
        best_child = self.best_child()

        self.over_event.becomes_over(how_over=best_child.minmax_evaluation.over_event.how_over,
                                     who_is_winner=best_child.minmax_evaluation.over_event.who_is_winner)

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
            if not self.is_over() and child.minmax_evaluation.is_winner(self.tree_node.player_to_move):
                self.becoming_over_from_children()
                is_newly_over = True

        # check if all children are over but not winning for self.player_to_move
        if not self.is_over() and not self.children_not_over:
            self.becoming_over_from_children()
            is_newly_over = True

        return is_newly_over

    def update_children_values(
            self,
            children_nodes_to_consider
    ) -> None:
        for child in children_nodes_to_consider:
            self.record_sort_value_of_child(child)
        self.children_sorted_by_value_ = sort_dic(self.children_sorted_by_value_)

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
        if self.tree_node.all_legal_moves_generated:
            self.value_white_minmax = best_child.minmax_evaluation.get_value_white()
        elif self.tree_node.player_to_move == chess.WHITE:
            self.value_white_minmax = max(best_child.minmax_evaluation.get_value_white(), self.value_white_evaluator)
        else:
            self.value_white_minmax = min(best_child.minmax_evaluation.get_value_white(), self.value_white_evaluator)

    def update_best_move_sequence(
            self,
            children_nodes_with_updated_best_move_seq: List[NodeMinmaxEvaluation]
    ) -> bool:
        """ triggered if a children notifies an updated best node sequence
        returns boolean: True if self.best_node_sequence is modified, False otherwise"""
        has_best_node_seq_changed: bool = False
        best_node = self.best_node_sequence[0]

        # returns the bestnode if it is in children_nodes_with_updated_best_move_seq else None
        # best_node_in_update = next((x for x in children_nodes_with_updated_best_move_seq
        #                           if x.tree_node == best_node), None)

        if best_node in children_nodes_with_updated_best_move_seq:
            self.best_node_sequence = [best_node] + best_node.minmax_evaluation.best_node_sequence
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

        self.best_node_sequence = [best_child] + best_child.minmax_evaluation.best_node_sequence
        assert self.best_node_sequence

    def is_value_subjectively_better_than_evaluation(self, value_white):
        subjective_value = self.subjective_value(value_white)
        return subjective_value >= self.value_white_evaluator

    def minmax_value_update_from_children(
            self,
            children_with_updated_value
    ):
        """ updates the values of children in self.sortedchildren
        updates value minmax and updates the best move"""

        # todo to be tested!!

        # updates value
        value_white_before_update = self.get_value_white()
        best_child_before_update = self.best_child()
        self.update_children_values(children_with_updated_value)
        self.update_value_minmax()

        value_white_after_update = self.get_value_white()
        has_value_changed: bool = value_white_before_update != value_white_after_update

        # # updates best_move #todo maybe split in two functions but be careful one has to be done oft the other
        if best_child_before_update is None:
            best_child_before_update_not_the_best_anymore = True
        else:
            # here we compare the values in the self.children_sorted_by_value which might include more
            # than just the basic values #todo make that more clear at some point maybe even creating a value object
            updated_value_of_best_child_before_update = self.children_sorted_by_value[best_child_before_update]
            best_value_children_after = self.best_child_value()
            best_child_before_update_not_the_best_anymore = not self.are_equal_values(
                updated_value_of_best_child_before_update,
                best_value_children_after)

        best_node_seq_before_update = self.best_node_sequence.copy()
        if self.tree_node.all_legal_moves_generated:
            if best_child_before_update_not_the_best_anymore:
                self.one_of_best_children_becomes_best_next_node()
        else:
            # we only consider a child as best if it is more promising than the evaluation of self
            # in self.value_white_evaluator
            if self.is_value_subjectively_better_than_evaluation(
                    best_child_before_update.minmax_evaluation.get_value_white()):
                self.one_of_best_children_becomes_best_next_node()
            else:
                self.best_node_sequence = []
        best_node_seq_after_update = self.best_node_sequence
        has_best_node_seq_changed = best_node_seq_before_update != best_node_seq_after_update

        return has_value_changed, has_best_node_seq_changed

    def dot_description(self):
        value_mm = "{:.3f}".format(self.value_white_minmax) if self.value_white_minmax is not None else 'None'
        value_eval = "{:.3f}".format(self.value_white_evaluator) if self.value_white_evaluator is not None else 'None'
        return '\n wh_val_mm: ' + value_mm + '\n wh_val_eval: ' \
            + value_eval + '\n moves*' + \
            self.description_best_move_sequence() + '\nover: ' + self.over_event.get_over_tag()

    def description_best_move_sequence(self):
        res = ''
        parent_node = self
        for child_node in self.best_node_sequence:
            move = parent_node.tree_node.moves_children.inverse[child_node]
            parent_node = child_node
            res += '_' + str(move)
        return res

    def description_tree_visualizer_move(self, child):
        return ''

    def test(self):
        super().test()
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
        # todo test if the sequence is empty, does it make sense? now yes and test if it is when the opened children
        #  have a value less promissing than the self.value_white_evaluator
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
            assert (old_best_node.best_node_sequence[0] == node)

            assert (old_best_node.best_node_sequence[0] == old_best_node.best_child())
            old_best_node = node

    def print_best_line(self):
        print('Best line from node ' + str(self.tree_node.id) + ':', end=' ')
        parent_node = self.tree_node
        for child in self.best_node_sequence:
            print(parent_node.moves_children.inverse[child], '(' + str(child.tree_node.id) + ')', end=' ')
            parent_node = child
        print(' ')

    def my_logit(self, x):  # todo look out for numerical problem with utmatic rounding to 0 or especillay to 1
        y = min(max(x, .000000000000000000000001), .9999999999999999)
        return math.log(y / (1 - y)) * max(1, abs(x))  # the * min(1,x) is a hack to prioritize game over

    def get_all_of_the_best_moves(
            self,
            how_equal=None
    ):
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

    def best_node_sequence_filtered_from_over(self):
        # todo investigate the case of having over in the best lines? does it make sense? what does it mean?
        res = [self]
        for best_child in self.best_node_sequence:
            if not best_child.is_over():
                res.append(best_child)
        return res

    def best_node_sequence_not_over(self):
        """ computes and returns the best line that does not contain any over state"""
        node = self
        res = []
        while node.are_all_moves_and_children_opened():
            # not sure this is the right condition in the long run but will work for now. assumes opening all moves all the time
            node = node.best_child_not_over()
            res.append(node)
        return res
