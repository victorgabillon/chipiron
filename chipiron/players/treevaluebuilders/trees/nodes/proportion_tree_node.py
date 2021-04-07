from players.treevaluebuilders.trees.nodes.tree_node_with_descendants import NodeWithDescendants
from players.treevaluebuilders.notations_and_statics import ZIPF, ZIPF_TWO, ZIPF_ONE
import numba
import numpy as np
import random

dicZStyle = {'Zipf': ZIPF, 'ZipfOne': ZIPF_ONE, 'ZipfTwo': ZIPF_TWO}


class VisitsAndProportionsNode(NodeWithDescendants):

    def __init__(self, board, half_move, id_number, parent_node, zipf_style_str):
        super().__init__(board, half_move, id_number, parent_node)
        self.proportions = {}
        self.zipf_style = dicZStyle[zipf_style_str]

    def update_over(self, children_with_updated_over):
        """ updating the over_event of the node based on notification of change of over_event in children"""

        for child in children_with_updated_over:
            assert child.is_over()
            if child in self.proportions:
                self.proportions.pop(child)

        return super().update_over(children_with_updated_over)

    # def create_update_instructions_after_node_birth(self):
    #     update_instructions = super().create_update_instructions_after_node_birth()
    #     update_instructions['proportions'] = UpdateInstructionsProportionsBlock(should_update_proportions=True)
    #     return update_instructions

    def perform_updates(self, update_instructions):
        new_update_instructions = super().perform_updates(update_instructions)

        if update_instructions['base']['children_with_updated_value']:
            if not self.is_over() and self.children_not_over:
                # note that the previous compute can switch the over state so need to be careful
                self.compute_proportions(self.zipf_style)  # todo: could be optimised?

        return new_update_instructions

    def print_proportions(self):
        print('---here are the proportions:')
        for child_node, proportion in self.proportions.items():
            print(self.moves_children.inverse[child_node], child_node.id, proportion)

    def gaps(self, sorted_children):  # todo numba this?
        gap = [0] * len(sorted_children)
        best_children = sorted_children[self.best_index_for_value]
        assert (self.best_index_for_value == 0)  # todo make it ok to have ascending order too asome point

        for counter, child in enumerate(sorted_children[1:]):
            gap[counter + 1] = float(best_children.get_value_white() - child.get_value_white())

        gap[0] = gap[1]
        return np.array(gap)

    def compute_proportions(self, zipf_style):
        sorted_children = self.sort_children()
        assert (len(sorted_children) == len(self.children_not_over))
        self.proportions = {}

        if len(sorted_children) == 1:
            self.proportions[sorted_children[0]] = 1
            assert (len(self.proportions) == len(self.children_not_over))

            return

        gap = self.gaps(sorted_children)

        propo = fast_compute_proportions(gap, zipf_style)

        for counter, child in enumerate(sorted_children):
            self.proportions[child] = propo[counter]

        assert (len(self.proportions) == len(self.children_not_over))

    # self.print_proportions()

    def choose_child_with_proportions(self,
                                      children_exception_set=set()):  # set of nodes that cannot be picked

        assert (len(self.children_not_over) > len(children_exception_set))  # to be able to pick

        # todo maybe proportions and proportions can be valuesorted dict with smart updates
        proportions = []
        children_candidates = []

        print('##', [(child.descendants.get_count(), str(self.moves_children.inverse[child])) for counter, child in
                     enumerate(self.children_not_over)])
        print('@@', [(float(self.proportions[child]), str(self.moves_children.inverse[child])) for counter, child in
                     enumerate(self.children_not_over)])

        for counter, child in enumerate(self.children_not_over):
            if child not in children_exception_set:
                proportions.append(float(self.proportions[child]))
                children_candidates.append(child)

       # print('#s#', proportions)
        #print('@s@', children_candidates)
        min_child = random.choices(children_candidates, proportions, k=1)

        return min_child[0]

    def choose_child_with_visits_and_proportions(self,
                                                 children_exception_set=set()):  # set of nodes that cannot be picked

        assert (len(self.children_not_over) > len(children_exception_set))  # to be able to pick

        # todo maybe proportions and proportions can be valuesorted dict with smart updates
        proportions = [0] * len(self.children_not_over)

        min_ = 100000000000000000000000000000000000000000000000.
        min_child = None
        id_min = 100000000000000000000000000.

        print('##', [(child.descendants.get_count(), str(self.moves_children.inverse[child])) for counter, child in
                     enumerate(self.children_not_over)])
        print('@@', [(float(self.proportions[child]), str(self.moves_children.inverse[child])) for counter, child in
                     enumerate(self.children_not_over)])

        for counter, child in enumerate(self.children_not_over):
            proportions[counter] = child.descendants.get_count() / float(self.proportions[child])

            if child not in children_exception_set:
                if False:  # child.is_opening_a_priority():
                    min_child = child
                    return min_child
                elif (proportions[counter], child.id) < (min_, id_min):
                    min_ = proportions[counter]
                    min_child = child
                    id_min = child.id

        return min_child

    def test(self):
        super().test()
        self.test_proportions()

    def test_proportions(self):
        if not self.is_over():
            assert (len(self.proportions) == len(self.children_not_over))

    def description_tree_visualizer_move(self, child):
        if not self.is_over():
            return '#'  # '"{:.5f}".format(self.proportions[child])
        else:
            return ''


@numba.jit(nopython=True)
def fast_compute_proportions(gap, style):  # NOT NORMALIZED TO SAVE TIME!!!
    propo = [0.] * len(gap)
    #  sum_ = 0

    for counter, moves in enumerate(gap):
        addterm = 0
        for counter2, move2 in enumerate(gap[counter + 1:]):
            truecounter2 = counter + counter2 + 1
            if gap[truecounter2] == 0:
                addterm += 1  #
            else:
                addterm += (gap[counter]) ** 2 / float((gap[truecounter2]) ** 2)
        propo[counter] = 1 / float(counter + 1 + addterm)
    #     sum_ += propo[counter]

    if style == ZIPF_ONE:
        propo[0] = 1
    if style == ZIPF_TWO:
        propo[0] = .5
        propo[1] = .5

    # for counter, move in enumerate(gap):
    #    propo[counter] = propo[counter] / float(sum_)

    return propo
