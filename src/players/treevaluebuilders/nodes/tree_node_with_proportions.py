from src.players.treevaluebuilders.nodes.tree_node_with_descendants import NodeWithDescendants
import numba
import numpy as np


class ProportionsNode(NodeWithDescendants):

    def __init__(self, board, half_move, id_number, parent_node):
        super().__init__(board, half_move, id_number, parent_node)
        self.proportions = {}

    def update_over(self, children_with_updated_over):
        """ updating the over_event of the node based on notification of change of over_event in children"""

        for child in children_with_updated_over:
            assert child.is_over()
            if child in self.proportions:
                self.proportions.pop(child)

        return super().update_over(children_with_updated_over)

    def perform_updates(self, update_instructions):
        new_update_instructions = super().perform_updates(update_instructions)

        if update_instructions['base']['children_with_updated_value']:
            if not self.is_over() and self.children_not_over:
                # note that the previous compute can switch the over state so need to be careful
                self.compute_proportions()  # todo: could be optimised?

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

    def compute_proportions(self):
        sorted_children = self.sort_children_not_over()
        assert (len(sorted_children) == len(self.children_not_over))
        self.proportions = {}

        if len(sorted_children) == 1:
            self.proportions[sorted_children[0]] = 1
            assert (len(self.proportions) == len(self.children_not_over))

            return

        gap = self.gaps(sorted_children)
        propo = fast_compute_proportions(gap)

        for counter, child in enumerate(sorted_children):
            self.proportions[child] = propo[counter]

        assert (len(self.proportions) == len(self.children_not_over))

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
def fast_compute_proportions(gap):  # NOT NORMALIZED TO SAVE TIME!!!
    propo = [0.] * len(gap)

    for counter, moves in enumerate(gap):
        addterm = 0
        for counter2, move2 in enumerate(gap[counter + 1:]):
            truecounter2 = counter + counter2 + 1
            if gap[truecounter2] == 0:
                addterm += 1  #
            else:
                addterm += (gap[counter]) ** 2 / float((gap[truecounter2]) ** 2)
        propo[counter] = 1 / float(counter + 1 + addterm)

    # if style == ZIPF_ONE:
    #     propo[0] = 1
    # if style == ZIPF_TWO:
    #     propo[0] = .5
    #     propo[1] = .5

    return propo
