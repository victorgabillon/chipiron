from src.players.treevalue.node_selector.notations_and_statics import zipf_picks_random

PRIORITIES = [NO_PRIORITY, PRIORITY_BEST, PRIORITY_TWO_BEST] = range(3)
dicPriorities_Sampling = {'no_priority': NO_PRIORITY, 'priority_best': PRIORITY_BEST,
                          'priority_two_best': PRIORITY_TWO_BEST}


class MoveExplorer:
    def __init__(self, priority_sampling):
        self.priority_sampling = dicPriorities_Sampling[priority_sampling]

    def alter_with_priorities(self, parent_node, child_node, proportion):
        if self.priority_sampling == NO_PRIORITY:
            return proportion
        if self.priority_sampling == PRIORITY_BEST:
            if child_node == parent_node.best_child():
                return 1
            else:
                return proportion
        if self.priority_sampling == PRIORITY_TWO_BEST:
            if child_node == parent_node.best_child():
                return .5
            elif child_node == parent_node.second_best_child():
                return .5
            else:
                return proportion


class ZipfMoveExplorer(MoveExplorer):
    def __init__(self, priority_sampling, random_generator):
        super().__init__(priority_sampling)
        self.random_generator = random_generator

    def sample_child_to_explore(self, tree_node_to_sample_from):
        sorted_not_over_children = tree_node_to_sample_from.sort_children_not_over()
        return zipf_picks_random(ordered_list_elements=sorted_not_over_children, random_generator=self.random_generator)


class ProportionMoveExplorer(MoveExplorer):

    def __init__(self, priority_sampling, random_generator):
        super().__init__(priority_sampling)
        self.random_generator = random_generator

    def sample_child_to_explore(self, tree_node_to_sample_from,
                                children_exception_set=set()):  # set of nodes that cannot be picked

        assert (len(tree_node_to_sample_from.children_not_over) > len(children_exception_set))  # to be able to pick

        # todo maybe proportions and proportions can be valuesorted dict with smart updates
        proportions = []
        children_candidates = []

        for child in tree_node_to_sample_from.children_not_over:
            if child not in children_exception_set:
                child_proportion = self.alter_with_priorities(tree_node_to_sample_from,
                                                              child,
                                                              tree_node_to_sample_from.proportions[child])
                proportions.append(child_proportion)
                children_candidates.append(child)

        min_child = self.random_generator.choices(children_candidates, proportions, k=1)
        return min_child[0]


class VisitProportionMoveExplorer(MoveExplorer):

    def __init__(self, priority_sampling):
        super().__init__(priority_sampling)

    def sample_child_to_explore(self, tree_node_to_sample_from,
                                children_exception_set=set()):  # set of nodes that cannot be picked

        assert (len(tree_node_to_sample_from.children_not_over) > len(children_exception_set))  # to be able to pick

        # todo maybe proportions and proportions can be valuesorted dict with smart updates
        proportions = [0] * len(tree_node_to_sample_from.children_not_over)

        min_index = 100000000000000000000000000000000000000000000000.
        min_child = None
        id_min = 100000000000000000000000000.

        for counter, child in enumerate(tree_node_to_sample_from.children_not_over):
            child_proportion = self.alter_with_priorities(tree_node_to_sample_from,
                                                          child,
                                                          tree_node_to_sample_from.proportions[child])
            child_index = child.descendants.get_count() / child_proportion

            if child not in children_exception_set:
                if (child_index, child.id) < (min_index, id_min):
                    min_index = child_index
                    min_child = child
                    id_min = child.id

        return min_child
