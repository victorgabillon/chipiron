import random
from enum import Enum

from chipiron.players.move_selector.treevalue.node_selector.notations_and_statics import zipf_picks_random
from chipiron.players.move_selector.treevalue.nodes.algorithm_node import AlgorithmNode


class SamplingPriorities(str, Enum):
    NO_PRIORITY: str = 'no_priority'
    PRIORITY_BEST: str = 'priority_best'
    PRIORITY_TWO_BEST: str = 'priority_two_best'


# PRIORITIES = [NO_PRIORITY, PRIORITY_BEST, PRIORITY_TWO_BEST] = range(3)
# dicPriorities_Sampling = {'no_priority': NO_PRIORITY, 'priority_best': PRIORITY_BEST,
#                         'priority_two_best': PRIORITY_TWO_BEST}


class MoveExplorer:
    priority_sampling: SamplingPriorities

    def __init__(
            self,
            priority_sampling: SamplingPriorities
    ):
        self.priority_sampling = priority_sampling


class ZipfMoveExplorer(MoveExplorer):
    def __init__(
            self,
            priority_sampling: SamplingPriorities,
            random_generator: random.Random
    ) -> None:
        super().__init__(priority_sampling)
        self.random_generator = random_generator

    def sample_child_to_explore(
            self,
            tree_node_to_sample_from: AlgorithmNode
    ) -> AlgorithmNode:
        sorted_not_over_children = tree_node_to_sample_from.minmax_evaluation.sort_children_not_over()

        child = zipf_picks_random(
            ordered_list_elements=sorted_not_over_children,
            random_generator=self.random_generator
        )
        assert isinstance(child,AlgorithmNode)
        return child
