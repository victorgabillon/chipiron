import math
from typing import Protocol

import chess

import chipiron.players.move_selector.treevalue.trees as trees
from chipiron.players.move_selector.treevalue.indices.node_indices.index_data import MinMaxPathValue, \
    RecurZipfQuoolExplorationData, IntervalExplo
from chipiron.players.move_selector.treevalue.nodes.algorithm_node.algorithm_node import AlgorithmNode
from chipiron.utils.small_tools import Interval, intersect_intervals, distance_number_to_interval


class NodeExplorationIndexManager(Protocol):

    def update_root_node_index(
            self,
            root_node: AlgorithmNode,
    ) -> None:
        ...

    def update_node_indices(
            self,
            child_node: AlgorithmNode,
            parent_node: AlgorithmNode,
            tree: trees.MoveAndValueTree,
            child_rank: int,
    ) -> None:
        ...


class NullNodeExplorationIndexManager(NodeExplorationIndexManager):

    def update_root_node_index(
            self,
            root_node: AlgorithmNode,
    ) -> None:
        ...

    def update_node_indices(
            self,
            child_node: AlgorithmNode,
            parent_node: AlgorithmNode,
            tree: trees.MoveAndValueTree,
            child_rank: int,
    ) -> None:
        raise Exception('should not be raised')


class UpdateIndexGlobalMinChange:

    def update_root_node_index(
            self,
            root_node: AlgorithmNode,
    ) -> None:

        assert isinstance(root_node.exploration_index_data, MinMaxPathValue)
        root_value: float = root_node.minmax_evaluation.get_value_white()

        root_node.exploration_index_data.min_path_value = root_value
        root_node.exploration_index_data.max_path_value = root_value
        root_node.exploration_index_data.index = 0

    def update_node_indices(
            self,
            child_node: AlgorithmNode,
            parent_node: AlgorithmNode,
            tree: trees.MoveAndValueTree,
            child_rank: int,
    ) -> None:
        assert isinstance(parent_node.exploration_index_data, MinMaxPathValue)
        assert isinstance(child_node.exploration_index_data, MinMaxPathValue)

        assert parent_node.exploration_index_data is not None
        assert parent_node.exploration_index_data.min_path_value is not None
        assert parent_node.exploration_index_data.max_path_value is not None
        assert child_node.exploration_index_data is not None

        child_value: float = child_node.minmax_evaluation.get_value_white()

        child_node_min_path_value = min(
            child_value,
            parent_node.exploration_index_data.min_path_value
        )

        child_node_max_path_value = max(
            child_value,
            parent_node.exploration_index_data.max_path_value
        )

        # computes local_child_index the amount of change for the child node to become better than its parent
        child_index: float = abs(
            child_node_max_path_value - child_node_min_path_value) / 2

        # the amount of change for the child to become better than any of its ancestor
        # and become the overall best bode, the max is computed with the parent index
        # child_index: float = max(local_child_index, parent_index)

        # the index of the child node is updated now
        # as a child node can have multiple parents we take the min if an index was previously computed
        if child_node.exploration_index_data.index is None:
            child_node.exploration_index_data.index = child_index
            child_node.exploration_index_data.max_path_value = child_node_max_path_value
            child_node.exploration_index_data.min_path_value = child_node_min_path_value
        else:
            assert child_node.exploration_index_data.max_path_value is not None
            assert child_node.exploration_index_data.min_path_value is not None
            child_node.exploration_index_data.index = min(child_node.exploration_index_data.index, child_index)
            child_node.exploration_index_data.max_path_value = min(child_node_max_path_value,
                                                                   child_node.exploration_index_data.max_path_value)
            child_node.exploration_index_data.min_path_value = max(child_node_min_path_value,
                                                                   child_node.exploration_index_data.min_path_value)


class UpdateIndexZipfFactoredProba:

    def update_root_node_index(
            self,
            root_node: AlgorithmNode,
    ) -> None:

        assert isinstance(root_node.exploration_index_data, RecurZipfQuoolExplorationData)

        root_node.exploration_index_data.zipf_factored_proba = 1
        root_node.exploration_index_data.index = 0

    def update_node_indices(
            self,
            child_node: AlgorithmNode,
            parent_node: AlgorithmNode,
            tree: trees.MoveAndValueTree,
            child_rank: int,
    ) -> None:
        assert isinstance(parent_node.exploration_index_data, RecurZipfQuoolExplorationData)

        parent_zipf_factored_proba: float | None = parent_node.exploration_index_data.zipf_factored_proba
        assert parent_zipf_factored_proba is not None

        child_zipf_proba: float = 1 / (child_rank + 1)
        child_zipf_factored_proba: float = child_zipf_proba * parent_zipf_factored_proba
        inverse_depth: float = 1 / (tree.node_depth(child_node) + 1)
        child_index: float = child_zipf_factored_proba * inverse_depth
        child_index = -child_index

        assert child_node.exploration_index_data is not None
        assert isinstance(child_node.exploration_index_data, RecurZipfQuoolExplorationData)

        # the index of the child node is updated now
        # as a child node can have multiple parents we take the min if an index was previously computed
        if child_node.exploration_index_data.index is None:
            child_node.exploration_index_data.index = child_index
            child_node.exploration_index_data.zipf_factored_proba = child_zipf_factored_proba

        else:
            assert child_node.exploration_index_data.zipf_factored_proba is not None
            child_node.exploration_index_data.index = min(child_node.exploration_index_data.index, child_index)
            child_node.exploration_index_data.zipf_factored_proba = min(
                child_node.exploration_index_data.zipf_factored_proba, child_zipf_factored_proba)


class UpdateIndexLocalMinChange:

    def update_root_node_index(
            self,
            root_node: AlgorithmNode,
    ) -> None:

        assert isinstance(root_node.exploration_index_data, IntervalExplo)

        root_node.exploration_index_data.index = 0
        root_node.exploration_index_data.interval = Interval(
            min_value=-math.inf,
            max_value=math.inf
        )

    def update_node_indices(
            self,
            child_node: AlgorithmNode,
            parent_node: AlgorithmNode,
            tree: trees.MoveAndValueTree,
            child_rank: int,
    ) -> None:

        assert isinstance(parent_node.exploration_index_data, IntervalExplo)

        assert parent_node.exploration_index_data is not None
        assert child_node.exploration_index_data is not None

        inter_level_interval: Interval | None = None

        if parent_node.exploration_index_data.index is None:
            child_node.exploration_index_data.index = None
        else:
            assert parent_node.exploration_index_data.interval is not None
            if len(parent_node.tree_node.moves_children) == 1:
                local_index = parent_node.exploration_index_data.index
                inter_level_interval = parent_node.exploration_index_data.interval
            else:
                if parent_node.tree_node.board.turn == chess.WHITE:
                    best_child = parent_node.minmax_evaluation.best_child()
                    second_best_child = parent_node.minmax_evaluation.second_best_child()
                    child_white_value = child_node.minmax_evaluation.get_value_white()
                    local_interval = Interval()
                    if child_node == best_child:
                        local_interval.max_value = math.inf
                        local_interval.min_value = second_best_child.minmax_evaluation.get_value_white()
                    else:
                        local_interval.max_value = math.inf
                        local_interval.min_value = best_child.minmax_evaluation.get_value_white()
                    # print('intersectWHITE', parent_node.id, parent_node.tree_node.board.turn, local_interval,
                    #      parent_node.exploration_index_data.interval)

                    inter_level_interval = intersect_intervals(
                        local_interval,
                        parent_node.exploration_index_data.interval
                    )
                    if inter_level_interval is not None:
                        local_index = distance_number_to_interval(value=child_white_value,
                                                                  interval=inter_level_interval)
                    else:
                        local_index = None
                if parent_node.tree_node.board.turn == chess.BLACK:
                    best_child = parent_node.minmax_evaluation.best_child()
                    # print('parent_nodess', parent_node.id, child_node.id)

                    second_best_child = parent_node.minmax_evaluation.second_best_child()
                    child_white_value = child_node.minmax_evaluation.get_value_white()
                    local_interval = Interval()
                    if child_node == best_child:
                        local_interval.max_value = second_best_child.minmax_evaluation.get_value_white()
                        local_interval.min_value = -math.inf
                    else:
                        local_interval.max_value = best_child.minmax_evaluation.get_value_white()
                        local_interval.min_value = -math.inf
                    #  print('intersect', local_interval, parent_node.exploration_index_data.interval)

                    inter_level_interval = intersect_intervals(local_interval,
                                                               parent_node.exploration_index_data.interval)
                    if inter_level_interval is not None:
                        local_index = distance_number_to_interval(
                            value=child_white_value,
                            interval=inter_level_interval
                        )
                    else:
                        local_index = None
            # print('t', child_node.id, local_index, inter_level_interval)
            assert isinstance(child_node.exploration_index_data, IntervalExplo)

            if child_node.exploration_index_data.index is None:
                child_node.exploration_index_data.index = local_index
                child_node.exploration_index_data.interval = inter_level_interval

            elif local_index is not None:
                if local_index < child_node.exploration_index_data.index:
                    child_node.exploration_index_data.interval = inter_level_interval
                child_node.exploration_index_data.index = min(child_node.exploration_index_data.index, local_index)


# TODO their might be ways to optimize the computation such as not recomptuing for the whole tree
def update_all_indices(
        tree: trees.MoveAndValueTree,
        index_manager: NodeExplorationIndexManager
) -> None:
    """
    The idea is to compute an index $ind(n)$ for a node $n$ that measures the minimum amount of change
     in the value of all the nodes such that this node $n$ becomes the best.

    This can be computed recursively as :
    ind(n) = max( ind(parent(n),.5*abs(value(n)-value(parent(n))))

    Args:
        index_manager:
        tree: a tree

    Returns:

    """
    if isinstance(index_manager, NullNodeExplorationIndexManager):
        return

    tree_nodes: trees.RangedDescendants = tree.descendants

    index_manager.update_root_node_index(
        root_node=tree.root_node,
    )

    half_move: int
    for half_move in tree_nodes:
        # todo how are we sure that the hm comes in order?
        # print('hmv', half_move)
        parent_node: AlgorithmNode
        for parent_node in tree_nodes[half_move].values():
            child_node: AlgorithmNode
            # for child_node in parent_node.moves_children.values():
            child_rank: int
            for child_rank, child_node in enumerate(parent_node.minmax_evaluation.children_sorted_by_value_):
                #   assert (1 == 2)
                index_manager.update_node_indices(
                    child_node=child_node,
                    tree=tree,
                    child_rank=child_rank,
                    parent_node=parent_node
                )


# TODO their might be ways to optimize the computation such as not recomptuing for the whole tree


def print_all_indices(
        tree: trees.MoveAndValueTree,
) -> None:
    tree_nodes: trees.RangedDescendants = tree.descendants

    half_move: int
    for half_move in tree_nodes:
        # todo how are we sure that the hm comes in order?
        # print('hmv', half_move)
        parent_node: AlgorithmNode
        for parent_node in tree_nodes[half_move].values():
            if parent_node.exploration_index_data is not None:
                print('parent_node', parent_node.tree_node.id, parent_node.exploration_index_data.index)
