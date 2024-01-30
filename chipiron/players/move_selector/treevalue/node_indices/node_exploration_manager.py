import chess

import chipiron.players.move_selector.treevalue.trees as trees
import chipiron.players.move_selector.treevalue.nodes as nodes
from typing import Protocol
import math
from chipiron.utils.small_tools import Interval, intersect_intervals, distance_number_to_interval


class NodeExplorationIndexManager(Protocol):

    def update_root_node_index(
            self,
            root_node: nodes.AlgorithmNode,
    ) -> None:
        ...

    def update_node_indices(
            self,
            child_node: nodes.AlgorithmNode,
            parent_node: nodes.AlgorithmNode,
            tree: trees.MoveAndValueTree,
            child_rank: int,
    ) -> None:
        ...


class NullNodeExplorationIndexManager:

    def update_root_node(
            self,
            root_node: nodes.AlgorithmNode,
    ) -> None:
        ...

    def update_node_indices(
            self,
            child_node: nodes.AlgorithmNode,
            parent_node: nodes.AlgorithmNode,
            tree: trees.MoveAndValueTree,
            child_rank: int,
    ) -> None:
        raise Exception('should not be raised')


class UpdateIndexGlobalMinChange:

    def update_root_node_index(
            self,
            root_node: nodes.AlgorithmNode,
    ) -> None:
        root_value: float = root_node.minmax_evaluation.get_value_white()

        root_node.exploration_index_data.min_path_value = root_value
        root_node.exploration_index_data.max_path_value = root_value
        root_node.exploration_index_data.index = 0

    def update_node_indices(
            self,
            child_node: nodes.AlgorithmNode,
            parent_node: nodes.AlgorithmNode,
            tree: trees.MoveAndValueTree,
            child_rank: int,
    ) -> None:
        # print('child_node.id', child_node.id)
        parent_index: float = parent_node.exploration_index_data.index
        child_value: float = child_node.minmax_evaluation.get_value_white()
        parent_value: float = parent_node.minmax_evaluation.get_value_white()

        child_node_min_path_value = min(
            child_value,
            parent_node.exploration_index_data.min_path_value
        )

        child_node_max_path_value = max(
            child_value,
            parent_node.exploration_index_data.max_path_value
        )

        # computes local_child_index the amount of change for the child node to become better than its parent
        root_value = tree.root_node.minmax_evaluation.get_value_white()
        child_index: float = abs(
            child_node_max_path_value - child_node_min_path_value) / 2
        # local_child_index: float = abs(child_value - root_value) / 2

        #   print('child_index', child_index)
        # the amount of change for the child to become better than any of its ancestor
        # and become the overall best bode, the max is computed with the parent index
        # child_index: float = max(local_child_index, parent_index)
        # print('child_index', child_index)

        # the index of the child node is updated now
        # as a child node can have multiple parents we take the min if an index was previously computed
        if child_node.exploration_index_data.index is None:
            child_node.exploration_index_data.index = child_index
            child_node.exploration_index_data.max_path_value = child_node_max_path_value
            child_node.exploration_index_data.min_path_value = child_node_min_path_value
        else:
            child_node.exploration_index_data.index = min(child_node.exploration_index_data.index, child_index)
            child_node.exploration_index_data.max_path_value = min(child_node_max_path_value,
                                                                   child_node.exploration_index_data.max_path_value)
            child_node.exploration_index_data.min_path_value = max(child_node_min_path_value,
                                                                   child_node.exploration_index_data.min_path_value)


class UpdateIndexZipfFactoredProba:

    def update_root_node_index(
            self,
            root_node: nodes.AlgorithmNode,
    ) -> None:

        root_node.exploration_index_data.zipf_factored_proba = 1
        root_node.exploration_index_data.index = 0

    def update_node_indices(
            self,
            child_node: nodes.AlgorithmNode,
            parent_node: nodes.AlgorithmNode,
            tree: trees.MoveAndValueTree,
            child_rank: int,
    ) -> None:
        parent_zipf_factored_proba: float = parent_node.exploration_index_data.zipf_factored_proba
        child_zipf_proba = 1 / (child_rank + 1)
        child_zipf_factored_proba = child_zipf_proba * parent_zipf_factored_proba
        inverse_depth = 1 / (tree.node_depth(child_node) + 1)
        child_index = child_zipf_factored_proba * inverse_depth
        child_index = -child_index

        # the index of the child node is updated now
        # as a child node can have multiple parents we take the min if an index was previously computed
        if child_node.exploration_index_data.index is None:
            child_node.exploration_index_data.index = child_index
            child_node.exploration_index_data.zipf_factored_proba = child_zipf_factored_proba

        else:
            child_node.exploration_index_data.index = min(child_node.exploration_index_data.index, child_index)
            child_node.exploration_index_data.zipf_factored_proba = min(
                child_node.exploration_index_data.zipf_factored_proba, child_zipf_factored_proba)


class UpdateIndexLocalMinChange:

    def update_root_node_index(
            self,
            root_node: nodes.AlgorithmNode,
    ) -> None:

        root_node.exploration_index_data.index = 0
        root_node.exploration_index_data.interval = Interval(min_value=-math.inf, max_value=math.inf)

    def update_node_indices(
            self,
            child_node: nodes.AlgorithmNode,
            parent_node: nodes.AlgorithmNode,
            tree: trees.MoveAndValueTree,
            child_rank: int,
    ) -> None:

        depth: int = tree.node_depth(child_node)
        if parent_node.exploration_index_data.index is None:
            child_node.exploration_index_data.index = None
        else:
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
                    print('intersectWHITE', parent_node.id, parent_node.tree_node.board.turn, local_interval, parent_node.exploration_index_data.interval)

                    inter_level_interval = intersect_intervals(local_interval, parent_node.exploration_index_data.interval)
                    if inter_level_interval is not None:
                        local_index = distance_number_to_interval(value=child_white_value, interval=inter_level_interval)
                    else:
                        local_index = None
                if parent_node.tree_node.board.turn == chess.BLACK:
                    best_child = parent_node.minmax_evaluation.best_child()
                    second_best_child = parent_node.minmax_evaluation.second_best_child()
                    child_white_value = child_node.minmax_evaluation.get_value_white()
                    local_interval = Interval()
                    if child_node == best_child:
                        local_interval.max_value = second_best_child.minmax_evaluation.get_value_white()
                        local_interval.min_value = -math.inf
                    else:
                        local_interval.max_value = best_child.minmax_evaluation.get_value_white()
                        local_interval.min_value = -math.inf
                    print('intersect', local_interval, parent_node.exploration_index_data.interval)

                    inter_level_interval = intersect_intervals(local_interval, parent_node.exploration_index_data.interval)
                    if inter_level_interval is not None:
                        local_index = distance_number_to_interval(value=child_white_value, interval=inter_level_interval)
                    else:
                        local_index = None
            print('t',child_node.id,local_index,inter_level_interval)
            if child_node.exploration_index_data.index is None:
                child_node.exploration_index_data.index = local_index
                child_node.exploration_index_data.interval = inter_level_interval

            elif local_index is not None:
                if local_index <child_node.exploration_index_data.index :
                    child_node.exploration_index_data.interval = inter_level_interval
                child_node.exploration_index_data.index = min(child_node.exploration_index_data.index, local_index)


# TODO their might be ways to optimize the computation such as not recomptuing for the whole tree
def update_all_indices(
        tree: trees.move_and_value_tree,
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

    # for half_move in tree_nodes:
    #    for node in tree_nodes[half_move].values():
    #        for child in node.moves_children.values():
    #            child.exploration_index_data.index = None  # rootnode is not set to zero haha

    # update_indices.init_root_node_indices(root_node=tree.root_node)

    index_manager.update_root_node_index(
        root_node=tree.root_node,
    )

    half_move: int
    for half_move in tree_nodes:
        # todo how are we sure that the hm comes in order?
        # print('hmv', half_move)
        parent_node: nodes.AlgorithmNode
        for parent_node in tree_nodes[half_move].values():
            # print('sparent_node', parent_node.tree_node.id,parent_node.minmax_evaluation)
            child_node: nodes.AlgorithmNode
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
def update_all_indices_base(
        tree: trees.move_and_value_tree,
) -> None:
    """
    The idea is to compute an index $ind(n)$ for a node $n$ that measures the minimum amount of change
     in the value of all the nodes such that this node $n$ becomes the best.

    This can be computed recursively as :
    ind(n) = max( ind(parent(n),.5*abs(value(n)-value(parent(n))))

    Args:
        tree: a tree

    Returns:

    """
    tree_nodes: trees.RangedDescendants = tree.descendants

    for half_move in tree_nodes:
        for node in tree_nodes[half_move].values():
            for child in node.moves_children.values():
                child.exploration_manager.index = None  # rootnode is not set to zero haha

    start = True
    half_move: int
    for half_move in tree_nodes:
        # todo how are we sure that the hm comes in order?
        # print('hmv', half_move)
        parent_node: nodes.AlgorithmNode
        for parent_node in tree_nodes[half_move].values():
            if start:
                parent_node.exploration_manager.index = 0  # very dirty hack to put the root node to zero (improve please!!)
                start = False
            # print('parent_node', parent_node.tree_node.id)
            child_node: nodes.AlgorithmNode
            for child_node in parent_node.moves_children.values():

                # print('child_node', child_node.tree_node.id)

                parent_index: float = parent_node.exploration_manager.index
                child_value: float = child_node.minmax_evaluation.get_value_white()
                parent_value: float = parent_node.minmax_evaluation.get_value_white()

                # computes local_child_index the amount of change for the child node to become better than its parent
                local_child_index: float = abs(child_value - parent_value) / 2

                # the amount of change for the child to become better than any of its ancestor
                # and become the overall best bode, the max is computed with the parent index
                #   print('child_node', child_node.tree_node.id, local_child_index, parent_index)

                # print('local_child_index', child_node.tree_node.id, local_child_index)

                child_index: float = max(local_child_index, parent_index)
                # print('child_index', child_node.tree_node.id, child_index)

                # the index of the child node is updated now
                # as a child node can have multiple parents we take the min if an index was previously computed
                if child_node.exploration_manager.index is None:
                    child_node.exploration_manager.index = child_index
                else:
                    child_node.exploration_manager.index = min(child_node.exploration_manager.index, child_index)


#  import time
#  time.sleep(1)
#  print('-------------------------')


def update_all_indices_recurzipfsequool(
        tree: trees.move_and_value_tree
) -> None:
    """
    The idea is to compute an index $ind(n)$ for a node $n$ that measures

    Args:
        tree: a tree

    Returns:

    """
    tree_nodes: trees.RangedDescendants = tree.descendants

    for half_move in tree_nodes:
        for node in tree_nodes[half_move].values():
            for child in node.moves_children.values():
                child.exploration_manager.index = None  # rootnode is not set to zero haha

    start = True
    half_move: int
    for half_move in tree_nodes:
        parent_node: nodes.AlgorithmNode
        for parent_node in tree_nodes[half_move].values():

            if start:
                parent_node.exploration_manager.index = 0  # very dirty hack to put the root node to zero (improve please!!)
                parent_node.exploration_manager.zipf_factored_proba = 1
                start = False
            child_node: nodes.AlgorithmNode
            for child_rank, child_node in enumerate(parent_node.minmax_evaluation.children_sorted_by_value_):
                parent_zipf_factored_proba: float = parent_node.exploration_manager.zipf_factored_proba
                child_zipf_proba = 1 / (child_rank + 1)
                child_zipf_factored_proba = child_zipf_proba * parent_zipf_factored_proba
                inverse_depth = 1 / (tree.node_depth(child_node) + 1)
                child_index = child_zipf_factored_proba * inverse_depth
                child_index = -child_index

                # the index of the child node is updated now
                # as a child node can have multiple parents we take the min if an index was previously computed
                if child_node.exploration_manager.index is None:
                    child_node.exploration_manager.index = child_index
                    child_node.exploration_manager.zipf_factored_proba = child_zipf_factored_proba

                else:
                    child_node.exploration_manager.index = min(child_node.exploration_manager.index, child_index)
                    child_node.exploration_manager.zipf_factored_proba = min(
                        child_node.exploration_manager.zipf_factored_proba, child_zipf_factored_proba)

            # print('child_node', child_node.tree_node.id, child_node.exploration_manager.index)


def update_all_indices_min_local_change(
        tree: trees.move_and_value_tree
) -> None:
    tree_nodes: trees.RangedDescendants = tree.descendants

    for half_move in tree_nodes:
        for node in tree_nodes[half_move].values():
            for child in node.moves_children.values():
                child.exploration_manager.index = None  # rootnode is not set to zero haha

    if not tree.root_node.moves_children:
        return

    root_node_value_white = tree.root_node.minmax_evaluation.get_value_white()
    root_node_second_value_white = tree.root_node.minmax_evaluation.second_best_child().minmax_evaluation.get_value_white()

    for depth in range(tree.get_max_depth()):
        # print('depth',depth)
        for node in tree.all_nodes[depth].values():
            for child in node.moves_children.values():
                #      print('child', child.id)
                if node.exploration_manager.index is None:
                    index = None
                else:
                    if depth % 2 == 0:
                        if tree.root_node.best_child() in child.first_moves:  # todo what if it is inboth at the sam time
                            if child == node.best_child():
                                index = abs(child.value_white - root_node_second_value_white) / 2
                            else:
                                index = None
                        else:
                            index = abs(child.value_white - root_node_value_white) / 2
                    else:  # depth %2 ==1
                        if tree.root_node.best_child() in child.first_moves:
                            index = abs(child.value_white - root_node_second_value_white) / 2
                        else:  # not the best line
                            if child == node.best_child():
                                index = abs(child.value_white - root_node_value_white) / 2
                            else:  # not the best child response
                                index = None
                if index is not None:
                    if child.exploration_manager.index is None:  # if the index has beene initiated already by another parent node
                        child.exploration_manager.index = index
                        if child.id == tree.root_node.best_node_sequence[-1].id:
                            assert (tree.root_node.best_node_sequence[-1].exploration_manager.index is not None)

                    else:
                        child.exploration_manager.index = min(child.exploration_manager.index, index)
                        if child.id == tree.root_node.best_node_sequence[-1].id:
                            assert (tree.root_node.best_node_sequence[-1].exploration_manager.index is not None)

    assert (tree.root_node.best_node_sequence[-1].index is not None)


def print_all_indices(
        tree: trees.move_and_value_tree,

) -> None:
    tree_nodes: trees.RangedDescendants = tree.descendants

    half_move: int
    for half_move in tree_nodes:
        # todo how are we sure that the hm comes in order?
        # print('hmv', half_move)
        parent_node: nodes.AlgorithmNode
        for parent_node in tree_nodes[half_move].values():
            print('parent_node', parent_node.tree_node.id, parent_node.exploration_index_data.index)
