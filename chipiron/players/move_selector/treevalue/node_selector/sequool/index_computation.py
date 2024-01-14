import chipiron.players.move_selector.treevalue.trees as trees
import chipiron.players.move_selector.treevalue.nodes as nodes
from typing import Protocol
from enum import Enum


class IndexComputationType(str, Enum):
    MinGlobalChange = 'min_global_change'
    MinLocalChange = 'min_local_change'
    RecurZipf = 'recurzipf'


class UpdateAllIndices(Protocol):
    def __call__(self, tree: trees.move_and_value_tree) -> None:
        ...


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


def update_all_indices(
        all_nodes_not_opened: trees.RangedDescendants
) -> None:
    half_move: int
    for half_move in all_nodes_not_opened:
        node: nodes.AlgorithmNode
        node.index = None
        for node in all_nodes_not_opened.descendants_at_half_move[half_move].values():
            for child in node.moves_children.values():
                child.index = None

    if not tree.root_node.moves_children:
        return

    root_node_value_white = tree.root_node.minmax_evaluation.get_value_white()
    root_node_second_value_white = tree.root_node.minmax_evaluation.second_best_child().minmax_evaluation.get_value_white()

    for depth in range(tree.get_max_depth()):
        # print('depth',depth)
        for node in tree.all_nodes[depth].values():
            for child in node.moves_children.values():
                #      print('child', child.id)
                if node.index is None:
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
                        if self.root_node.best_child() in child.first_moves:
                            index = abs(child.value_white - root_node_second_value_white) / 2
                        else:  # not the best line
                            if child == node.best_child():
                                index = abs(child.value_white - root_node_value_white) / 2
                            else:  # not the best child response
                                index = None
                if index is not None:
                    if child.index is None:  # if the index has beene initiated already by another parent node
                        child.index = index
                        if child.id == tree.root_node.best_node_sequence[-1].id:
                            assert (tree.root_node.best_node_sequence[-1].index is not None)

                    else:
                        child.index = min(child.index, index)
                        if child.id == tree.root_node.best_node_sequence[-1].id:
                            assert (tree.root_node.best_node_sequence[-1].index is not None)

    assert (tree.root_node.best_node_sequence[-1].index is not None)
