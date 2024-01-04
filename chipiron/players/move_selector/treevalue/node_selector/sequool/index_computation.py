import players.move_selector.treevalue.trees as trees
import players.move_selector.treevalue.nodes as nodes
from typing import Protocol


class UpdateAllIndices(Protocol):
    def __call__(self, all_nodes_not_opened: trees.RangedDescendants) -> None:
        ...


# TODO their might be ways to optimize the computation such as not recomptuing for the whole tree
def update_all_indices_base(
        all_nodes_not_opened: trees.RangedDescendants
) -> None:
    """
    The idea is to compute an index $ind(n)$ for a node $n$ that measures the minimum amount of change
     in the value of all the nodes such that this node $n$ becomes the best.

    This can be computed recursively as :
    ind(n) = max( ind(parent(n),.5*abs(value(n)-value(parent(n))))

    Args:
        all_nodes_not_opened: a tree composed only of the nodes not yer opened by the algorithm

    Returns:

    """
    half_move: int
    for half_move in all_nodes_not_opened:
        parent_node: nodes.AlgorithmNode
        for parent_node in all_nodes_not_opened[half_move].values():
            child_node: nodes.AlgorithmNode
            for child_node in parent_node.moves_children.values():
                parent_index: float = parent_node.exploration_manager.index
                child_value: float = child_node.minmax_evaluation.get_value_white()
                parent_value: float = parent_node.minmax_evaluation.get_value_white()

                # computes local_child_index the amount of change for the child node to become better than its parent
                local_child_index: float = abs(child_value - parent_value) / 2

                # the amount of change for the child to become better than any of its ancestor
                # and become the overall best bode, the max is computed with the parent index
                child_index: float = max(local_child_index, parent_index)

                # the index of the child node is updated now
                # as a child node can have multiple parents we take the min if an index was previously computed
                if child_node.exploration_manager.index is None:
                    child_node.exploration_manager.index = child_index
                else:
                    child_node.exploration_manager.index = min(child_node.exploration_manager.index, child_index)


def update_all_indices_yoshio(
        all_nodes_not_opened: trees.RangedDescendants
) -> None:
    """
    The idea is to compute an index $ind(n)$ for a node $n$ that measures

    Args:
        all_nodes_not_opened: a tree composed only of the nodes not yer opened by the algorithm

    Returns:

    """
    half_move: int
    for half_move in all_nodes_not_opened:
        parent_node: nodes.AlgorithmNode
        for parent_node in all_nodes_not_opened[half_move].values():
            child_node: nodes.AlgorithmNode
            for child_node in parent_node.moves_children.values():
                parent_index: float = parent_node.exploration_manager.index
                child_value: float = child_node.minmax_evaluation.get_value_white()
                parent_value: float = parent_node.minmax_evaluation.get_value_white()

                # computes local_child_index the amount of change for the child node to become better than its parent
                sorted_not_over_children = parent_node.minmax_evaluation.sort_children_not_over()
                local_child_index: float = abs(child_value - parent_value) / 2

                # the amount of change for the child to become better than any of its ancestor
                # and become the overall best bode, the max is computed with the parent index
                child_index: float = max(local_child_index, parent_index)

                # the index of the child node is updated now
                # as a child node can have multiple parents we take the min if an index was previously computed
                if child_node.exploration_manager.index is None:
                    child_node.exploration_manager.index = child_index
                else:
                    child_node.exploration_manager.index = min(child_node.exploration_manager.index, child_index)


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
