from .algorithm_node import AlgorithmNode
from .itree_node import ITreeNode


def get_descendants(
        from_tree_node: ITreeNode
):
    des = {from_tree_node: None}  # include itself
    generation = set(from_tree_node.moves_children.values())
    while generation:
        next_depth_generation = set()
        for node in generation:
            assert node is not None
            des[node] = None
            for move, next_generation_child in node.moves_children.items():
                next_depth_generation.add(next_generation_child)
        generation = next_depth_generation
    return des


def get_descendants_candidate_to_open(
        from_tree_node: AlgorithmNode,
        max_depth: int | None = None
) -> list[AlgorithmNode]:
    """ returns descendants that are not over"""
    if not from_tree_node.all_legal_moves_generated and not from_tree_node.is_over():
        # should use are_all_moves_and_children_opened() but its messy!
        # also using is_over is  messy as over_events are defined in a child class!!!
        des = {from_tree_node: None}  # include itself maybe
    else:
        des = {}
    generation = set(from_tree_node.tree_node.moves_children_.values())
    depth: int = 1
    assert max_depth is not None
    while generation and depth <= max_depth:
        next_depth_generation = set()
        for node in generation:
            assert isinstance(node, AlgorithmNode)
            if not node.all_legal_moves_generated and not node.is_over():
                des[node] = None
            for move, next_generation_child in node.moves_children.items():
                next_depth_generation.add(next_generation_child)
        generation = next_depth_generation
    return list(des.keys())


def get_descendants_candidate_not_over(
        from_tree_node: AlgorithmNode,
        max_depth: int | None = None
) -> list[AlgorithmNode]:
    """ returns descendants that are not over
    returns himself if not opened"""
    assert (not from_tree_node.is_over())
    if not from_tree_node.moves_children:
        return [from_tree_node]
    des: dict[AlgorithmNode, None] = {}
    generation = set(from_tree_node.tree_node.moves_children_.values())

    depth: int = 1
    assert max_depth is not None
    while generation and depth <= max_depth:

        next_depth_generation = set()
        for node in generation:
            assert isinstance(node, AlgorithmNode)
            if not node.is_over():
                des[node] = None
            for move, next_generation_child in node.moves_children.items():
                next_depth_generation.add(next_generation_child)
        generation = next_depth_generation
    return list(des.keys())
