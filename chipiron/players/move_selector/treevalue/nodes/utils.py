import chess

from .itree_node import ITreeNode
from .node_minmax_evaluation import NodeMinmaxEvaluation
from .tree_node import TreeNode


def are_all_moves_and_children_opened(
        tree_node: TreeNode
) -> bool:
    return tree_node.all_legal_moves_generated and tree_node.non_opened_legal_moves == set()


def a_move_sequence_from_root(
        tree_node: ITreeNode
) -> list[str]:
    move_sequence_from_root: list[chess.Move] = []
    child: ITreeNode = tree_node
    while child.parent_nodes:
        parent: ITreeNode = next(iter(child.parent_nodes))
        move_sequence_from_root.append(parent.moves_children.inverse[child])
        child = parent
    move_sequence_from_root.reverse()
    return [str(i) for i in move_sequence_from_root]


def print_a_move_sequence_from_root(
        tree_node: TreeNode
) -> None:
    move_sequence_from_root: list[str] = a_move_sequence_from_root(tree_node=tree_node)
    print(f'a_move_sequence_from_root{move_sequence_from_root}')


def is_winning(
        node_minmax_evaluation: NodeMinmaxEvaluation,
        color: chess.Color
) -> bool:
    """ return if the color to play in the node is winning """
    assert node_minmax_evaluation.value_white_minmax is not None
    winning_if_color_white: bool = node_minmax_evaluation.value_white_minmax > .98 and color
    winning_if_color_black: bool = node_minmax_evaluation.value_white_minmax < -.98 and not color

    return winning_if_color_white or winning_if_color_black
