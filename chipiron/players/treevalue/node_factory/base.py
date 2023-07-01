from chipiron.players.treevalue.node_factory.factory import TreeNodeFactory
from chipiron.players.treevalue.nodes.tree_node import TreeNode
import chipiron.chessenvironment.board as board_mod


class Base(TreeNodeFactory):
    def create(self,
               board,
               half_move :int ,
               count :int,
               parent_node,
               board_depth,
               modifications: board_mod.BoardModification
):
        tree_node = TreeNode(
            board=board,
            half_move=half_move,
            id_number=count,
            parent_node=parent_node,
        )
        return tree_node
