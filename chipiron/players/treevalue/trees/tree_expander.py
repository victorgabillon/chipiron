from chipiron.players.treevalue.trees.move_and_value_tree import MoveAndValueTree


class TreeExpander:

    tree: MoveAndValueTree

    def __int__(self,tree):
        self.tree = tree


    def create_tree_node(self,
                         board: board_mod.IBoard,
                         board_modifications,
                         half_move: int,
                         parent_node) -> TreeNode:
        board_depth: int = half_move - self.tree_root_half_move
        new_node: TreeNode = self.node_factory.create(board=board,
                                                      half_move=half_move,
                                                      count=self.nodes_count,
                                                      parent_node=parent_node,
                                                      board_depth=board_depth)
        self.nodes_count += 1
        self.board_evaluator.compute_representation(new_node, parent_node, board_modifications)
        self.board_evaluator.add_evaluation_query(new_node)
        return new_node

    def create_tree_move(self,
                         board: board_mod.IBoard,
                         half_move: int,
                         fast_rep: str
                         ):
        node = self.descendants[half_move][fast_rep]  # add it to the list of descendants
        node.add_parent(parent_node)
        self.updater.update_after_link_creation(node, parent_node)