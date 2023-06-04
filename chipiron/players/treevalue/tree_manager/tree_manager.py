import chess
import chipiron.chessenvironment.board as board_mod
from chipiron.players.treevalue.nodes.tree_node_with_values import TreeNodeWithValue
from chipiron.players.treevalue.updates import UpdateInstructionsBatch
from chipiron.players.treevalue.tree_manager.tree_expander import TreeExpander, TreeExpansion, create_tree_move, \
    TreeExpansions
import chipiron.players.treevalue.nodes.tree_node as node
import chipiron.players.treevalue.trees as trees
import chipiron.players.treevalue.node_selector as node_sel


# todo should we use a discount? and discounted per round reward?
# todo maybe convenient to seperate this object into openner updater and dsiplayer
# todo have the reward with a discount
# DISCOUNT = 1/.99999


class TreeManager:
    """

    This class that and manages a tree by opening new nodes and updating the values and indexes on the nodes
    """
    tree_expander: TreeExpander

    def __init__(self,
                 tree_expander: TreeExpander
                 ) -> None:
        """
        """
        self.tree_expander = tree_expander

    def find_or_create_node(self,
                            board: board_mod.BoardChi,
                            tree: trees.MoveAndValueTree,
                            modifications: board_mod.BoardModification,
                            half_move: int,
                            parent_node: node.TreeNode) -> TreeExpansion:

        fast_rep: str = board.fast_representation()
        tree_expansion: TreeExpansion

        if tree.root_node is None \
                or tree.root_node.descendants.is_new_generation(half_move) \
                or fast_rep not in tree.root_node.descendants.descendants_at_half_move[half_move]:
            tree_expansion = self.tree_expander.create_tree_node(tree=tree,
                                                                 board=board,
                                                                 board_modifications=modifications,
                                                                 half_move=half_move,
                                                                 parent_node=parent_node)
            tree.root_node.descendants.add_descendant(tree_expansion.child_node)
        else:  # the node already exists
            tree_expansion = create_tree_move(tree=tree,
                                              half_move=half_move,
                                              fast_rep=fast_rep,
                                              parent_node=parent_node)
        return tree_expansion

    def open_node_move(self,
                       tree: trees.MoveAndValueTree,
                       parent_node: TreeNodeWithValue,
                       move: chess.Move) -> TreeExpansion:
        """
        Opening a Node that contains a board following a move.
        Args:
            tree:
            parent_node: The Parent node that we want to expand
            move: the move to play to expend the Node

        Returns:

        """
        assert (not parent_node.is_over())
        assert (move not in parent_node.moves_children)

        # The parent board is copied, we only copy the stack (history of previous board) if the depth is smaller than 2
        # Having the stack information allows checking for draw by repetition.
        # To limit computation we limit copying it all the time. The resulting policy will only be aware of immediate
        # risk of draw by repetition
        copy_stack: bool = (tree.node_depth(parent_node) < 2)
        board: board_mod.BoardChi = parent_node.board.copy(stack=copy_stack)

        # The move is played. The board is now a new board
        modifications: board_mod.BoardModification = board.play_move(move=move)

        # Creation of the child node. If the board already exited in another node, that node is returned as child_node.
        tree_expansion: TreeExpansion = self.find_or_create_node(board=board,
                                                                 tree=tree,
                                                                 modifications=modifications,
                                                                 half_move=parent_node.half_move + 1,
                                                                 parent_node=parent_node)

        # add it to the list of opened move and out of the non-opened moves
        parent_node.moves_children[move] = tree_expansion.child_node
        parent_node.non_opened_legal_moves.remove(move)
        tree.move_count += 1  # counting moves
        parent_node.children_not_over.append(
            tree_expansion.child_node)  # default action checks for over event are performed later

        return tree_expansion

    def open_and_update(self,
                        tree: trees.MoveAndValueTree,
                        opening_instructions_batch):  # set of nodes and moves to open
        # first open
        update_instructions_batch = self.batch_opening(tree=tree,
                                                       opening_instructions_batch=opening_instructions_batch)
        # then updates
        self.update_backward(update_instructions_batch)

    def open(self,
             tree: trees.MoveAndValueTree,
             opening_instructions: node_sel.OpeningInstructions) -> TreeExpansions:

        # place to store the tree expansion logs generated by the openings
        tree_expansions: TreeExpansions = TreeExpansions()

        opening_instruction : node_sel.OpeningInstruction
        for opening_instruction in opening_instructions.values():
            # open
            tree_expansion: TreeExpansion = self.open_node_move(tree=tree,
                                                                parent_node=opening_instruction.node_to_open,
                                                                move=opening_instruction.move_to_play)

            # concatenate the tree expansions
            tree_expansions.add(tree_expansion=tree_expansion)

        self.tree_expander.board_evaluator.evaluate_all_queried_nodes()

        return tree_expansions

        #    def batch_opening(self, tree, opening_instructions_batch):#

        # update_instructions_batch is key sorted dict, sorted by depth to ensure proper backprop from the back

        # place to store the update instructions generated by the openings
        #      update_instructions_batch = UpdateInstructionsBatch()

        #       for opening_instructions in opening_instructions_batch.values():
        # open
        #           new_update_instructions_batch \
        #          tree_expansion =     TreeExpansion= self.open_node_move(tree=tree,
        parent_node = opening_instructions.node_to_open,

    #   move = opening_instructions.move_to_play)

    # concatenate the update instructions
    #         update_instructions_batch.merge(new_update_instructions_batch)

    #     self.tree_expander.board_evaluator.evaluate_all_queried_nodes()

    #     return update_instructions_batch  # todo never new_nodes used no?

    def update_backward(self, tree_expansions: TreeExpansions):

        update_instructions_batch: UpdateInstructionsBatch = generate_update_instructions(
            tree_expansions=tree_expansions)


        all_extra_opening_instructions_batch = set()
        while update_instructions_batch:
            node_to_update, update_instructions = update_instructions_batch.popitem()
            extra_update_instructions_batch = self.update_node(node_to_update, update_instructions)
            update_instructions_batch.merge(extra_update_instructions_batch)
        return all_extra_opening_instructions_batch

    def update_node(self, node_to_update, update_instructions):

        ##UPDATES
        new_update_instructions = node_to_update.perform_updates(update_instructions)

        update_instructions_batch = UpdateInstructionsBatch()
        for parent_node in node_to_update.parent_nodes:
            if parent_node is not None and not new_update_instructions.empty():  # todo is it ever empty?
                assert (parent_node not in update_instructions_batch)
                update_instructions_batch[parent_node] = new_update_instructions

        return update_instructions_batch

    def print_some_stats(self,
                         tree):
        print('Tree stats: move_count', tree.move_count, ' node_count',
              tree.root_node.descendants.get_count())
        sum_ = 0
        tree.root_node.descendants.print_stats()
        for half_move in tree.root_node.descendants:
            sum_ += len(tree.root_node.descendants[half_move])
            print('half_move', half_move, len(tree.root_node.descendants[half_move]), sum_)

    def print_parents(self, node):
        node_to_print = node
        while node_to_print:
            parents = list(node_to_print.parent_nodes.keys())
            node_to_print = parents[0]

    def test_the_tree(self,
                      tree):
        self.test_count()
        for half_move in tree.descendants:
            for fen in tree.descendants[half_move]:
                node = tree.descendants[half_move][fen]
                node.test()

                # todo add a test for testing if the over match what the board evaluator says!

    def test_count(self,
                   tree):
        assert (tree.root_node.descendants.get_count() == tree.nodes_count)

    def print_best_line(self,
                        tree):
        tree.root_node.print_best_line()


def generate_update_instructions(tree_expansions: TreeExpansions) -> UpdateInstructionsBatch:
    # TODO is the way of merging now overkill?

    update_instructions_batch: UpdateInstructionsBatch = UpdateInstructionsBatch()

    tree_expansion: TreeExpansion
    for tree_expansion in tree_expansions:
        update_instructions = tree_expansion.child_node.create_update_instructions_after_node_birth()
        # update_instructions_batch is key sorted dict, sorted by depth to ensure proper backprop from the back
        new_update_instructions_batch = UpdateInstructionsBatch({tree_expansion.parent_node: update_instructions})

        # concatenate the update instructions
        update_instructions_batch.merge(new_update_instructions_batch)

    return update_instructions_batch
