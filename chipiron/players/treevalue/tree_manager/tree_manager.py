import chess
import chipiron.chessenvironment.board as board_mod
from graphviz import Digraph
from chipiron.players.treevalue.nodes.tree_node import TreeNode
from chipiron.players.treevalue.nodes.tree_node_with_values import TreeNodeWithValue
import pickle
from chipiron.players.treevalue.updates import UpdateInstructionsBatch
from chipiron.players.treevalue.trees.move_and_value_tree import MoveAndValueTree
from chipiron.players.treevalue.tree_manager.tree_expander import TreeExpander


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
                            tree: MoveAndValueTree,
                            modifications: board_mod.BoardModification,
                            half_move: int,
                            parent_node: TreeNode) -> TreeNode:

        fast_rep: str = board.fast_representation()
        node: TreeNode

        if tree.root_node is None \
                or tree.root_node.descendants.is_new_generation(half_move) \
                or fast_rep not in tree.root_node.descendants.descendants_at_half_move[half_move]:
            node = self.tree_expander.create_tree_node(tree=tree,
                                                       board=board,
                                                       board_modifications=modifications,
                                                       half_move=half_move,
                                                       parent_node=parent_node)
            tree.root_node.descendants.add_descendant(node)
        else:  # the node already exists
            node = self.tree_expander.create_tree_move(tree=tree,
                                                       half_move=half_move,
                                                       fast_rep=fast_rep,
                                                       parent_node=parent_node)
        return node

    def open_node_move(self,
                       tree, parent_node: TreeNodeWithValue, move: chess.Move) -> object:
        """
        Opening a Node that contains a board following a move.
        Args:
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
        child_node: TreeNode = self.find_or_create_node(board=board,
                                                        tree=tree,
                                                        modifications=modifications,
                                                        half_move=parent_node.half_move + 1,
                                                        parent_node=parent_node)

        # add it to the list of opened move and out of the non-opened moves
        parent_node.moves_children[move] = child_node
        parent_node.non_opened_legal_moves.remove(move)
        tree.move_count += 1  # counting moves
        parent_node.children_not_over.append(child_node)  # default action checks for over event are performed later

        update_instructions = child_node.create_update_instructions_after_node_birth()
        # update_instructions_batch is key sorted dict, sorted by depth to ensure proper backprop from the back
        update_instructions_batch = UpdateInstructionsBatch({parent_node: update_instructions})
        return update_instructions_batch

    def add_dot(self, dot: object, treenode: object) -> object:

        nd = treenode.dot_description()
        dot.node(str(treenode.id), nd)
        for ind, move in enumerate(treenode.moves_children):
            if treenode.moves_children[move] is not None:
                child = treenode.moves_children[move]
                cdd = str(child.id)
                dot.edge(str(treenode.id), cdd, str(move.uci()))
                self.add_dot(dot, child)

    def display_special(self, node, format, index):
        dot = Digraph(format=format)
        print(';;;', type(node))
        nd = node.dot_description()
        dot.node(str(node.id), nd)
        sorted_moves = [(str(move), move) for move in node.moves_children.keys()]
        sorted_moves.sort()
        for move_key in sorted_moves:
            move = move_key[1]
            child = node.moves_children[move]
            if node.moves_children[move] is not None:
                child = node.moves_children[move]
                cdd = str(child.id)
                edge_description = index[move] + '|' + str(move.uci()) + '|' + node.description_tree_visualizer_move(
                    child)
                dot.edge(str(node.id), cdd, edge_description)
                dot.node(str(child.id), child.dot_description())
                print('--move:', edge_description)
                print('--child:', child.dot_description())
        return dot

    def display(self,
                tree, format):
        dot = Digraph(format=format)
        self.add_dot(dot, tree.root_node)
        return dot

    def save_pdf_to_file(self,
                         tree):
        dot = self.display('pdf')
        round_ = len(tree.root_node.board.move_stack) + 2
        color = 'white' if tree.root_node.player_to_move else 'black'
        dot.render('chipiron/runs/treedisplays/TreeVisual_' + str(int(round_ / 2)) + color + '.pdf')

    def save_raw_data_to_file(self,
                              tree, count='#'):
        round_ = len(tree.root_node.board.move_stack) + 2
        color = 'white' if tree.root_node.player_to_move else 'black'
        filename = 'chipiron/runs/treedisplays/TreeData_' + str(int(round_ / 2)) + color + '-' + str(count) + '.td'

        import sys
        sys.setrecursionlimit(100000)
        with open(filename, "wb") as f:
            pickle.dump([tree.descendants, tree.root_node], f)

    def open_and_update(self,
                        tree,
                        opening_instructions_batch):  # set of nodes and moves to open
        # first open
        update_instructions_batch = self.batch_opening(tree=tree,
                                                       opening_instructions_batch=opening_instructions_batch)
        # then updates
        self.update_backward(update_instructions_batch)

    def batch_opening(self, tree, opening_instructions_batch):

        # update_instructions_batch is key sorted dict, sorted by depth to ensure proper backprop from the back

        # place to store the update instructions generated by the openings
        update_instructions_batch = UpdateInstructionsBatch()

        for opening_instructions in opening_instructions_batch.values():
            # open
            new_update_instructions_batch = self.open_node_move(tree=tree,
                                                                parent_node=opening_instructions.node_to_open,
                                                                move=opening_instructions.move_to_play)

            # concatenate the update instructions
            update_instructions_batch.merge(new_update_instructions_batch)

        self.tree_expander.board_evaluator.evaluate_all_queried_nodes()

        return update_instructions_batch  # todo never new_nodes used no?

    def update_backward(self, update_instructions_batch):

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
