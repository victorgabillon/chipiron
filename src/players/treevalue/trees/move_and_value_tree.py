import chess
import src.chessenvironment.board as board_mod
from graphviz import Digraph
from src.players.treevalue.nodes.tree_node import TreeNode
from src.players.treevalue.nodes.tree_node_with_values import TreeNodeWithValue
import pickle
from src.players.treevalue.trees.updates import UpdateInstructionsBatch
from src.players.treevalue.node_factory.factory import TreeNodeFactory


# todo should we use a discount? and discounted per round reward?
# todo maybe convenient to seperate this object into openner updater and dsiplayer
# todo have the reward with a discount
# DISCOUNT = 1/.99999


class MoveAndValueTree:
    """
    This class defines the Tree that is build out of all the combinations of moves given a starting board position.
    The root node contains the starting board.
    Each node contains a board and has as many children node as there are legal move in the board.
    A children node then contains the board that is obtained by playing a particular moves in the board of the parent
    node.
    """

    def __init__(self,
                 board_evaluator,
                 starting_board: board_mod.IBoard = None) -> None:
        """

        Args:
            board_evaluator (object):
        """
        if starting_board is not None:  # for tree visualizer...
            # the tree is built at half_move  self.half_move
            self.tree_root_half_move = starting_board.ply()

        # number of nodes in the tree
        self.nodes_count = 0

        # integer counting the number of moves in the tree.
        # the interest of self.move_count over the number of nodes in the descendants
        # is that is always increasing at each opening,
        # while self.node_count can stay the same if the nodes already existed.
        self.move_count = 0

        self.board_evaluator = board_evaluator

        # to be defined later ...
        self.root_node = None
        self.descendants = None

    def node_depth(self, node):
        return node.half_move - self.tree_root_half_move


class MoveAndValueTreeBuilder:

    def __init__(self,
                 node_factory: TreeNodeFactory) -> None:
        """

        Args:

        """

        self.node_factory = node_factory

    def add_root_node(self, board, tree):
        # creates the node
        tree.root_node = self.find_or_create_node(tree=tree,
                                                  board=board,
                                                  modifications=None,
                                                  half_move=tree.tree_root_half_move,
                                                  parent_node=None)
        # ask for its evaluation
        tree.board_evaluator.evaluate_all_queried_nodes()
        tree.descendants = tree.root_node.descendants

    def create_tree_node(self,
                         tree,
                         board,
                         board_modifications,
                         half_move: int,
                         parent_node: TreeNode) -> TreeNode:
        board_depth: int = half_move - tree.tree_root_half_move
        new_node: TreeNode = self.node_factory.create(board=board,
                                                      half_move=half_move,
                                                      count=tree.nodes_count,
                                                      parent_node=parent_node,
                                                      board_depth=board_depth)
        tree.nodes_count += 1
        tree.board_evaluator.compute_representation(new_node, parent_node, board_modifications)
        tree.board_evaluator.add_evaluation_query(new_node)

        # self.updater.update_after_node_creation(new_node, parent_node)
        return new_node

    def find_or_create_node(self,
                            tree,
                            board: board_mod.IBoard,
                            modifications: board_mod.BoardModification,
                            half_move: int,
                            parent_node: TreeNode) -> TreeNode:
        fast_rep: str = board.fast_representation()
        node: TreeNode
        if tree.root_node is not None:
            print('hal',half_move,tree.root_node.descendants.descendants_at_half_move)
        if tree.root_node is None \
                or tree.root_node.descendants.is_new_generation(half_move) \
                or fast_rep not in tree.root_node.descendants.descendants_at_half_move[half_move]:
            node = self.create_tree_node(tree=tree,
                                         board=board,
                                         board_modifications=modifications, half_move=half_move,
                                         parent_node=parent_node)
            print('oo')
        else:  # the node already exists
            node = tree.descendants[half_move][fast_rep]  # add it to the list of descendants
            node.add_parent(parent_node)
        #    self.updater.update_after_link_creation(node, parent_node)
        # self.updater.update_after_either_node_or_link_creation(node, parent_node)
        tree.move_count += 1  # counting moves
        return node

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

    def display(self, format):
        dot = Digraph(format=format)
        self.add_dot(dot, self.root_node)
        return dot

    def save_pdf_to_file(self):
        dot = self.display('pdf')
        round_ = len(self.root_node.board.move_stack) + 2
        color = 'white' if self.root_node.player_to_move else 'black'
        dot.render('chipiron/runs/treedisplays/TreeVisual_' + str(int(round_ / 2)) + color + '.pdf')

    def save_raw_data_to_file(self, count='#'):
        round_ = len(self.root_node.board.move_stack) + 2
        color = 'white' if self.root_node.player_to_move else 'black'
        filename = 'chipiron/runs/treedisplays/TreeData_' + str(int(round_ / 2)) + color + '-' + str(count) + '.td'

        import sys
        sys.setrecursionlimit(100000)
        with open(filename, "wb") as f:
            pickle.dump([self.descendants, self.root_node], f)

    def print_some_stats(self):
        print('Tree stats: move_count', self.move_count, ' node_count', self.root_node.descendants.get_count())
        sum_ = 0
        self.root_node.descendants.print_stats()
        for half_move in self.root_node.descendants:
            sum_ += len(self.root_node.descendants[half_move])
            print('half_move', half_move, len(self.root_node.descendants[half_move]), sum_)

    def print_parents(self, node):
        node_to_print = node
        while node_to_print:
            parents = list(node_to_print.parent_nodes.keys())
            node_to_print = parents[0]

    def print_best_line(self):
        self.root_node.print_best_line()

    def open_node_move(self,
                       tree,
                       parent_node: TreeNodeWithValue,
                       move: chess.Move) -> object:
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
        child_node: TreeNodeWithValue = self.find_or_create_node(tree=tree,
                                                                 board=board,
                                                                 modifications=modifications,
                                                                 half_move=parent_node.half_move + 1,
                                                                 parent_node=parent_node)

        # add it to the list of opened move and out of the non-opened moves
        parent_node.moves_children[move] = child_node
        parent_node.non_opened_legal_moves.remove(move)
        parent_node.children_not_over.append(child_node)  # default action checks for over event are performed later
        update_instructions = child_node.create_update_instructions_after_node_birth()
        # update_instructions_batch is key sorted dict, sorted by depth to ensure proper backprop from the back
        update_instructions_batch = UpdateInstructionsBatch({parent_node: update_instructions})
        return update_instructions_batch

    def batch_opening(self,
                      tree,
                      opening_instructions_batch,
                      board_evaluator):
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

        board_evaluator.evaluate_all_queried_nodes()

        return update_instructions_batch  # todo never new_nodes used no?

    def update_backward(self,
                        update_instructions_batch):
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

    def open_and_update(self,
                        tree,
                        opening_instructions_batch,
                        board_evaluator):  # set of nodes and moves to open
        # first open
        update_instructions_batch = self.batch_opening(tree=tree,
                                                       opening_instructions_batch=opening_instructions_batch,
                                                       board_evaluator=board_evaluator)
        # then updates
        self.update_backward(update_instructions_batch)
