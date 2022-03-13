import chess
from src.chessenvironment.board import BoardChi, BoardModification
from graphviz import Digraph
from src.players.treevaluebuilders.trees.nodes.tree_node import TreeNode
from src.players.treevaluebuilders.trees.nodes.tree_node_with_values import TreeNodeWithValue
import pickle
from src.players.treevaluebuilders.trees.updates import UpdateInstructionsBatch


# todo should we use a discount? and discounted per round reward?
# todo maybe convenient to seperate this object into openner updater and dsiplayer
# todo have the reward with a discount
# DISCOUNT = 1/.99999


class MoveAndValueTree:
    """
    This class defines the Tree that is builds out of all the combinations of moves given a starting board position.
    The root node contains the starting board.
    Each node contains a board and has as many children node as there are legal move in the board.
    A children node then contains the board that is obtained by playing a particular moves in the board of the parent
    node.
    """

    def __init__(self, board_evaluator, node_factory, starting_board=None):
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

        # the factory in charge of creating new nodes
        self.node_factory = node_factory

        # to be defined later ...
        self.root_node = None
        self.descendants = None

    def node_depth(self, node):
        return node.half_move - self.tree_root_half_move

    def add_root_node(self, board):
        # creates the node
        self.root_node = self.find_or_create_node(board, None, self.tree_root_half_move, None)
        # ask for its evaluation
        self.board_evaluator.evaluate_all_queried_nodes()
        self.descendants = self.root_node.descendants

    def create_tree_node(self, board, half_move, count, parent_node):
        return TreeNode(board, half_move, count, parent_node)

    def update_after_node_creation(self, node, parent_node):
        # print('update_after_node_creation')
        pass  # these are empty function for higher order object to do more sophisticated things upon node creation

    def update_after_link_creation(self, node, parent_node):
        # print('update_after_link_creation')

        pass  # these are empty function for higher order object to do more sophisticated things upon link creation

    def update_after_either_node_or_link_creation(self, node, parent_node):
        pass  # these are empty function for higher order object to do more sophisticated things upon node creation

    def create_tree_node_and_more(self, board, board_modifications, half_move, parent_node):
        new_node = self.create_tree_node(board, half_move, self.nodes_count, parent_node)
        self.nodes_count += 1
        self.board_evaluator.compute_representation(new_node, parent_node, board_modifications)
        self.board_evaluator.add_evaluation_query(new_node)
        self.update_after_node_creation(new_node, parent_node)
        return new_node

    def find_or_create_node(self, board: BoardChi,
                            modifications: BoardModification,
                            half_move: int,
                            parent_node: TreeNode) -> TreeNode:
        fast_rep = board.fast_representation()
        if self.root_node is not None:
            assert (self.root_node.descendants.is_in_the_acceptable_range(half_move))
        if self.root_node is None or self.root_node.descendants.is_new_generation(half_move) or fast_rep not in \
                self.root_node.descendants.descendants_at_half_move[half_move]:
            node = self.create_tree_node_and_more(board, modifications, half_move, parent_node)
        else:  # the node already exists
            node = self.descendants[half_move][fast_rep]  # add it to the list of descendants
            node.add_parent(parent_node)
            self.update_after_link_creation(node, parent_node)
        self.update_after_either_node_or_link_creation(node, parent_node)
        return node

    def open_node_move(self, parent_node: TreeNodeWithValue, move: chess.Move) -> object:
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
        copy_stack: bool = (self.node_depth(parent_node) < 2)
        board: BoardChi = parent_node.board.copy(stack=copy_stack)

        # The move is played. The board is now a new board
        modifications: BoardModification = board.play_move(move=move)

        # Creation of the child node. If the board already exited in another node, that node is returned as child_node.
        child_node: TreeNodeWithValue = self.find_or_create_node(board=board,
                                                                 modifications=modifications,
                                                                 half_move=parent_node.half_move + 1,
                                                                 parent_node=parent_node)

        # add it to the list of opened move and out of the non-opened moves
        parent_node.moves_children[move] = child_node
        parent_node.non_opened_legal_moves.remove(move)
        self.move_count += 1  # counting moves
        parent_node.children_not_over.append(child_node)  # default action checks for over event are performed later

        update_instructions = child_node.create_update_instructions_after_node_birth()
        # update_instructions_batch is key sorted dict, sorted by depth to ensure proper backprop from the back
        update_instructions_batch = UpdateInstructionsBatch({parent_node: update_instructions})
        return update_instructions_batch

    def add_dot(self, dot, treenode):

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
        # print('--------------')
        # print('--------------')
        # print('parent:', nd)
        # print('--------------')
        # print('dcvf', node.proportions)
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

    def open_and_update(self,
                        opening_instructions_batch):  # set of nodes and moves to open
        # first open
        update_instructions_batch = self.batch_opening(opening_instructions_batch)
        # then updates
        self.update_backward(update_instructions_batch)

    def batch_opening(self, opening_instructions_batch):

        # update_instructions_batch is key sorted dict, sorted by depth to ensure proper backprop from the back

        # place to store the update instructions generated by the openings
        update_instructions_batch = UpdateInstructionsBatch()

        for opening_instructions in opening_instructions_batch.values():
            # open
            new_update_instructions_batch = self.open_node_move(opening_instructions.node_to_open,
                                                                opening_instructions.move_to_play)

            # concatenate the update instructions
            update_instructions_batch.merge(new_update_instructions_batch)

        self.board_evaluator.evaluate_all_queried_nodes()

        return update_instructions_batch  # todo never new_nodes used no?

    def update_backward(self, update_instructions_batch):

        all_extra_opening_instructions_batch = set()
        while update_instructions_batch:
            node_to_update, update_instructions = update_instructions_batch.popitem()
            extra_update_instructions_batch = self.update_node(node_to_update, update_instructions)
            update_instructions_batch.merge(extra_update_instructions_batch)
        return all_extra_opening_instructions_batch

    def update_node(self, node_to_update, update_instructions):
        #        bestnextnodeid = node_to_update.best_node_sequence[0].id if node_to_update.best_node_sequence else None
        #       print('-----------OL', node_to_update.id, node_to_update.get_value_white(), node_to_update.half_move, bestnextnodeid, node_to_update.half_move)
        #      update_instructions.print_info()

        ##UPDATES
        new_update_instructions = node_to_update.perform_updates(update_instructions)

        update_instructions_batch = UpdateInstructionsBatch()
        for parent_node in node_to_update.parent_nodes:
            if parent_node is not None and not new_update_instructions.empty():  # todo is it ever empty?
                assert (parent_node not in update_instructions_batch)
                update_instructions_batch[parent_node] = new_update_instructions

        return update_instructions_batch

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

    def test_the_tree(self):
        self.test_count()
        for half_move in self.descendants:
            for fen in self.descendants[half_move]:
                node = self.descendants[half_move][fen]
                node.test()

                # todo add a test for testing if the over match what the board evaluator says!

    def test_count(self):
        assert (self.root_node.descendants.get_count() == self.nodes_count)

    def print_best_line(self):
        self.root_node.print_best_line()
