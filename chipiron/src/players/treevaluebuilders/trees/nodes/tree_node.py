from bidict import bidict


# todo check if the transfer to half move is done from depth

class TreeNode:

    def __init__(self, board, half_move, id_number, parent_node):
        # id is a number to identify this node for easier debug
        self.id = id_number

        # the node represents a board position. we also store the fast representation of the board.
        self.board = board
        self.fast_rep = board.fast_representation()

        # number of half-moves since the start of the game to get to the board position in self.board
        self.half_move = half_move
        assert (isinstance(half_move, int))

        # the color of the player that has to move in the board
        self.player_to_move = self.board.chess_board.turn

        # bijection dictionary between moves and children nodes. node is set to None is not created
        self.moves_children = bidict({})

        # the set of parent nodes to this node. Note that a node can have multiple parents!
        self.parent_nodes = {parent_node}

        # all_legal_moves_generated  is a boolean saying whether all moves have been generated.
        # If true the moves are either opened in which case the corresponding opened node is stored in
        # the dictionary self.moves_children, otherwise it is stored in self.non_opened_legal_moves
        self.all_legal_moves_generated = False
        self.non_opened_legal_moves = set()

    def add_parent(self, new_parent_node):
        assert (new_parent_node not in self.parent_nodes)  # there cannot be two ways to link the same child-parent
        self.parent_nodes.add(new_parent_node)

    def print_moves_children(self):
        print('here are the ', len(self.moves_children), ' moves-children link of node', self.id, ': ', end=' ')
        for move, child in self.moves_children.items():
            if child is None:
                print(move, child, end=' ')
            else:
                print(move, child.id, end=' ')
        print(' ')

    def a_move_sequence_from_root(self):
        move_sequence_from_root = []
        child = self
        parent = next(iter(child.parent_nodes))
        while parent is not None:
            parent = next(iter(child.parent_nodes))
           # print('~~',parent.moves_children)
            move_sequence_from_root.append(parent.moves_children.inverse[child])
            child = parent
            parent = next(iter(child.parent_nodes))
        move_sequence_from_root.reverse()
        return [str(i) for i in move_sequence_from_root]

    def print_a_move_sequence_from_root(self):
        move_sequence_from_root = self.a_move_sequence_from_root()
        print('a_move_sequence_from_root', move_sequence_from_root)

    def are_all_moves_and_children_opened(self):
        return self.all_legal_moves_generated and self.non_opened_legal_moves == set()

    def test(self):
        # print('testing node', self.id)
        self.test_all_legal_moves_generated()

    def dot_description(self):
        return 'id:' + str(self.id) + ' dep: ' + str(self.half_move) + '\nfen:'+str(self.board.chess_board)

    def test_all_legal_moves_generated(self):
        # print('test_all_legal_moves_generated')
        if self.all_legal_moves_generated:
            for move in self.board.get_legal_moves():
                assert (bool(move in self.moves_children) != bool(move in self.non_opened_legal_moves))
        else:
            move_not_in = []
            legal_moves = list(self.board.get_legal_moves())
            for move in legal_moves:
                if move not in self.moves_children:
                    move_not_in.append(move)
            if move_not_in == []:
                pass
                # print('test', move_not_in, list(self.board.get_legal_moves()), self.moves_children)
                # print(self.board.chess_board)
            assert (move_not_in != [] or legal_moves == [])

    def get_descendants(self):
        des = {self: None}  # include itself
        generation = set(self.moves_children.values())
        while generation:
            next_depth_generation = set()
            for node in generation:

                des[node] = None
                for move, next_generation_child in node.moves_children.items():
                    next_depth_generation.add(next_generation_child)
            generation = next_depth_generation
        return des

    def get_descendants_candidate_to_open(self):
        """ returns descendants that are both not opened and not over"""
        if self.is_over(): #this is messy as over is defined in a child class!!!
            return {}
        if not self.all_legal_moves_generated:  # should use are_all_moves_and_children_opened() but its messy!
            des = {self: None}  # include itself maybe
        else:
            des = {}
        generation = set(self.moves_children.values())
        while generation:
            next_depth_generation = set()
            for node in generation:
                if not node.all_legal_moves_generated:
                    des[node] = None
                for move, next_generation_child in node.moves_children.items():
                    if not next_generation_child.is_over():
                        next_depth_generation.add(next_generation_child)
            generation = next_depth_generation
        return des
