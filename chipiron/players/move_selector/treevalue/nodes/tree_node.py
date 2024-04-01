from dataclasses import dataclass, field

import chess
from bidict import bidict

import chipiron.environments.chess.board as boards
from .itree_node import ITreeNode


@dataclass(slots=True)
class TreeNode:
    # id is a number to identify this node for easier debug
    id_: int

    # number of half-moves since the start of the game to get to the board position in self.board
    half_move_: int

    # the node represents a board position. we also store the fast representation of the board.
    board_: boards.BoardChi

    # the set of parent nodes to this node. Note that a node can have multiple parents!
    parent_nodes_: set[ITreeNode]

    # all_legal_moves_generated  is a boolean saying whether all moves have been generated.
    # If true the moves are either opened in which case the corresponding opened node is stored in
    # the dictionary self.moves_children, otherwise it is stored in self.non_opened_legal_moves
    all_legal_moves_generated: bool = False
    non_opened_legal_moves: set[chess.Move] = field(default_factory=set)

    # bijection dictionary between moves and children nodes. node is set to None is not created
    moves_children_: bidict[chess.Move, ITreeNode | None] = field(default_factory=bidict)

    fast_rep: str = field(default_factory=str)

    # the color of the player that has to move in the board
    player_to_move_: chess.Color = field(default_factory=chess.Color)

    def __post_init__(self) -> None:
        if self.board_:
            self.fast_rep = self.board_.fast_representation()
            self.player_to_move_: chess.Color = self.board_.turn

    @property
    def id(self) -> int:
        return self.id_

    @property
    def player_to_move(self) -> chess.Color:
        return self.player_to_move_

    @property
    def board(self) -> boards.BoardChi:
        return self.board_

    @property
    def half_move(self) -> int:
        return self.half_move_

    @property
    def moves_children(self) -> bidict[chess.Move, ITreeNode | None]:
        return self.moves_children_

    @property
    def parent_nodes(self) -> set[ITreeNode]:
        return self.parent_nodes_

    def is_root_node(self) -> bool:
        return not self.parent_nodes

    # @property
    # def all_legal_moves_generated(self) -> bool:
    #    return self.all_legal_moves_generated

    def add_parent(
            self,
            new_parent_node: ITreeNode
    ) -> None:
        assert (new_parent_node not in self.parent_nodes)  # there cannot be two ways to link the same child-parent
        self.parent_nodes.add(new_parent_node)

    def print_moves_children(self) -> None:
        print('here are the ', len(self.moves_children_), ' moves-children link of node', self.id, ': ', end=' ')
        for move, child in self.moves_children_.items():
            if child is None:
                print(move, child, end=' ')
            else:
                print(move, child.id, end=' ')
        print(' ')

    def test(self) -> None:
        # print('testing node', selbestf.id)
        self.test_all_legal_moves_generated()

    def dot_description(self) -> str:
        return 'id:' + str(self.id) + ' dep: ' + str(self.half_move) + '\nfen:' + str(self.board)

    def test_all_legal_moves_generated(self) -> None:
        # print('test_all_legal_moves_generated')
        if self.all_legal_moves_generated:
            for move in self.board.legal_moves:
                assert (bool(move in self.moves_children_) != bool(move in self.non_opened_legal_moves))
        else:
            move_not_in = []
            legal_moves = list(self.board.legal_moves)
            for move in legal_moves:
                if move not in self.moves_children_:
                    move_not_in.append(move)
            if move_not_in == []:
                pass
                # print('test', move_not_in, list(self.board.get_legal_moves()), self.moves_children)
                # print(self.board)
            assert (move_not_in != [] or legal_moves == [])

    def get_descendants(self) -> dict[ITreeNode | None, None]:

        des: dict[ITreeNode | None, None] = {self: None}  # include itself
        generation = set(self.moves_children_.values())
        while generation:
            next_depth_generation = set()
            for node in generation:
                if node is not None:
                    des[node] = None
                    for move, next_generation_child in node.moves_children.items():
                        next_depth_generation.add(next_generation_child)
            generation = next_depth_generation
        return des
