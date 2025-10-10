"""
Interface for a chess board.
"""

import typing
from dataclasses import asdict
from typing import Any, Iterator, Protocol, Self

import chess
import valanga
import yaml

from chipiron.environments.chess_env.board.board_modification import BoardModificationP
from chipiron.environments.chess_env.move import moveUci
from chipiron.environments.chess_env.move.imove import moveKey

from .utils import FenPlusHistory, FenPlusMoveHistory, fen

# identifier that should be unique to any position
boardKey = tuple[
    int, int, int, int, int, int, bool, int, int | None, int, int, int, int, int
]

# identifier that removes the info (such as rounds) to count easily repeating position at difference round of the game
boardKeyWithoutCounters = tuple[
    int, int, int, int, int, int, bool, int, int | None, int, int, int
]


class LegalMoveKeyGeneratorP(Protocol):
    """Protocol for a legal move generator that yields move keys."""

    all_generated_keys: list[moveKey] | None
    # whether to sort the legal_moves by their respective uci for easy comparison of various implementations
    sort_legal_moves: bool = False

    def __iter__(self) -> Iterator[moveKey]:
        """Returns an iterator over the legal move keys."""
        ...

    def __next__(self) -> moveKey:
        """Returns the next legal move key."""
        ...

    def more_than_one_move(self) -> bool:
        """Checks if there is more than one legal move available.

        Returns:
            bool: True if there is more than one legal move, False otherwise.
        """
        ...

    def get_all(self) -> list[moveKey]:
        """Returns a list of all legal move keys."""
        ...

    def get_uci_from_move_key(self, move_key: moveKey) -> moveUci:
        """Returns the UCI string corresponding to the given move key.
        Args:
            move_key (moveKey): The move key to convert to UCI.
        Returns:
            moveUci: The UCI string corresponding to the given move key."""
        ...

    def copy_with_reset(self) -> Self:
        """Creates a copy of the legal move generator with an optional reset of generated moves.

        Returns:
            Self: A new instance of the legal move generator with the specified generated moves.
        """
        ...

    @property
    def fen(self) -> fen:
        """Returns the FEN string representation of the board."""
        ...


def compute_key(
    pawns: int,
    knights: int,
    bishops: int,
    rooks: int,
    queens: int,
    kings: int,
    turn: bool,
    castling_rights: int,
    ep_square: int | None,
    white: int,
    black: int,
    promoted: int,
    fullmove_number: int,
    halfmove_clock: int,
) -> boardKey:
    """
    Computes and returns a unique key representing the current state of the chess board.

    The key is computed by concatenating various attributes of the board, including the positions of pawns, knights,
    bishops, rooks, queens, and kings, as well as the current turn, castling rights, en passant square, halfmove clock,
    occupied squares for each color, promoted pieces, and the fullmove number.
    It is faster than calling the fen.
    Returns:
        str: A unique key representing the current state of the chess board.
    """
    string: boardKey = (
        pawns,
        knights,
        bishops,
        rooks,
        queens,
        kings,
        turn,
        castling_rights,
        ep_square,
        white,
        black,
        promoted,
        fullmove_number,
        halfmove_clock,
    )
    return string


# Note that we do not use Dict[Square, Piece] because of the rust version that would need to transform
# tuple[chess.PieceType, chess.Color] into Piece and would lose time
PieceMap = typing.Annotated[
    dict[chess.Square, tuple[chess.PieceType, chess.Color]],
    "a dictionary that list the pieces on the board",
]


class IBoard(Protocol):
    """Interface for a chess board."""

    fast_representation_: boardKey

    def get_uci_from_move_key(self, move_key: moveKey) -> moveUci:
        """Returns the UCI string corresponding to the given move key.
        Args:
            move_key (moveKey): The move key to convert to UCI.
        Returns:
            moveUci: The UCI string corresponding to the given move key."""
        return self.legal_moves.get_uci_from_move_key(move_key)

    def get_move_key_from_uci(self, move_uci: moveUci) -> moveKey:
        """Returns the move key corresponding to the given UCI string.
        Args:
            move_uci (moveUci): The UCI string to convert to a move key.
        Returns:
            moveKey: The move key corresponding to the given UCI string.
            Raises:
                KeyError: If the UCI string is not found in the legal moves.
        """
        number_moves: int = len(self.legal_moves.get_all())
        i: int

        for i in range(number_moves):
            if self.legal_moves.get_uci_from_move_key(i) == move_uci:
                return i

        raise KeyError(
            "the code should not have reached this point: problem with"
            " legal moves / uci relation in boards object it seems"
        )

    def play_move_key(self, move: moveKey) -> BoardModificationP | None:
        """Plays the move corresponding to the given move key.
        Args:
            move (moveKey): The move key to play.
        Returns:
            BoardModificationP | None: The result of the move, or None if the move is illegal.
        """
        ...

    def play_move_uci(self, move_uci: moveUci) -> BoardModificationP | None:
        """Plays the move corresponding to the given UCI string.
        Args:
            move_uci (moveUci): The UCI string to play.
        Returns:
            BoardModificationP | None: The result of the move, or None if the move is illegal.
        """
        ...

    @property
    def fen(self) -> fen:
        """Returns the FEN string representation of the board.
        Returns:
            fen: The FEN string representation of the board."""
        ...

    @property
    def move_history_stack(
        self,
    ) -> list[moveUci]:
        """Returns the move history stack.
        Returns:
            list[moveUci]: The move history stack.
        """
        ...

    def ply(self) -> int:
        """
        Returns the number of half-moves (plies) that have been played on the board.

        :return: The number of half-moves played on the board.
        :rtype: int
        """
        ...

    @property
    def turn(self) -> valanga.Colors:
        """
        Get the current turn color.

        Returns:
            chess.Color: The color of the current turn.
        """
        ...

    def copy(self, stack: bool, deep_copy_legal_moves: bool = True) -> Self:
        """
        Create a copy of the current board.

        Args:
            stack (bool): Whether to copy the move stack as well.

        Returns:
            BoardChi: A new instance of the BoardChi class with the copied board.
        """
        ...

    def is_game_over(self) -> bool:
        """
        Check if the game is over.

        Returns:
            bool: True if the game is over, False otherwise.
        """
        ...

    @property
    def pawns(self) -> chess.Bitboard: ...

    @property
    def knights(self) -> chess.Bitboard: ...

    @property
    def bishops(self) -> chess.Bitboard: ...

    @property
    def rooks(self) -> chess.Bitboard: ...

    @property
    def queens(self) -> chess.Bitboard: ...

    @property
    def kings(self) -> chess.Bitboard: ...

    @property
    def white(self) -> chess.Bitboard: ...

    @property
    def black(self) -> chess.Bitboard: ...

    @property
    def halfmove_clock(self) -> int: ...

    @property
    def promoted(self) -> chess.Bitboard: ...

    @property
    def fullmove_number(self) -> int: ...

    @property
    def castling_rights(self) -> chess.Bitboard: ...

    @property
    def occupied(self) -> chess.Bitboard: ...

    def occupied_color(self, color: chess.Color) -> chess.Bitboard: ...

    def result(self, claim_draw: bool = False) -> str: ...

    def termination(self) -> chess.Termination | None: ...

    def dump(self, file: Any) -> None:
        # create minimal info for reconstruction that is the class FenPlusMoveHistory

        current_fen: fen = self.fen
        fen_plus_moves: FenPlusMoveHistory = FenPlusMoveHistory(
            current_fen=current_fen, historical_moves=self.move_history_stack
        )

        yaml.dump(asdict(fen_plus_moves), file, default_flow_style=False)

    @property
    def ep_square(self) -> int | None:
        """Returns the en passant square if it exists, otherwise None."""
        ...

    @property
    def fast_representation(self) -> boardKey:
        """
        Returns a fast representation of the board.

        This method computes and returns a string representation of the board
        that can be quickly generated and used for various purposes.

        :return: A string representation of the board.
        :rtype: str
        """
        return self.fast_representation_

    @property
    def fast_representation_without_counters(self) -> boardKeyWithoutCounters:
        """
        Returns a fast representation of the board.

        This method computes and returns a string representation of the board
        that can be quickly generated and used for various purposes.

        :return: A string representation of the board.
        :rtype: str
        """
        assert self.fast_representation_ is not None
        return self.fast_representation_[:-2]

    def is_zeroing(self, move: moveKey) -> bool:
        """Check if a move is a zeroing move (i.e., checks if the given move is a capture or pawn move.

        Args:
            move (moveKey): The move to check.

        Returns:
            bool: True if the move is a zeroing move, False otherwise.
        """
        ...

    def is_attacked(self, a_color: chess.Color) -> bool: ...

    @property
    def legal_moves(self) -> LegalMoveKeyGeneratorP: ...

    def number_of_pieces_on_the_board(self) -> int: ...

    def piece_map(self) -> dict[chess.Square, tuple[chess.PieceType, chess.Color]]: ...

    def has_kingside_castling_rights(self, color: chess.Color) -> bool: ...

    def has_queenside_castling_rights(self, color: chess.Color) -> bool: ...

    def print_chess_board(self) -> str: ...

    def tell_result(self) -> None: ...

    def into_fen_plus_history(self) -> FenPlusHistory: ...
