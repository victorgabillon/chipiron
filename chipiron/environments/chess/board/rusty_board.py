from collections import Counter
from dataclasses import dataclass, field
from typing import Self, Any

import chess
import shakmaty_python_binding

from chipiron.environments.chess.board.board_modification import BoardModification, compute_modifications
from chipiron.environments.chess.move import moveUci
from .iboard import IBoard, board_key, board_key_without_counters


# todo implement rewind (and a test for it)

@dataclass
class RustyBoardChi(IBoard[shakmaty_python_binding.MyMove]):
    """
    Rusty Board Chipiron
    object that describes the current board. it wraps the chess Board from the chess package so it can have more in it
    but im not sure its really necessary.i keep it for potential usefulness

    This is the Rust version for speedy execution
    It is based on the binding library shakmaty_python_binding to use the rust library shakmaty
    """

    # the shakmaty implementation of the board that we wrap here
    chess_: shakmaty_python_binding.MyChess

    compute_board_modification: bool

    # to count the number of occurrence of each board to be able to compute
    # three-fold repetition as shakmaty does not do it atm
    rep_to_count: Counter[board_key_without_counters]

    fast_representation_: board_key = field(init=False)

    # the move history is kept here because shakmaty_python_binding.MyChess does not have a move stack at the moment
    move_stack: list[moveUci] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.fast_representation_ = self.compute_key()
        self.rep_to_count[self.fast_representation_without_counters] = 1

    def __str__(self) -> str:
        """
        Returns a string representation of the board.

        Returns:
            str: A string representation of the board.
        """
        return ''

    def play_move_old(
            self,
            move: shakmaty_python_binding.MyMove
    ) -> BoardModification | None:
        board_modifications: BoardModification | None

        if self.compute_board_modification:
            # board_modifications = self.chess_.push_and_return_modification(move.uci())
            ...
        else:
            self.chess_.play(move)
            board_modifications = None
        self.move_stack.append(move.uci())
        return board_modifications

    def play_min(self, move: shakmaty_python_binding.MyMove) -> None:
        self.chess_.play(move)

    def play_move(
            self,
            move: shakmaty_python_binding.MyMove
    ) -> BoardModification | None:
        """
        Plays a move on the board and returns the board modification.

        Args:
            move: The move to play.

        Returns:
            The board modification resulting from the move or None.
        """
        # todo: illegal moves seem accepted, do we care? if we dont write it in the doc
        # assert self.board.is_legal(move)
        #
        board_modifications: BoardModification | None = None

        if self.compute_board_modification:
            previous_pawns = self.chess_.pawns()
            previous_kings = self.chess_.kings()
            previous_queens = self.chess_.queens()
            previous_rooks = self.chess_.rooks()
            previous_bishops = self.chess_.bishops()
            previous_knights = self.chess_.knights()
            previous_occupied_white = self.chess_.white()
            previous_occupied_black = self.chess_.black()

            self.play_min(move)

            new_pawns = self.chess_.pawns()
            new_kings = self.chess_.kings()
            new_queens = self.chess_.queens()
            new_rooks = self.chess_.rooks()
            new_bishops = self.chess_.bishops()
            new_knights = self.chess_.knights()
            new_occupied_white = self.chess_.white()
            new_occupied_black = self.chess_.black()

            board_modifications = compute_modifications(
                previous_bishops=previous_bishops,
                previous_pawns=previous_pawns,
                previous_kings=previous_kings,
                previous_knights=previous_knights,
                previous_queens=previous_queens,
                previous_occupied_white=previous_occupied_white,
                previous_rooks=previous_rooks,
                previous_occupied_black=previous_occupied_black,
                new_kings=new_kings,
                new_bishops=new_bishops,
                new_pawns=new_pawns,
                new_queens=new_queens,
                new_rooks=new_rooks,
                new_knights=new_knights,
                new_occupied_black=new_occupied_black,
                new_occupied_white=new_occupied_white
            )
        else:
            self.play_min(move)

        # update after move
        self.legal_moves_ = None  # the legals moves needs to be recomputed as the board has changed
        fast_representation: board_key = self.compute_key()
        self.fast_representation_ = fast_representation
        self.rep_to_count.update([self.fast_representation_without_counters])
        self.move_stack.append(move.uci())
        return board_modifications

    def ply(self) -> int:
        """
        Returns the number of half-moves (plies) that have been played on the board.

        :return: The number of half-moves played on the board.
        :rtype: int
        """
        ply: int = self.chess_.ply()
        return ply

    @property
    def turn(self) -> chess.Color:
        """
        Get the current turn color.

        Returns:
            chess.Color: The color of the current turn.
        """
        return bool(self.chess_.turn())

    def is_game_over(self) -> bool:
        """
        Check if the game is over.

        Returns:
            bool: True if the game is over, False otherwise.
        """
        claim_draw: bool = True if len(self.move_stack) >= 5 else False
        three_fold_repetition: bool = max(self.rep_to_count.values()) > 2 if claim_draw else False
        # todo check the move stack : check for repetition as the rust version not do it
        return three_fold_repetition or self.chess_.is_game_over()

    def copy(
            self,
            stack: bool
    ) -> Self:
        """
        Create a copy of the current board.

        Args:
            stack (bool): Whether to copy the move stack as well.

        Returns:
            RustyBoardChi: A new instance of the BoardChi class with the copied board.
        """
        chess_copy: shakmaty_python_binding.MyChess = self.chess_.copy()
        move_stack_ = self.move_stack.copy() if stack else []
        return type(self)(
            chess_=chess_copy,
            move_stack=move_stack_,
            compute_board_modification=self.compute_board_modification,
            rep_to_count=self.rep_to_count.copy()
        )

    @property
    def legal_moves(self) -> set[shakmaty_python_binding.MyMove]:
        # todo minimize this call and understand when the role of the ariable all legal move generated
        return self.chess_.legal_moves()

    def number_of_pieces_on_the_board(self) -> int:
        """
        Returns the number of pieces currently on the board.

        Returns:
            int: The number of pieces on the board.
        """
        return self.chess_.number_of_pieces_on_the_board()

    @property
    def fen(self) -> str:
        """
        Returns the Forsyth-Edwards Notation (FEN) representation of the chess board.

        :return: The FEN string representing the current state of the board.
        """
        return self.chess_.fen()

    def piece_at(
            self,
            square: chess.Square
    ) -> chess.Piece | None:
        """
        Returns the piece at the specified square on the chess board.

        Args:
            square (chess.Square): The square on the chess board.

        Returns:
            chess.Piece | None: The piece at the specified square, or None if there is no piece.

        """
        color: bool
        role: int
        piece_or_none = self.chess_.piece_at(square)
        piece: chess.Piece | None
        if piece_or_none is None:
            piece = None
        else:
            piece = chess.Piece(piece_type=piece_or_none[1], color=piece_or_none[0])
        return piece

    def piece_map(
            self
    ) -> dict[chess.Square, tuple[int, bool]]:
        dict_raw = self.chess_.piece_map()
        return dict_raw

    def has_kingside_castling_rights(
            self,
            color: chess.Color
    ) -> bool:
        """
        Check if the specified color has kingside castling rights.

        Args:
            color (chess.Color): The color to check for kingside castling rights.

        Returns:
            bool: True if the specified color has kingside castling rights, False otherwise.
        """
        return self.chess_.has_kingside_castling_rights(color)

    def has_queenside_castling_rights(
            self,
            color: chess.Color
    ) -> bool:
        """
        Check if the specified color has queenside castling rights.

        Args:
            color (chess.Color): The color to check for queenside castling rights.

        Returns:
            bool: True if the specified color has kingside castling rights, False otherwise.
        """
        return self.chess_.has_queenside_castling_rights(color)

    def print_chess_board(self) -> None:
        """
        Prints the current state of the chess board.

        This method prints the current state of the chess board, including the position of all the pieces.
        It also prints the FEN (Forsyth–Edwards Notation) representation of the board.

        Returns:
            None
        """
        print(self.chess_.fen())

    def tell_result(self) -> None:
        ...

    @property
    def move_history_stack(self) -> list[moveUci]:
        return self.move_stack

    def dump(self, f: Any) -> None:
        ...

    def is_attacked(
            self,
            a_color: chess.Color
    ) -> bool:
        """Check if any piece of the color `a_color` is attacked.

        Args:
            a_color (chess.Color): The color of the pieces to check.

        Returns:
            bool: True if any piece of the specified color is attacked, False otherwise.
        """
        return self.chess_.is_attacked(a_color)

    @property
    def pawns(self) -> chess.Bitboard:
        return self.chess_.pawns()

    @property
    def knights(self) -> chess.Bitboard:
        return self.chess_.knights()

    @property
    def bishops(self) -> chess.Bitboard:
        return self.chess_.bishops()

    @property
    def rooks(self) -> chess.Bitboard:
        return self.chess_.rooks()

    @property
    def queens(self) -> chess.Bitboard:
        return self.chess_.queens()

    @property
    def white(self) -> chess.Bitboard:
        return self.chess_.white()

    @property
    def black(self) -> chess.Bitboard:
        return self.chess_.black()

    @property
    def occupied(self) -> chess.Bitboard:
        return self.chess_.occupied()

    def result(
            self,
            claim_draw: bool = False
    ) -> str:
        claim_draw_: bool = True if len(self.move_stack) >= 5 and claim_draw else False
        three_fold_repetition: bool = max(self.rep_to_count.values()) > 2 if claim_draw_ else False

        if three_fold_repetition:
            return '1/2-1/2'
        else:
            return self.chess_.result()

    @property
    def castling_rights(self) -> chess.Bitboard:
        return self.chess_.castling_rights()

    def termination(self) -> None:
        return None

    def occupied_color(self, color: chess.Color) -> chess.Bitboard:
        if color == chess.WHITE:
            return self.chess_.white()
        else:
            return self.chess_.black()

    @property
    def halfmove_clock(self) -> int:
        return self.chess_.halfmove_clock()

    @property
    def kings(self) -> chess.Bitboard:
        return self.chess_.kings()

    @property
    def promoted(self) -> chess.Bitboard:
        return self.chess_.promoted()

    @property
    def fullmove_number(self) -> int:
        return self.chess_.fullmove_clock()

    @property
    def ep_square(self) -> int | None:
        ep: int = self.chess_.ep_square()
        if ep == -1:
            return None
        else:
            return ep

    def is_zeroing(
            self,
            move: shakmaty_python_binding.MyMove
    ) -> bool:
        return move.is_zeroing()
