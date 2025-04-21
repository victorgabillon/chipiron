from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterator, Self

import chess
import shakmaty_python_binding

from chipiron.environments.chess.board.board_modification import (
    BoardModification,
    BoardModificationP,
    BoardModificationRust,
)
from chipiron.environments.chess.move import moveUci
from chipiron.environments.chess.move.imove import moveKey

from .iboard import (
    IBoard,
    LegalMoveKeyGeneratorP,
    boardKey,
    boardKeyWithoutCounters,
    compute_key,
)
from .utils import FenPlusHistory, fen


class LegalMoveKeyGeneratorRust(LegalMoveKeyGeneratorP):
    # whether to sort the legal_moves by their respective uci for easy comparison of various implementations
    sort_legal_moves: bool

    generated_moves: list[shakmaty_python_binding.MyMove] | None

    all_generated_keys: list[moveKey] | None

    chess_rust_binding: shakmaty_python_binding.MyChess

    def __init__(
        self,
        sort_legal_moves: bool,
        chess_rust_binding: shakmaty_python_binding.MyChess,
        generated_moves: list[shakmaty_python_binding.MyMove] | None = None,
    ):
        self.chess_rust_binding = chess_rust_binding
        self.generated_moves = generated_moves
        if generated_moves is not None:
            self.number_moves = len(generated_moves)
            self.it: Iterator[int] = iter(range(self.number_moves))
            self.all_generated_keys = list(range(self.number_moves))
            if sort_legal_moves:

                def f(i: int) -> moveUci:
                    assert self.generated_moves is not None
                    return self.generated_moves[i].uci()

                self.all_generated_keys = sorted(
                    list(range(self.number_moves)),
                    # key=lambda i: self.generated_moves[i].uci()
                    key=f,
                )
            else:
                self.all_generated_keys = list(range(self.number_moves))
        else:
            self.all_generated_keys = None
        self.sort_legal_moves = sort_legal_moves

    @property
    def fen(self) -> fen:
        return self.chess_rust_binding.fen()

    def reset(self, generated_moves: list[shakmaty_python_binding.MyMove]) -> None:
        self.generated_moves = generated_moves
        self.number_moves = len(generated_moves)
        self.it = iter(range(self.number_moves))
        self.all_generated_keys = list(range(self.number_moves))

    def copy_with_reset(
        self, generated_moves: list[shakmaty_python_binding.MyMove] | None = None
    ) -> "LegalMoveKeyGeneratorRust":
        legal_move_copy = LegalMoveKeyGeneratorRust(
            chess_rust_binding=self.chess_rust_binding,
            generated_moves=generated_moves,
            sort_legal_moves=self.sort_legal_moves,
        )
        return legal_move_copy

    def set_legal_moves(
        self, generated_moves: list[shakmaty_python_binding.MyMove]
    ) -> None:
        self.generated_moves = generated_moves
        self.number_moves = len(generated_moves)

    def __iter__(self) -> Iterator[moveKey]:
        if self.generated_moves is None:
            self.generated_moves = self.chess_rust_binding.legal_moves()
        if self.sort_legal_moves:
            assert self.generated_moves is not None

            def f(i: int) -> moveUci:
                assert self.generated_moves is not None
                return self.generated_moves[i].uci()

            self.it = iter(
                sorted(
                    list(range(self.number_moves)),
                    # key=lambda i: self.generated_moves[i].uci()
                    key=f,
                )
            )
        else:
            self.it = iter(range(self.number_moves))
        return self

    def __next__(self) -> moveKey:
        return self.it.__next__()

    def copy(
        self, copied_chess_rust_binding: shakmaty_python_binding.MyChess | None = None
    ) -> "LegalMoveKeyGeneratorRust":
        if copied_chess_rust_binding is None:
            copied_chess_rust_binding_ = self.chess_rust_binding
        else:
            copied_chess_rust_binding_ = copied_chess_rust_binding
        legal_move_copy = LegalMoveKeyGeneratorRust(
            chess_rust_binding=copied_chess_rust_binding_,
            generated_moves=(
                self.generated_moves.copy()
                if self.generated_moves is not None
                else None
            ),
            sort_legal_moves=self.sort_legal_moves,
        )
        if self.all_generated_keys is not None:
            legal_move_copy.all_generated_keys = self.all_generated_keys.copy()
        else:
            legal_move_copy.all_generated_keys = legal_move_copy.all_generated_keys
        return legal_move_copy

    def get_all(self) -> list[moveKey]:
        if self.generated_moves is None:
            self.generated_moves = self.chess_rust_binding.legal_moves()
            self.number_moves = len(self.generated_moves)
            self.all_generated_keys = None

        if self.all_generated_keys is None:
            if self.sort_legal_moves:

                def f(i: int) -> moveUci:
                    assert self.generated_moves is not None
                    return self.generated_moves[i].uci()

                s = sorted(
                    list(range(self.number_moves)),
                    # key=lambda i: self.generated_moves[i].uci()
                    key=f,
                )
                return s
            else:
                return list(range(self.number_moves))
        else:
            return self.all_generated_keys

    def more_than_one_move(self) -> bool:
        if self.generated_moves is None:
            self.generated_moves = self.chess_rust_binding.legal_moves()
            self.number_moves = len(self.generated_moves)
            self.all_generated_keys = None
        return len(self.generated_moves) > 0


# todo implement rewind (and a test for it)


@dataclass
class RustyBoardChi(IBoard):
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
    rep_to_count: Counter[boardKeyWithoutCounters]

    fast_representation_: boardKey

    # storing the info here for fast access as it seems calls to rust bingings can be costy
    pawns_: int
    kings_: int
    queens_: int
    rooks_: int
    bishops_: int
    knights_: int
    white_: int
    black_: int
    turn_: bool
    ep_square_: int | None
    promoted_: int
    castling_rights_: int

    legal_moves_: LegalMoveKeyGeneratorRust

    # the move history is kept here because shakmaty_python_binding.MyChess does not have a move stack at the moment
    move_stack: list[moveUci] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rep_to_count[self.fast_representation_without_counters] = 1

    def __str__(self) -> str:
        """
        Returns a string representation of the board.

        Returns:
            str: A string representation of the board.
        """
        return self.fen

    def play_move_old(
        self, move: shakmaty_python_binding.MyMove
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

    def play_min_2(self, move: shakmaty_python_binding.MyMove) -> None:
        # _str, ply, turn, is_game_over = self.chess_.play_and_return(move)
        (
            self.castling_rights_,
            self.pawns_,
            self.knights_,
            self.bishops_,
            self.rooks_,
            self.queens_,
            self.kings_,
            self.white_,
            self.black_,
            turn_int,
            ep_square_int,
            self.promoted_,
        ) = self.chess_.play_and_return_o(move)
        self.turn_ = bool(turn_int)
        if ep_square_int == -1:
            self.ep_square_ = None
        else:
            self.ep_square_ = ep_square_int
        # print('ar',a)

    def play_min_3(self, move: shakmaty_python_binding.MyMove) -> BoardModificationRust:
        # _str, ply, turn, is_game_over = self.chess_.play_and_return(move)
        (
            (
                self.castling_rights_,
                self.pawns_,
                self.knights_,
                self.bishops_,
                self.rooks_,
                self.queens_,
                self.kings_,
                self.white_,
                self.black_,
                turn_int,
                ep_square_int,
                self.promoted_,
            ),
            appearances,
            removals,
        ) = self.chess_.play_and_return_modifications(move)
        self.turn_ = bool(turn_int)
        if ep_square_int == -1:
            self.ep_square_ = None
        else:
            self.ep_square_ = ep_square_int
        # print('ar',a)

        board_modifications: BoardModificationRust = self.convert(appearances, removals)
        # board_modifications: BoardModification = BoardModificationRust(appearances_=appearances,removals_=removals)
        return board_modifications

    def convert(
        self,
        appearances: set[tuple[int, int, int]],
        removals: set[tuple[int, int, int]],
    ) -> BoardModificationRust:
        board_modifications: BoardModificationRust = BoardModificationRust(
            appearances_=appearances, removals_=removals
        )

        #        board_modifications: BoardModification = BoardModification(
        #            appearances={PieceInSquare(square=a[0],piece=a[1],color=bool(a[2])) for a in appearances},
        #            removals={PieceInSquare(square=r[0],piece=r[1],color=bool(r[2])) for r in removals}
        #        )
        return board_modifications

    def play_move(
        self, move: shakmaty_python_binding.MyMove
    ) -> BoardModificationP | None:
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
        board_modifications: BoardModificationRust | None = None

        if self.compute_board_modification:
            if True:
                board_modifications = self.play_min_3(move)

            # else:
            #     previous_pawns = self.pawns_
            #     previous_kings = self.kings_
            #     previous_queens = self.queens_
            #     previous_rooks = self.rooks_
            #     previous_bishops = self.bishops_
            #     previous_knights = self.knights_
            #     previous_occupied_white = self.white_
            #     previous_occupied_black = self.black_
            #
            #     self.play_min_2(move)
            #
            #     new_pawns = self.pawns_
            #     new_kings = self.kings_
            #     new_queens = self.queens_
            #     new_rooks = self.rooks_
            #     new_bishops = self.bishops_
            #     new_knights = self.knights_
            #     new_occupied_white = self.white_
            #     new_occupied_black = self.black_
            #
            #     board_modifications = compute_modifications(
            #         previous_bishops=previous_bishops,
            #         previous_pawns=previous_pawns,
            #         previous_kings=previous_kings,
            #         previous_knights=previous_knights,
            #         previous_queens=previous_queens,
            #         previous_occupied_white=previous_occupied_white,
            #         previous_rooks=previous_rooks,
            #         previous_occupied_black=previous_occupied_black,
            #         new_kings=new_kings,
            #         new_bishops=new_bishops,
            #         new_pawns=new_pawns,
            #         new_queens=new_queens,
            #         new_rooks=new_rooks,
            #         new_knights=new_knights,
            #         new_occupied_black=new_occupied_black,
            #         new_occupied_white=new_occupied_white
            #     )
        else:
            self.play_min_2(move)

        # update after move
        self.legal_moves_ = (
            self.legal_moves_.copy_with_reset()
        )  # the legals moves needs to be recomputed as the board has changed

        fast_representation: boardKey = compute_key(
            pawns=self.pawns_,
            knights=self.knights_,
            bishops=self.bishops_,
            rooks=self.rooks_,
            queens=self.queens_,
            kings=self.kings_,
            turn=self.turn_,
            castling_rights=self.castling_rights_,
            ep_square=self.ep_square_,
            white=self.white_,
            black=self.black_,
            promoted=self.promoted_,
            fullmove_number=self.chess_.fullmove_number(),
            halfmove_clock=self.chess_.halfmove_clock(),
        )
        self.fast_representation_ = fast_representation
        self.rep_to_count.update([self.fast_representation_without_counters])
        self.move_stack.append(move.uci())

        # self.turn_ = not self.turn_

        return board_modifications

    def play_move_uci(self, move_uci: moveUci) -> BoardModificationP | None:
        chess_move: shakmaty_python_binding.MyMove = shakmaty_python_binding.MyMove(
            uci=move_uci, my_chess=self.chess_
        )
        return self.play_move(move=chess_move)

    # todo look like this function might move to iboard when the dust settle
    def play_move_key(self, move: moveKey) -> BoardModificationP | None:
        assert self.legal_moves_.generated_moves is not None
        my_move: shakmaty_python_binding.MyMove = self.legal_moves_.generated_moves[
            move
        ]
        return self.play_move(move=my_move)

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
        # return bool(self.chess_.turn())
        return self.turn_

    def is_game_over(self) -> bool:
        """
        Check if the game is over.

        Returns:
            bool: True if the game is over, False otherwise.
        """
        claim_draw: bool = True if len(self.move_stack) >= 5 else False
        three_fold_repetition: bool = (
            max(self.rep_to_count.values()) > 2 if claim_draw else False
        )
        # todo check the move stack : check for repetition as the rust version not do it
        # todo remove this hasatrribute at some point
        if hasattr(self, "is_game_over_"):
            return three_fold_repetition or self.is_game_over_
        else:
            return three_fold_repetition or self.chess_.is_game_over()

    def copy(self, stack: bool, deep_copy_legal_moves: bool = True) -> Self:
        """
        Create a copy of the current board.

        Args:
            stack (bool): Whether to copy the move stack as well.

        Returns:
            RustyBoardChi: A new instance of the BoardChi class with the copied board.
        """
        chess_copy: shakmaty_python_binding.MyChess = self.chess_.copy()
        move_stack_ = self.move_stack.copy() if stack else []

        legal_moves_copy: LegalMoveKeyGeneratorRust
        if deep_copy_legal_moves:
            legal_moves_copy = self.legal_moves_.copy(
                copied_chess_rust_binding=chess_copy
            )
        else:
            legal_moves_copy = self.legal_moves_
            legal_moves_copy.chess_rust_binding = chess_copy

        return type(self)(
            chess_=chess_copy,
            move_stack=move_stack_,
            compute_board_modification=self.compute_board_modification,
            rep_to_count=self.rep_to_count.copy(),
            fast_representation_=self.fast_representation_,
            pawns_=self.pawns_,
            knights_=self.knights_,
            kings_=self.kings_,
            rooks_=self.rooks_,
            queens_=self.queens_,
            bishops_=self.bishops_,
            black_=self.black_,
            white_=self.white_,
            turn_=self.turn_,
            ep_square_=self.ep_square_,
            promoted_=self.promoted_,
            castling_rights_=self.castling_rights_,
            legal_moves_=legal_moves_copy,
        )

    @property
    def legal_moves(self) -> LegalMoveKeyGeneratorRust:
        # todo minimize this call and understand when the role of the variable all legal move generated
        return self.legal_moves_

    def number_of_pieces_on_the_board(self) -> int:
        """
        Returns the number of pieces currently on the board.

        Returns:
            int: The number of pieces on the board.
        """
        return self.chess_.number_of_pieces_on_the_board()

    @property
    def fen(self) -> fen:
        """
        Returns the Forsyth-Edwards Notation (FEN) representation of the chess board.

        :return: The FEN string representing the current state of the board.
        """
        return self.chess_.fen()

    def piece_at(self, square: chess.Square) -> chess.Piece | None:
        """
        Returns the piece at the specified square on the chess board.

        Args:
            square (chess.Square): The square on the chess board.

        Returns:
            chess.Piece | None: The piece at the specified square, or None if there is no piece.

        """
        piece_or_none = self.chess_.piece_at(square)
        piece: chess.Piece | None
        if piece_or_none is None:
            piece = None
        else:
            piece = chess.Piece(piece_type=piece_or_none[1], color=piece_or_none[0])
        return piece

    def piece_map(self) -> dict[chess.Square, tuple[int, bool]]:
        dict_raw = self.chess_.piece_map()
        return dict_raw

    def has_kingside_castling_rights(self, color: chess.Color) -> bool:
        """
        Check if the specified color has kingside castling rights.

        Args:
            color (chess.Color): The color to check for kingside castling rights.

        Returns:
            bool: True if the specified color has kingside castling rights, False otherwise.
        """
        return self.chess_.has_kingside_castling_rights(color)

    def has_queenside_castling_rights(self, color: chess.Color) -> bool:
        """
        Check if the specified color has queenside castling rights.

        Args:
            color (chess.Color): The color to check for queenside castling rights.

        Returns:
            bool: True if the specified color has kingside castling rights, False otherwise.
        """
        return self.chess_.has_queenside_castling_rights(color)

    def print_chess_board(self) -> str:
        """
        Prints the current state of the chess board.

        This method prints the current state of the chess board, including the position of all the pieces.
        It also prints the FEN (Forsyth–Edwards Notation) representation of the board.

        Returns:
            None
        """
        return str(self.chess_.fen())

    def tell_result(self) -> None: ...

    @property
    def move_history_stack(self) -> list[moveUci]:
        return self.move_stack

    def dump(self, f: Any) -> None: ...

    def is_attacked(self, a_color: chess.Color) -> bool:
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
        return self.knights_
        # return self.chess_.knights()

    @property
    def bishops(self) -> chess.Bitboard:
        return self.bishops_
        #        return self.chess_.bishops()

    @property
    def rooks(self) -> chess.Bitboard:
        return self.rooks_
        # return self.chess_.rooks()

    @property
    def queens(self) -> chess.Bitboard:
        return self.queens_

    #        return self.chess_.queens()

    @property
    def kings(self) -> chess.Bitboard:
        return self.kings_

    #        return self.chess_.kings()

    @property
    def white(self) -> chess.Bitboard:
        return self.white_
        # return self.chess_.white()

    @property
    def black(self) -> chess.Bitboard:
        return self.black_
        # return self.chess_.black()

    @property
    def occupied(self) -> chess.Bitboard:
        return self.chess_.occupied()

    def result(self, claim_draw: bool = False) -> str:

        claim_draw_: bool = True if len(self.move_stack) >= 5 and claim_draw else False
        three_fold_repetition: bool = (
            max(self.rep_to_count.values()) > 2 if claim_draw_ else False
        )

        if three_fold_repetition:
            return "1/2-1/2"
        else:
            return self.chess_.result()

    @property
    def castling_rights(self) -> chess.Bitboard:
        # return self.chess_.castling_rights()
        return self.castling_rights_

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
    def promoted(self) -> chess.Bitboard:
        return self.promoted_

    @property
    def fullmove_number(self) -> int:
        return self.chess_.fullmove_number()

    @property
    def ep_square(self) -> int | None:
        return self.ep_square_

    def is_zeroing(self, move: moveKey) -> bool:
        assert self.legal_moves_.generated_moves is not None
        chess_move: shakmaty_python_binding.MyMove = self.legal_moves_.generated_moves[
            move
        ]
        return chess_move.is_zeroing()

    def into_fen_plus_history(self) -> FenPlusHistory:
        return FenPlusHistory(
            current_fen=self.fen, historical_moves=self.move_history_stack
        )
