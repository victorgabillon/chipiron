"""
Module that contains the BoardChi class that wraps the chess.Board class from the chess package
"""

import typing
from typing import Iterator

import chess
import chess.polyglot
from chess import Outcome, _BoardState

from chipiron.environments.chess.board.board_modification import (
    BoardModification,
    BoardModificationP,
    PieceInSquare,
    compute_modifications,
)
from chipiron.environments.chess.move import moveUci
from chipiron.environments.chess.move.imove import moveKey
from chipiron.utils.logger import chipiron_logger

from .iboard import IBoard, LegalMoveKeyGeneratorP, boardKey, compute_key
from .utils import FenPlusHistory, fen

# todo check if we need this here
COLORS = [WHITE, BLACK] = [True, False]


class LegalMoveKeyGenerator(LegalMoveKeyGeneratorP):
    generated_moves: dict[moveKey, chess.Move]
    all_generated_keys: list[moveKey] | None

    # whether to sort the legal_moves by their respective uci for easy comparison of various implementations
    sort_legal_moves: bool = False

    chess_board: chess.Board

    def __init__(self, chess_board: chess.Board, sort_legal_moves: bool) -> None:
        self.chess_board = chess_board
        self.it: Iterator[chess.Move] = self.chess_board.generate_legal_moves()
        self.generated_moves = {}
        self.sort_legal_moves = sort_legal_moves
        self.count = 0
        self.all_generated_keys = None

    @property
    def fen(self) -> fen:
        return self.chess_board.fen()

    def __str__(self) -> str:

        the_string: str = "Legals Moves: "
        ucis: list[moveUci] = [
            chess_move.uci() for move_key, chess_move in self.generated_moves.items()
        ]
        keys: list[moveKey] = [
            move_key for move_key, chess_move in self.generated_moves.items()
        ]

        the_string = the_string + f"Generated ucis {ucis}"
        the_string = the_string + f" and generated ucis {keys}"

        return the_string

    def __iter__(self) -> Iterator[moveKey]:
        self.it = self.chess_board.generate_legal_moves()
        self.count = 0
        return self

    def __next__(self) -> moveKey:
        new_move: chess.Move = self.it.__next__()
        # move_key_ = new_move.uci()
        move_key_ = self.count
        self.generated_moves[move_key_] = new_move
        self.count += 1
        return move_key_

    def copy(
        self, copied_chess_board: chess.Board | None = None
    ) -> "LegalMoveKeyGenerator":
        if copied_chess_board is None:
            copied_chess_board_ = self.chess_board
        else:
            copied_chess_board_ = copied_chess_board
        legal_move_copy = LegalMoveKeyGenerator(
            chess_board=copied_chess_board_, sort_legal_moves=self.sort_legal_moves
        )

        legal_move_copy.generated_moves = self.generated_moves.copy()
        if self.all_generated_keys is not None:
            legal_move_copy.all_generated_keys = self.all_generated_keys.copy()
        else:
            legal_move_copy.all_generated_keys = None
        legal_move_copy.count = self.count
        return legal_move_copy

    def reset(self) -> None:
        self.it = self.chess_board.generate_legal_moves()
        self.generated_moves = {}
        self.count = 0
        self.all_generated_keys = None

    def copy_with_reset(self) -> "LegalMoveKeyGenerator":
        legal_move_copy = LegalMoveKeyGenerator(
            chess_board=self.chess_board, sort_legal_moves=self.sort_legal_moves
        )
        return legal_move_copy

    def get_all(self) -> list[moveKey]:

        if self.all_generated_keys is None:
            list_keys: list[moveKey]
            if self.sort_legal_moves:
                list_keys = sorted(
                    list(self), key=lambda i: self.generated_moves[i].uci()
                )
            else:
                list_keys = list(self)
            self.all_generated_keys = list_keys
            return list_keys
        else:
            return self.all_generated_keys

    def more_than_one_move(self) -> bool:
        # assume legal_moves not empty

        iter_move: Iterator[chess.Move] = self.chess_board.generate_legal_moves()

        # remove the  first element
        next(iter_move)

        return any(iter_move)


class BoardChi(IBoard):
    """
    Board Chipiron
    object that describes the current board. it wraps the chess Board from the chess package so it can have more in it
    """

    chess_board: chess.Board
    compute_board_modification: bool
    legal_moves_: LegalMoveKeyGenerator
    fast_representation_: boardKey

    # whether to sort the legal_moves by their respective uci for easy comparison of various implementations
    sort_legal_moves: bool

    def __init__(
        self,
        chess_board: chess.Board,
        compute_board_modification: bool,
        fast_representation_: boardKey,
        legal_moves_: LegalMoveKeyGenerator,
    ) -> None:
        """
        Initializes a new instance of the BoardChi class.

        Args:
            board: The chess.Board object to wrap.
        """
        self.chess_board = chess_board
        self.compute_board_modification = compute_board_modification
        self.fast_representation_ = fast_representation_
        self.legal_moves_ = legal_moves_

    def play_mon(self, move: chess.Move) -> None:
        self.chess_board.push(move)

    def play_move(
        self, move: chess.Move, use_compute_modification_function: bool = False
    ) -> BoardModificationP | None:
        """
        Plays a move on the board and returns the board modification.

        Args:
            move: The move to play.

        Returns:

            The board modification resulting from the move or None.
        """

        # todo: illegal moves seem accepted, do we care? if we dont write it in the doc
        # assert self.chess_board.is_legal(move)
        #
        board_modifications: BoardModificationP | None = None

        if self.compute_board_modification:
            if not use_compute_modification_function:
                board_modifications = self.push_and_return_modification(move)  # type: ignore
            else:
                previous_pawns = self.chess_board.pawns
                previous_kings = self.chess_board.kings
                previous_queens = self.chess_board.queens
                previous_rooks = self.chess_board.rooks
                previous_bishops = self.chess_board.bishops
                previous_knights = self.chess_board.knights
                previous_occupied_white = self.chess_board.occupied_co[chess.WHITE]
                previous_occupied_black = self.chess_board.occupied_co[chess.BLACK]

                self.play_mon(move)
                # self.play_mon(move)

                new_pawns = self.chess_board.pawns
                new_kings = self.chess_board.kings
                new_queens = self.chess_board.queens
                new_rooks = self.chess_board.rooks
                new_bishops = self.chess_board.bishops
                new_knights = self.chess_board.knights
                new_occupied_white = self.chess_board.occupied_co[chess.WHITE]
                new_occupied_black = self.chess_board.occupied_co[chess.BLACK]

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
                    new_occupied_white=new_occupied_white,
                )
        else:
            self.chess_board.push(move)

        # update after move
        self.legal_moves_ = (
            self.legal_moves_.copy_with_reset()
        )  # the legals moves needs to be recomputed as the board has changed

        fast_representation: boardKey = compute_key(
            pawns=self.chess_board.pawns,
            knights=self.chess_board.knights,
            bishops=self.chess_board.bishops,
            rooks=self.chess_board.rooks,
            queens=self.chess_board.queens,
            kings=self.chess_board.kings,
            turn=self.chess_board.turn,
            castling_rights=self.chess_board.castling_rights,
            ep_square=self.chess_board.ep_square,
            white=self.chess_board.occupied_co[chess.WHITE],
            black=self.chess_board.occupied_co[chess.BLACK],
            promoted=self.chess_board.promoted,
            fullmove_number=self.chess_board.fullmove_number,
            halfmove_clock=self.chess_board.halfmove_clock,
        )
        self.fast_representation_ = fast_representation
        return board_modifications

    def play_move_uci(self, move_uci: moveUci) -> BoardModificationP | None:
        chess_move: chess.Move = chess.Move.from_uci(uci=move_uci)
        return self.play_move(move=chess_move)

    # todo look like this function might move to iboard when the dust settle
    def play_move_key(self, move: moveKey) -> BoardModificationP | None:
        # chess_move: chess.Move = chess.Move.from_uci(uci=move)
        # if True:
        #    if self.legal_moves_ is not None and move in self.legal_moves_.generated_moves:
        chess_move: chess.Move = self.legal_moves_.generated_moves[move]
        board_modification = self.play_move(move=chess_move)
        return board_modification

    def rewind_one_move(self) -> None:
        """
        Rewinds the board state to the previous move.
        """
        if self.ply() > 0:
            self.chess_board.pop()
        else:
            chipiron_logger.warning(
                f"Cannot rewind more as self.halfmove_clock equals {self.ply()}"
            )

    @typing.no_type_check
    def push_and_return_modification(
        self, move: chess.Move
    ) -> BoardModification | None:
        """
        Mostly reuse the push function of the chess library but records the modifications to the bitboard so that
        we can do the same with other parallel representations such as tensor in pytorch

        Args:
             move: The move to push.

        Returns:
            The board modification resulting from the move, or None if the move is a null move.
        """
        board_modifications = BoardModification()
        # Push move and remember board state
        move = self.chess_board._to_chess960(move)

        board_state = _BoardState(self.chess_board)
        self.chess_board.castling_rights = (
            self.chess_board.clean_castling_rights()
        )  # Before pushing stack
        self.chess_board.move_stack.append(
            self.chess_board._from_chess960(
                self.chess_board.chess960,
                move.from_square,
                move.to_square,
                move.promotion,
                move.drop,
            )
        )
        self.chess_board._stack.append(board_state)

        # Reset en passant square.
        ep_square = self.chess_board.ep_square
        self.chess_board.ep_square = None

        # Increment move counters.
        self.chess_board.halfmove_clock += 1
        if self.chess_board.turn == BLACK:
            self.chess_board.fullmove_number += 1

        # On a null move, simply swap turns and reset the en passant square.
        if not move:
            self.chess_board.turn = not self.chess_board.turn
            return

        # Drops.
        if move.drop:
            self.chess_board._set_piece_at(
                move.to_square, move.drop, self.chess_board.turn
            )
            self.chess_board.turn = not self.turn
            return

        # Zero the half-move clock.
        if self.chess_board.is_zeroing(move):
            self.chess_board.halfmove_clock = 0

        from_bb = chess.BB_SQUARES[move.from_square]
        to_bb = chess.BB_SQUARES[move.to_square]

        promoted = bool(self.chess_board.promoted & from_bb)
        piece_type = self.chess_board._remove_piece_at(move.from_square)
        # start added
        piece_in_square: PieceInSquare = PieceInSquare(
            square=move.from_square, piece=piece_type, color=self.chess_board.turn
        )

        board_modifications.add_removal(removal=piece_in_square)
        # end added
        assert (
            piece_type is not None
        ), f"push() expects move to be pseudo-legal, but got {move} in {self.chess_board.board_fen()}"
        capture_square = move.to_square
        captured_piece_type = self.chess_board.piece_type_at(capture_square)
        # start added
        castling = (
            piece_type == chess.KING and self.chess_board.occupied_co[self.turn] & to_bb
        )

        if captured_piece_type is not None and not castling:
            captured_piece_in_square: PieceInSquare = PieceInSquare(
                square=capture_square,
                piece=captured_piece_type,
                color=not self.chess_board.turn,
            )

            board_modifications.add_removal(removal=captured_piece_in_square)
        # end added

        # Update castling rights.
        self.chess_board.castling_rights &= ~to_bb & ~from_bb
        if piece_type == chess.KING and not promoted:
            if self.turn == WHITE:
                self.chess_board.castling_rights &= ~chess.BB_RANK_1
            else:
                self.chess_board.castling_rights &= ~chess.BB_RANK_8
        elif (
            captured_piece_type == chess.KING and not self.chess_board.promoted & to_bb
        ):
            if self.turn == WHITE and chess.square_rank(move.to_square) == 7:
                self.chess_board.castling_rights &= ~chess.BB_RANK_8
            elif self.turn == BLACK and chess.square_rank(move.to_square) == 0:
                self.chess_board.castling_rights &= ~chess.BB_RANK_1

        # Handle special pawn moves.
        if piece_type == chess.PAWN:
            diff = move.to_square - move.from_square

            if diff == 16 and chess.square_rank(move.from_square) == 1:
                self.chess_board.ep_square = move.from_square + 8
            elif diff == -16 and chess.square_rank(move.from_square) == 6:
                self.chess_board.ep_square = move.from_square - 8
            elif (
                move.to_square == ep_square
                and abs(diff) in [7, 9]
                and not captured_piece_type
            ):
                # Remove pawns captured en passant.
                down = -8 if self.turn == WHITE else 8
                capture_square = ep_square + down
                captured_piece_type = self.chess_board._remove_piece_at(capture_square)
                # start added
                pawn_captured_piece_in_square: PieceInSquare = PieceInSquare(
                    square=capture_square,
                    piece=captured_piece_type,
                    color=not self.chess_board.turn,
                )

                board_modifications.add_removal(removal=pawn_captured_piece_in_square)
                # end added

        # Promotion.
        if move.promotion:
            promoted = True
            piece_type = move.promotion

        # Castling.
        castling = (
            piece_type == chess.KING and self.chess_board.occupied_co[self.turn] & to_bb
        )
        if castling:
            a_side = chess.square_file(move.to_square) < chess.square_file(
                move.from_square
            )

            self.chess_board._remove_piece_at(move.from_square)
            self.chess_board._remove_piece_at(move.to_square)

            remove_rook_in_square: PieceInSquare = PieceInSquare(
                square=move.to_square, piece=chess.ROOK, color=self.chess_board.turn
            )
            board_modifications.add_removal(removal=remove_rook_in_square)
            # start added

            if a_side:
                king_square = (
                    chess.C1 if self.chess_board.turn == chess.WHITE else chess.C8
                )
                rook_square = chess.D1 if self.chess_board.turn == WHITE else chess.D8
                self.chess_board._set_piece_at(
                    chess.C1 if self.turn == WHITE else chess.C8, chess.KING, self.turn
                )
                self.chess_board._set_piece_at(
                    chess.D1 if self.turn == WHITE else chess.D8, chess.ROOK, self.turn
                )
            else:
                king_square = (
                    chess.G1 if self.chess_board.turn == chess.WHITE else chess.G8
                )
                rook_square = chess.F1 if self.chess_board.turn == WHITE else chess.F8
                self.chess_board._set_piece_at(
                    chess.G1 if self.turn == WHITE else chess.G8, chess.KING, self.turn
                )
                self.chess_board._set_piece_at(
                    chess.F1 if self.turn == WHITE else chess.F8, chess.ROOK, self.turn
                )

            # start added
            king_in_square: PieceInSquare = PieceInSquare(
                square=king_square, piece=chess.KING, color=self.chess_board.turn
            )
            board_modifications.add_appearance(appearance=king_in_square)
            rook_in_square: PieceInSquare = PieceInSquare(
                square=rook_square, piece=chess.ROOK, color=self.chess_board.turn
            )
            board_modifications.add_appearance(appearance=rook_in_square)
            # end added

        # Put the piece on the target square.
        if not castling:
            was_promoted = bool(self.chess_board.promoted & to_bb)
            self.chess_board._set_piece_at(
                move.to_square, piece_type, self.chess_board.turn, promoted
            )

            if captured_piece_type:
                self.chess_board._push_capture(
                    move, capture_square, captured_piece_type, was_promoted
                )

            promote_piece_in_square: PieceInSquare = PieceInSquare(
                square=move.to_square, piece=piece_type, color=self.chess_board.turn
            )
            board_modifications.add_appearance(appearance=promote_piece_in_square)

        # Swap turn.
        self.chess_board.turn = not self.chess_board.turn
        return board_modifications

    def compute_key_old(self) -> str:
        """
        Computes and returns a unique key representing the current state of the chess board.

        The key is computed by concatenating various attributes of the board, including the positions of pawns, knights,
        bishops, rooks, queens, and kings, as well as the current turn, castling rights, en passant square, halfmove clock,
        occupied squares for each color, promoted pieces, and the fullmove number.

        Returns:
            str: A unique key representing the current state of the chess board.
        """
        string = (
            str(self.chess_board.pawns)
            + str(self.chess_board.knights)
            + str(self.chess_board.bishops)
            + str(self.chess_board.rooks)
            + str(self.chess_board.queens)
            + str(self.chess_board.kings)
            + str(self.chess_board.turn)
            + str(self.chess_board.castling_rights)
            + str(self.chess_board.ep_square)
            + str(self.chess_board.halfmove_clock)
            + str(self.chess_board.occupied_co[WHITE])
            + str(self.chess_board.occupied_co[BLACK])
            + str(self.chess_board.promoted)
            + str(self.chess_board.fullmove_number)
        )
        return string

    def print_chess_board(self) -> str:
        """
        Prints the current state of the chess board.

        This method prints the current state of the chess board, including the position of all the pieces.
        It also prints the FEN (Forsyth–Edwards Notation) representation of the board.

        Returns:
            None
        """
        return str(self.chess_board.fen)

    def number_of_pieces_on_the_board(self) -> int:
        """
        Returns the number of pieces currently on the board.

        Returns:
            int: The number of pieces on the board.
        """
        return self.chess_board.occupied.bit_count()

    def is_attacked(self, a_color: chess.Color) -> bool:
        """Check if any piece of the color `a_color` is attacked.

        Args:
            a_color (chess.Color): The color of the pieces to check.

        Returns:
            bool: True if any piece of the specified color is attacked, False otherwise.
        """
        all_squares_of_color = chess.SquareSet()
        for piece_type in [1, 2, 3, 4, 5, 6]:
            new_squares = self.chess_board.pieces(piece_type=piece_type, color=a_color)
            all_squares_of_color = all_squares_of_color.union(new_squares)
        all_attackers = chess.SquareSet()
        for square in all_squares_of_color:
            new_attackers = self.chess_board.attackers(not a_color, square)
            all_attackers = all_attackers.union(new_attackers)
        return bool(all_attackers)

    def is_game_over(self) -> bool:
        """
        Check if the game is over.

        Returns:
            bool: True if the game is over, False otherwise.
        """
        # assume that player claim draw otherwise the opponent might be overoptimistic
        # in winning position where draw by repetition occur
        claim_draw: bool = True if len(self.chess_board.move_stack) >= 4 else False

        is_game_over: bool = self.chess_board.is_game_over(claim_draw=claim_draw)
        return is_game_over

    def ply(self) -> int:
        """
        Returns the number of half-moves (plies) that have been played on the board.

        :return: The number of half-moves played on the board.
        :rtype: int
        """
        return self.chess_board.ply()

    @property
    def turn(self) -> chess.Color:
        """
        Get the current turn color.

        Returns:
            chess.Color: The color of the current turn.
        """
        return self.chess_board.turn

    @property
    def fen(self) -> fen:
        """
        Returns the Forsyth-Edwards Notation (FEN) representation of the chess board.

        :return: The FEN string representing the current state of the board.
        """
        return self.chess_board.fen()

    @property
    def legal_moves(self) -> LegalMoveKeyGenerator:
        """
        Returns a generator that yields all the legal moves for the current board state.

        Returns:
            chess.LegalMoveGenerator: A generator that yields legal moves.
        """
        # return self.chess_board.legal_moves
        return self.legal_moves_

    def piece_at(self, square: chess.Square) -> chess.Piece | None:
        """
        Returns the piece at the specified square on the chess board.

        Args:
            square (chess.Square): The square on the chess board.

        Returns:
            chess.Piece | None: The piece at the specified square, or None if there is no piece.

        """
        return self.chess_board.piece_at(square)

    def piece_map(
        self, mask: chess.Bitboard = chess.BB_ALL
    ) -> dict[chess.Square, tuple[int, bool]]:
        result = {}
        for square in chess.scan_reversed(self.chess_board.occupied & mask):
            piece_type: int | None = self.chess_board.piece_type_at(square)
            assert piece_type is not None
            mask = chess.BB_SQUARES[square]
            color = bool(self.chess_board.occupied_co[WHITE] & mask)
            result[square] = (piece_type, color)
        return result

    def has_castling_rights(self, color: chess.Color) -> bool:
        """
        Check if the specified color has castling rights.

        Args:
            color (chess.Color): The color to check for castling rights.

        Returns:
            bool: True if the color has castling rights, False otherwise.
        """
        return self.chess_board.has_castling_rights(color)

    def has_queenside_castling_rights(self, color: chess.Color) -> bool:
        """
        Check if the specified color has queenside castling rights.

        Args:
            color (chess.Color): The color to check for queenside castling rights.

        Returns:
            bool: True if the specified color has queenside castling rights, False otherwise.
        """
        return self.chess_board.has_queenside_castling_rights(color)

    def has_kingside_castling_rights(self, color: chess.Color) -> bool:
        """
        Check if the specified color has kingside castling rights.

        Args:
            color (chess.Color): The color to check for kingside castling rights.

        Returns:
            bool: True if the specified color has kingside castling rights, False otherwise.
        """
        return self.chess_board.has_kingside_castling_rights(color)

    def copy(self, stack: bool, deep_copy_legal_moves: bool = True) -> "BoardChi":
        """
        Create a copy of the current board.

        Args:
            stack (bool): Whether to copy the move stack as well.

        Returns:
            BoardChi: A new instance of the BoardChi class with the copied board.
        """
        chess_board_copy: chess.Board = self.chess_board.copy(stack=stack)

        legal_moves_copy: LegalMoveKeyGenerator
        if deep_copy_legal_moves:
            # deep_copy
            legal_moves_copy = self.legal_moves_.copy(
                copied_chess_board=chess_board_copy
            )
        else:
            # faster as move generated are not deep copied but tricky (should not be modified later!)
            legal_moves_copy = self.legal_moves_
            legal_moves_copy.chess_board = chess_board_copy

        return BoardChi(
            chess_board=chess_board_copy,
            compute_board_modification=self.compute_board_modification,
            fast_representation_=self.fast_representation_,
            legal_moves_=legal_moves_copy,
        )

    def __str__(self) -> str:
        """
        Returns a string representation of the board.

        Returns:
            str: A string representation of the board.
        """
        return self.chess_board.__str__()

    def tell_result(self) -> None:
        if self.chess_board.is_fivefold_repetition():
            ("is_fivefold_repetition")
        if self.chess_board.is_seventyfive_moves():
            chipiron_logger.info("is seventy five  moves")
        if self.chess_board.is_insufficient_material():
            chipiron_logger.info("is_insufficient_material")
        if self.chess_board.is_stalemate():
            chipiron_logger.info("is_stalemate")
        if self.chess_board.is_checkmate():
            chipiron_logger.info("is_checkmate")
        chipiron_logger.info(self.chess_board.result())

    def result(self, claim_draw: bool = False) -> str:
        return self.chess_board.result(claim_draw=claim_draw)

    @property
    def move_history_stack(self) -> list[moveUci]:
        return [move.uci() for move in self.chess_board.move_stack]

    @property
    def pawns(self) -> chess.Bitboard:
        return self.chess_board.pawns

    @property
    def knights(self) -> chess.Bitboard:
        return self.chess_board.knights

    @property
    def bishops(self) -> chess.Bitboard:
        return self.chess_board.bishops

    @property
    def rooks(self) -> chess.Bitboard:
        return self.chess_board.rooks

    @property
    def queens(self) -> chess.Bitboard:
        return self.chess_board.queens

    @property
    def kings(self) -> chess.Bitboard:
        return self.chess_board.kings

    @property
    def white(self) -> chess.Bitboard:
        return self.chess_board.occupied_co[chess.WHITE]

    @property
    def black(self) -> chess.Bitboard:
        return self.chess_board.occupied_co[chess.BLACK]

    @property
    def castling_rights(self) -> chess.Bitboard:
        return self.chess_board.castling_rights

    @property
    def occupied(self) -> chess.Bitboard:
        return self.chess_board.occupied

    def occupied_color(self, color: chess.Color) -> chess.Bitboard:
        return self.chess_board.occupied_co[color]

    def termination(self) -> chess.Termination:
        outcome: Outcome | None = self.chess_board.outcome(claim_draw=True)
        assert outcome is not None
        return outcome.termination

    @property
    def promoted(self) -> chess.Bitboard:
        return self.chess_board.promoted

    @property
    def fullmove_number(self) -> int:
        return self.chess_board.fullmove_number

    @property
    def halfmove_clock(self) -> int:
        return self.chess_board.halfmove_clock

    @property
    def ep_square(self) -> int | None:
        return self.chess_board.ep_square

    def is_zeroing(self, move: moveKey) -> bool:
        chess_move: chess.Move = self.legal_moves_.generated_moves[move]
        return self.chess_board.is_zeroing(chess_move)

    def into_fen_plus_history(self) -> FenPlusHistory:
        return FenPlusHistory(
            current_fen=self.fen,
            historical_moves=self.move_history_stack,
            historical_boards=self.chess_board._stack,
        )
