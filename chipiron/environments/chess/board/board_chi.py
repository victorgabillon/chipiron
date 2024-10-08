"""
Module that contains the BoardChi class that wraps the chess.Board class from the chess package
"""
import pickle
import typing

import chess
import chess.polyglot

from chipiron.environments.chess.board.board_modification import BoardModification, PieceInSquare
from .utils import fen

COLORS = [WHITE, BLACK] = [True, False]


class BoardChi:
    """
    Board Chipiron
    object that describes the current board. it wraps the chess Board from the chess package so it can have more in it
    """

    board: chess.Board
    compute_board_modification: bool
    legal_moves_: list[chess.Move] | None = None
    fast_representation_: str | None = None

    def __init__(
            self,
            board: chess.Board,
            compute_board_modification: bool,
    ) -> None:
        """
        Initializes a new instance of the BoardChi class.

        Args:
            board: The chess.Board object to wrap.
        """
        self.board = board
        self.compute_board_modification = compute_board_modification

    def play_move(
            self,
            move: chess.Move
    ) -> BoardModification | None:
        """
        Plays a move on the board and returns the board modification.

        Args:
            move: The move to play.

        Returns:
            The board modification resulting from the move or None.
        """
        # assert self.board.is_legal(move)
        #
        board_modifications: BoardModification | None

        if self.compute_board_modification:
            board_modifications = self.push_and_return_modification(move)  # type: ignore
            # raise Exception('None Modif looks not good in board.py')
        else:
            board_modifications = self.board.push(move)

        self.legal_moves_ = None  # the legals moves needs to be recomputed as the board has changed
        return board_modifications

    def rewind_one_move(self) -> None:
        """
        Rewinds the board state to the previous move.
        """
        if self.ply() > 0:
            self.board.pop()
        else:
            print('Cannot rewind more as self.halfmove_clock equals {}'.format(self.ply()))

    @typing.no_type_check
    def push_and_return_modification(
            self,
            move: chess.Move
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

        # Push move and remember board state.
        move = self.board._to_chess960(move)
        board_state = self.board._board_state()
        self.board.castling_rights = self.board.clean_castling_rights()  # Before pushing stack
        self.board.move_stack.append(
            self.board._from_chess960(self.board.chess960, move.from_square, move.to_square, move.promotion,
                                      move.drop))
        # board_state_: chess._BoardState[Generic[chess.BoardT]] = board_state
        self.board._stack.append(board_state)

        # Reset en passant square.
        ep_square = self.board.ep_square
        self.board.ep_square = None

        # Increment move counters.
        self.board.halfmove_clock += 1
        if self.board.turn == BLACK:
            self.board.fullmove_number += 1

        # On a null move, simply swap turns and reset the en passant square.
        if not move:
            self.board.turn = not self.board.turn
            return None

        # Drops.
        if move.drop:
            self.board._set_piece_at(move.to_square, move.drop, self.board.turn)
            self.board.turn = not self.board.turn
            return None

        # Zero the half-move clock.
        if self.board.is_zeroing(move):
            self.board.halfmove_clock = 0

        from_bb = chess.BB_SQUARES[move.from_square]
        to_bb = chess.BB_SQUARES[move.to_square]
        promoted = bool(self.board.promoted & from_bb)
        piece_type = self.board._remove_piece_at(move.from_square)
        piece_in_square: PieceInSquare = PieceInSquare(
            square=move.from_square,
            piece=piece_type,
            color=self.board.turn
        )
        board_modifications.add_removal(removal=piece_in_square)
        # print('^&',(move.from_square, piece_type, self.turn),move)
        assert piece_type is not None, f"push() expects move to be pseudo-legal, but got {move} in {self.board.board_fen()}"
        capture_square = move.to_square
        captured_piece_type = self.board.piece_type_at(capture_square)
        if captured_piece_type is not None:
            captured_piece_in_square: PieceInSquare = PieceInSquare(
                square=capture_square,
                piece=captured_piece_type,
                color=not self.board.turn
            )
            board_modifications.add_removal(removal=captured_piece_in_square)

        # Update castling rights.
        self.board.castling_rights &= ~to_bb & ~from_bb
        if piece_type == chess.KING and not promoted:
            if self.board.turn == WHITE:
                self.board.castling_rights &= ~chess.BB_RANK_1
            else:
                self.board.castling_rights &= ~chess.BB_RANK_8
        elif captured_piece_type == chess.KING and not self.board.promoted & to_bb:
            if self.board.turn == WHITE and chess.square_rank(move.to_square) == 7:
                self.board.castling_rights &= ~chess.BB_RANK_8
            elif self.board.turn == BLACK and chess.square_rank(move.to_square) == 0:
                self.board.castling_rights &= ~chess.BB_RANK_1

        # Handle special pawn moves.
        if piece_type == chess.PAWN:
            diff = move.to_square - move.from_square

            if diff == 16 and chess.square_rank(move.from_square) == 1:
                self.board.ep_square = move.from_square + 8
            elif diff == -16 and chess.square_rank(move.from_square) == 6:
                self.board.ep_square = move.from_square - 8
            elif move.to_square == ep_square and abs(diff) in [7, 9] and not captured_piece_type:
                # Remove pawns captured en passant.
                down = -8 if self.board.turn == WHITE else 8
                capture_square = ep_square + down
                captured_color = self.board.color_at(capture_square)
                captured_piece_type = self.board._remove_piece_at(capture_square)
                pawn_captured_piece_in_square: PieceInSquare = PieceInSquare(
                    square=capture_square,
                    piece=captured_piece_type,
                    color=not self.board.turn
                )
                board_modifications.add_removal(removal=pawn_captured_piece_in_square)
                assert (not self.board.turn == captured_color)

        # Promotion.
        if move.promotion:
            promoted = True
            piece_type = move.promotion

        # Castling.
        castling = piece_type == chess.KING and self.board.occupied_co[self.board.turn] & to_bb
        if castling:
            a_side = chess.square_file(move.to_square) < chess.square_file(move.from_square)

            self.board._remove_piece_at(move.from_square)
            self.board._remove_piece_at(move.to_square)
            remove_king_in_square: PieceInSquare = PieceInSquare(
                square=move.from_square,
                piece=chess.KING,
                color=self.board.turn
            )
            board_modifications.add_removal(removal=remove_king_in_square)
            remove_rook_in_square: PieceInSquare = PieceInSquare(
                square=move.to_square,
                piece=chess.ROOK,
                color=self.board.turn
            )
            board_modifications.add_removal(removal=remove_rook_in_square)

            if a_side:
                king_square = chess.C1 if self.board.turn == chess.WHITE else chess.C8
                rook_square = chess.D1 if self.board.turn == WHITE else chess.D8
                self.board._set_piece_at(king_square, chess.KING, self.board.turn)
                self.board._set_piece_at(rook_square, chess.ROOK, self.board.turn)
            else:
                king_square = chess.G1 if self.board.turn == chess.WHITE else chess.G8
                rook_square = chess.F1 if self.board.turn == WHITE else chess.F8
                self.board._set_piece_at(king_square, chess.KING, self.board.turn)
                self.board._set_piece_at(rook_square, chess.ROOK, self.board.turn)
            king_in_square: PieceInSquare = PieceInSquare(
                square=king_square,
                piece=chess.KING,
                color=self.board.turn
            )
            board_modifications.add_appearance(appearance=king_in_square)
            rook_in_square: PieceInSquare = PieceInSquare(
                square=rook_square,
                piece=chess.ROOK,
                color=self.board.turn
            )
            board_modifications.add_appearance(appearance=rook_in_square)

        # Put the piece on the target square.
        if not castling:
            was_promoted = bool(self.board.promoted & to_bb)
            self.board._set_piece_at(move.to_square, piece_type, self.board.turn, promoted)
            promote_piece_in_square: PieceInSquare = PieceInSquare(
                square=move.to_square,
                piece=piece_type,
                color=self.board.turn
            )
            board_modifications.add_appearance(appearance=promote_piece_in_square)

            if captured_piece_type:
                self.board._push_capture(move, capture_square, captured_piece_type, was_promoted)

        # Swap turn.
        self.board.turn = not self.board.turn

        return board_modifications

    def compute_key(self) -> str:
        """
        Computes and returns a unique key representing the current state of the chess board.

        The key is computed by concatenating various attributes of the board, including the positions of pawns, knights,
        bishops, rooks, queens, and kings, as well as the current turn, castling rights, en passant square, halfmove clock,
        occupied squares for each color, promoted pieces, and the fullmove number.

        Returns:
            str: A unique key representing the current state of the chess board.
        """
        string = str(self.board.pawns) + str(self.board.knights) + str(self.board.bishops) + str(
            self.board.rooks) + str(self.board.queens) + str(self.board.kings) + str(self.board.turn) + str(
            self.board.castling_rights) + str(self.board.ep_square) + str(self.board.halfmove_clock) + str(
            self.board.occupied_co[WHITE]) + str(self.board.occupied_co[BLACK]) + str(self.board.promoted) + str(
            self.board.fullmove_number)
        return string

    def fast_representation(self) -> str:
        """
        Returns a fast representation of the board.

        This method computes and returns a string representation of the board
        that can be quickly generated and used for various purposes.

        :return: A string representation of the board.
        :rtype: str
        """

        if self.fast_representation_ is None:
            self.fast_representation_ = self.compute_key()
        return self.fast_representation_

    def print_chess_board(self) -> None:
        """
        Prints the current state of the chess board.

        This method prints the current state of the chess board, including the position of all the pieces.
        It also prints the FEN (Forsyth–Edwards Notation) representation of the board.

        Returns:
            None
        """
        print(self)
        print(self.board.fen)

    def number_of_pieces_on_the_board(self) -> int:
        """
        Returns the number of pieces currently on the board.

        Returns:
            int: The number of pieces on the board.
        """
        return self.board.occupied.bit_count()

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
        all_squares_of_color = chess.SquareSet()
        for piece_type in [1, 2, 3, 4, 5, 6]:
            new_squares = self.board.pieces(piece_type=piece_type, color=a_color)
            all_squares_of_color = all_squares_of_color.union(new_squares)
        all_attackers = chess.SquareSet()
        for square in all_squares_of_color:
            new_attackers = self.board.attackers(not a_color, square)
            all_attackers = all_attackers.union(new_attackers)
        return bool(all_attackers)

    def is_game_over(self) -> bool:
        """
        Check if the game is over.

        Returns:
            bool: True if the game is over, False otherwise.
        """
        return self.board.is_game_over()

    def ply(self) -> int:
        """
        Returns the number of half-moves (plies) that have been played on the board.

        :return: The number of half-moves played on the board.
        :rtype: int
        """
        return self.board.ply()

    @property
    def turn(self) -> chess.Color:
        """
        Get the current turn color.

        Returns:
            chess.Color: The color of the current turn.
        """
        return self.board.turn


    @property
    def fen(self) -> str:
        """
        Returns the Forsyth-Edwards Notation (FEN) representation of the chess board.

        :return: The FEN string representing the current state of the board.
        """
        return self.board.fen()

    @property
    def legal_moves(self) -> set[chess.Move]:
        """
        Returns a generator that yields all the legal moves for the current board state.

        Returns:
            chess.LegalMoveGenerator: A generator that yields legal moves.
        """
        # return self.board.legal_moves
        if self.legal_moves_ is not None:
            print('rrrooooo')
            return self.legal_moves_
        else:
            self.legal_moves_ = set(self.board.legal_moves)
            return self.legal_moves_

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
        return self.board.piece_at(square)

    def piece_map(
            self,
            mask: chess.Bitboard = chess.BB_ALL
    ) -> dict[chess.Square, (int, bool)]:
        result = {}
        for square in chess.scan_reversed(self.board.occupied & mask):
            piece_type: int = self.board.piece_type_at(square)
            mask = chess.BB_SQUARES[square]
            color = bool(self.board.occupied_co[WHITE] & mask)
            result[square] = (piece_type, color)
        return result

    def has_castling_rights(
            self,
            color: chess.Color
    ) -> bool:
        """
        Check if the specified color has castling rights.

        Args:
            color (chess.Color): The color to check for castling rights.

        Returns:
            bool: True if the color has castling rights, False otherwise.
        """
        return self.board.has_castling_rights(color)

    def has_queenside_castling_rights(
            self,
            color: chess.Color
    ) -> bool:
        """
        Check if the specified color has queenside castling rights.

        Args:
            color (chess.Color): The color to check for queenside castling rights.

        Returns:
            bool: True if the specified color has queenside castling rights, False otherwise.
        """
        return self.board.has_queenside_castling_rights(color)

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
        return self.board.has_kingside_castling_rights(color)

    def copy(
            self,
            stack: bool
    ) -> 'BoardChi':
        """
        Create a copy of the current board.

        Args:
            stack (bool): Whether to copy the move stack as well.

        Returns:
            BoardChi: A new instance of the BoardChi class with the copied board.
        """
        board: chess.Board = self.board.copy(stack=stack)
        return BoardChi(
            board=board,
            compute_board_modification=self.compute_board_modification
        )

    def __str__(self) -> str:
        """
        Returns a string representation of the board.

        Returns:
            str: A string representation of the board.
        """
        return self.board.__str__()

    def tell_result(self) -> None:
        if self.board.is_fivefold_repetition():
            print('is_fivefold_repetition')
        if self.board.is_seventyfive_moves():
            print('is seventy five  moves')
        if self.board.is_insufficient_material():
            print('is_insufficient_material')
        if self.board.is_stalemate():
            print('is_stalemate')
        if self.board.is_checkmate():
            print('is_checkmate')
        print(self.board.result())

    def result(self) -> str:
        return self.board.result()

    @property
    def move_history_stack(self) -> list[chess.Move]:
        return self.board.move_stack

    def dump(self, f) -> None:
        pickle.dump(self, f)