import chess
from chipiron.environments.chess.board.board_tools import convert_to_fen
import chess.polyglot
from chipiron.environments.chess.board.board_modification import BoardModification
from .starting_position import StaringPositionArgs, StaringPositionArgsType

COLORS = [WHITE, BLACK] = [True, False]


class BoardChi(chess.Board):
    """
    Board Chipiron
    object that describes the current board. it wraps the chess Board from the chess package so it can have more in it
    but im not sure its really necessary.i keep it for potential usefulness
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def set_starting_position(self,
                              starting_position_arg: StaringPositionArgs = None,
                              fen=None):
        self.reset()
        if starting_position_arg is not None:
            match starting_position_arg.type:
                case StaringPositionArgsType.fromFile:
                    file_name = starting_position_arg.file_name
                    fen = self.load_from_file(file_name)
                    self.set_fen(fen)
                case StaringPositionArgsType.fromFile:
                    fen = starting_position_arg.fen
                    self.set_fen(fen)

        elif fen is not None:
            self.set_fen(fen)

    def play_move(self, move: chess.Move) -> BoardModification:
        return self.push_and_return_modification(move)

    def rewind_one_move(self) -> None:
        if self.ply() > 0:
            self.pop()
        else:
            print('Cannot rewind more as self.halfmove_clock equals {}'.format(self.ply()))

    def push_and_return_modification(self, move: chess.Move) -> BoardModification:
        """
        Mostly reuse the push function of the chess library but records the modifications to the bitboard so that
        we can do the same with other parallel representations such as tensor in pytorch
        """
        board_modifications = BoardModification()

        # Push move and remember board state.
        move = self._to_chess960(move)
        board_state = self._board_state()
        self.castling_rights = self.clean_castling_rights()  # Before pushing stack
        self.move_stack.append(
            self._from_chess960(self.chess960, move.from_square, move.to_square, move.promotion,
                                move.drop))
        self._stack.append(board_state)

        # Reset en passant square.
        ep_square = self.ep_square
        self.ep_square = None

        # Increment move counters.
        self.halfmove_clock += 1
        if self.turn == BLACK:
            self.fullmove_number += 1

        # On a null move, simply swap turns and reset the en passant square.
        if not move:
            self.turn = not self.turn
            return

        # Drops.
        if move.drop:
            self._set_piece_at(move.to_square, move.drop, self.turn)
            self.turn = not self.turn
            return

        # Zero the half-move clock.
        if self.is_zeroing(move):
            self.halfmove_clock = 0

        from_bb = chess.BB_SQUARES[move.from_square]
        to_bb = chess.BB_SQUARES[move.to_square]
        promoted = bool(self.promoted & from_bb)
        piece_type = self._remove_piece_at(move.from_square)
        board_modifications.add_removal((move.from_square, piece_type, self.turn))
        # print('^&',(move.from_square, piece_type, self.turn),move)
        assert piece_type is not None, f"push() expects move to be pseudo-legal, but got {move} in {self.board_fen()}"
        capture_square = move.to_square
        captured_piece_type = self.piece_type_at(capture_square)
        if captured_piece_type is not None:
            board_modifications.add_removal((capture_square, captured_piece_type, not self.turn))

        # Update castling rights.
        self.castling_rights &= ~to_bb & ~from_bb
        if piece_type == chess.KING and not promoted:
            if self.turn == WHITE:
                self.castling_rights &= ~chess.BB_RANK_1
            else:
                self.castling_rights &= ~chess.BB_RANK_8
        elif captured_piece_type == chess.KING and not self.promoted & to_bb:
            if self.turn == WHITE and chess.square_rank(move.to_square) == 7:
                self.castling_rights &= ~chess.BB_RANK_8
            elif self.turn == BLACK and chess.square_rank(move.to_square) == 0:
                self.castling_rights &= ~chess.BB_RANK_1

        # Handle special pawn moves.
        if piece_type == chess.PAWN:
            diff = move.to_square - move.from_square

            if diff == 16 and chess.square_rank(move.from_square) == 1:
                self.ep_square = move.from_square + 8
            elif diff == -16 and chess.square_rank(move.from_square) == 6:
                self.ep_square = move.from_square - 8
            elif move.to_square == ep_square and abs(diff) in [7, 9] and not captured_piece_type:
                # Remove pawns captured en passant.
                down = -8 if self.turn == WHITE else 8
                capture_square = ep_square + down
                captured_color = self.color_at(capture_square)
                captured_piece_type = self._remove_piece_at(capture_square)
                board_modifications.add_removal((capture_square, captured_piece_type, not self.turn))
                assert (not self.turn == captured_color)

        # Promotion.
        if move.promotion:
            promoted = True
            piece_type = move.promotion

        # Castling.
        castling = piece_type == chess.KING and self.occupied_co[self.turn] & to_bb
        if castling:
            a_side = chess.square_file(move.to_square) < chess.square_file(move.from_square)

            self._remove_piece_at(move.from_square)
            self._remove_piece_at(move.to_square)
            board_modifications.add_removal((move.from_square, chess.KING, self.turn))
            board_modifications.add_removal((move.to_square, chess.ROOK, self.turn))

            if a_side:
                king_square = chess.C1 if self.turn == chess.WHITE else chess.C8
                rook_square = chess.D1 if self.turn == WHITE else chess.D8
                self._set_piece_at(king_square, chess.KING, self.turn)
                self._set_piece_at(rook_square, chess.ROOK, self.turn)
            else:
                king_square = chess.G1 if self.turn == chess.WHITE else chess.G8
                rook_square = chess.F1 if self.turn == WHITE else chess.F8
                self._set_piece_at(king_square, chess.KING, self.turn)
                self._set_piece_at(rook_square, chess.ROOK, self.turn)
            board_modifications.add_appearance((king_square, chess.KING, self.turn))
            board_modifications.add_appearance((rook_square, chess.ROOK, self.turn))

        # Put the piece on the target square.
        if not castling:
            was_promoted = bool(self.promoted & to_bb)
            self._set_piece_at(move.to_square, piece_type, self.turn, promoted)
            board_modifications.add_appearance((move.to_square, piece_type, self.turn))

            if captured_piece_type:
                self._push_capture(move, capture_square, captured_piece_type, was_promoted)

        # Swap turn.
        self.turn = not self.turn

        return board_modifications

    def load_from_file(self, file_name):
        with  open('data/starting_boards/' + file_name, "r") as f:
            asciiBoard = f.read()
            fen = convert_to_fen(asciiBoard)
        return fen

    def compute_key(self) -> str:
        string = str(self.pawns) + str(self.knights) \
                 + str(self.bishops) + str(self.rooks) \
                 + str(self.queens) + str(self.kings) \
                 + str(self.turn) + str(self.castling_rights) \
                 + str(self.ep_square) + str(self.halfmove_clock) \
                 + str(self.occupied_co[WHITE]) + str(self.occupied_co[BLACK]) \
                 + str(self.promoted) \
                 + str(self.fullmove_number)
        return string

    def fast_representation(self) -> str:
        return self.compute_key()

    def print_chess_board(self):
        print(self)
        print(self.fen())

    def number_of_pieces_on_the_board(self):
        return bin(self.occupied).count('1')

    def is_attacked(self, a_color):
        """ check if any piece of the color a_color is attacked"""
        all_squares_of_color = chess.SquareSet()
        for piece_type in [1, 2, 3, 4, 5, 6]:
            new_squares = self.pieces(piece_type=piece_type, color=a_color)
            all_squares_of_color = all_squares_of_color.union(new_squares)
        all_attackers = chess.SquareSet()
        for square in all_squares_of_color:
            new_attackers = self.attackers(not a_color, square)
            all_attackers = all_attackers.union(new_attackers)
        return bool(all_attackers)
