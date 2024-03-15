import chess
from chipiron.environments.chess.board.board_tools import convert_to_fen
import chess.polyglot
from chipiron.environments.chess.board.board_modification import BoardModification
from .starting_position import AllStartingPositionArgs, FenStartingPositionArgs, \
    FileStartingPositionArgs
import typing

COLORS = [WHITE, BLACK] = [True, False]


class BoardChi:
    """
    Board Chipiron
    object that describes the current board. it wraps the chess Board from the chess package so it can have more in it
    but im not sure its really necessary.i keep it for potential usefulness
    """
    board: chess.Board

    def __init__(self, board: chess.Board) -> None:
        self.board = board

    def set_starting_position(
            self,
            starting_position_arg: AllStartingPositionArgs | None = None,
            fen=None
    ) -> None:

        self.board.reset()
        if starting_position_arg is not None:
            match starting_position_arg:
                case FileStartingPositionArgs():
                    starting_position_arg_file: FileStartingPositionArgs = starting_position_arg
                    file_name = starting_position_arg_file.file_name
                    fen = self.load_from_file(file_name)
                    self.board.set_fen(fen)
                case FenStartingPositionArgs():
                    starting_position_arg_fen: FenStartingPositionArgs = starting_position_arg
                    fen = starting_position_arg_fen.fen
                    self.board.set_fen(fen)

        elif fen is not None:
            self.board.set_fen(fen)

    def play_move(
            self,
            move: chess.Move
    ) -> BoardModification:
        assert (move in self.legal_moves)
        board_modif: BoardModification | None = self.push_and_return_modification(move)
        if board_modif is None:
            raise Exception('None Modif looks not good in board.py')
        else:
            return board_modif

    def rewind_one_move(self) -> None:
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
        board_modifications.add_removal((move.from_square, piece_type, self.board.turn))
        # print('^&',(move.from_square, piece_type, self.turn),move)
        assert piece_type is not None, f"push() expects move to be pseudo-legal, but got {move} in {self.board.board_fen()}"
        capture_square = move.to_square
        captured_piece_type = self.board.piece_type_at(capture_square)
        if captured_piece_type is not None:
            board_modifications.add_removal((capture_square, captured_piece_type, not self.board.turn))

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
                board_modifications.add_removal((capture_square, captured_piece_type, not self.board.turn))
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
            board_modifications.add_removal((move.from_square, chess.KING, self.board.turn))
            board_modifications.add_removal((move.to_square, chess.ROOK, self.board.turn))

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
            board_modifications.add_appearance((king_square, chess.KING, self.board.turn))
            board_modifications.add_appearance((rook_square, chess.ROOK, self.board.turn))

        # Put the piece on the target square.
        if not castling:
            was_promoted = bool(self.board.promoted & to_bb)
            self.board._set_piece_at(move.to_square, piece_type, self.board.turn, promoted)
            board_modifications.add_appearance((move.to_square, piece_type, self.board.turn))

            if captured_piece_type:
                self.board._push_capture(move, capture_square, captured_piece_type, was_promoted)

        # Swap turn.
        self.board.turn = not self.board.turn

        return board_modifications

    def load_from_file(self, file_name):
        with  open('data/starting_boards/' + file_name, "r") as f:
            asciiBoard = f.read()
            fen = convert_to_fen(asciiBoard)
        return fen

    def compute_key(self) -> str:
        string = str(self.board.pawns) + str(self.board.knights) \
                 + str(self.board.bishops) + str(self.board.rooks) \
                 + str(self.board.queens) + str(self.board.kings) \
                 + str(self.board.turn) + str(self.board.castling_rights) \
                 + str(self.board.ep_square) + str(self.board.halfmove_clock) \
                 + str(self.board.occupied_co[WHITE]) + str(self.board.occupied_co[BLACK]) \
                 + str(self.board.promoted) \
                 + str(self.board.fullmove_number)
        return string

    def fast_representation(self) -> str:
        return self.compute_key()

    def print_chess_board(self):
        print(self)
        print(self.board.fen())

    def number_of_pieces_on_the_board(self) -> int:
        return bin(self.board.occupied).count('1')

    def is_attacked(self, a_color):
        """ check if any piece of the color a_color is attacked"""
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
        return self.board.is_game_over()

    def ply(self):
        return self.board.ply()

    @property
    def turn(self):
        return self.board.turn

    def fen(self):
        return self.board.fen()

    @property
    def legal_moves(self):
        return self.board.legal_moves

    def piece_at(self, square):
        return self.board.piece_at(square)

    def has_castling_rights(self, color: chess.Color):
        return self.board.has_castling_rights(color)

    def has_queenside_castling_rights(self, color: chess.Color):
        return self.board.has_queenside_castling_rights(color)

    def has_kingside_castling_rights(self, color: chess.Color):
        return self.board.has_kingside_castling_rights(color)

    def copy(
            self,
            stack: bool
    ):
        board = self.board.copy(stack=stack)
        return BoardChi(board=board)

    def __str__(self) -> str:
        return self.board.__str__()
