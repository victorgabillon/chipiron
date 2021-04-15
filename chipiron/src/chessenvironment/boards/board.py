import chess
from src.chessenvironment.boards.board_tools import convertToFen
import chess.polyglot
from src.chessenvironment.boards.board_modification import BoardModification

COLORS = [WHITE, BLACK] = [True, False]


class MyBoard:
    """
    object that describes the current board. it wraps the chess Board from the chess package so it can have more in it
    but im not sure its really necessary.i keep it for potential usefulness
    """

    def __init__(self, starting_position_arg=None, chess_board=None, fen=None):

        if starting_position_arg is not None:
            if starting_position_arg['type'] == 'classic':
                self.chess_board = chess.Board()
            elif starting_position_arg['type'] == 'fromFile':
                file_name = starting_position_arg['options']['file_name']
                fen = self.load_from_file(file_name)
                self.chess_board = chess.Board(fen)
            elif starting_position_arg['type'] == 'fen':
                fen = starting_position_arg['fen']
                print(';fen', fen)
                self.chess_board = chess.Board(fen)

        if fen is not None:
            self.chess_board = chess.Board(fen)

        if chess_board is not None:
            self.chess_board = chess_board

    def is_legal(self, move):
        return move in self.chess_board.legal_moves

    def get_legal_moves(self):
        return self.chess_board.legal_moves

    def play(self, move):
        print('~~@', move)
        self.chess_board._and_return_modification(move)

    def push_and_return_modification(self, move):
        """
        mostly reuse the push function of the chess library but recordes the modificaitons to the bitboard so that
        we can do the same with other paralelle represenation such as tensor in pytorch
        """
        board_modifications = BoardModification()

        # Push move and remember board state.
        move = self.chess_board._to_chess960(move)
        board_state = self.chess_board._board_state()
        self.chess_board.castling_rights = self.chess_board.clean_castling_rights()  # Before pushing stack
        self.chess_board.move_stack.append(
            self.chess_board._from_chess960(self.chess_board.chess960, move.from_square, move.to_square, move.promotion,
                                            move.drop))
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
            self.chess_board._set_piece_at(move.to_square, move.drop, self.chess_board.turn)
            self.chess_board.turn = not self.chess_board.turn
            return

        # Zero the half-move clock.
        if self.chess_board.is_zeroing(move):
            self.chess_board.halfmove_clock = 0

        from_bb = chess.BB_SQUARES[move.from_square]
        to_bb = chess.BB_SQUARES[move.to_square]
        promoted = bool(self.chess_board.promoted & from_bb)
        piece_type = self.chess_board._remove_piece_at(move.from_square)
        board_modifications.add_removal((move.from_square, piece_type, self.chess_board.turn))
       # print('^&',(move.from_square, piece_type, self.chess_board.turn),move)
        assert piece_type is not None, f"push() expects move to be pseudo-legal, but got {move} in {self.chess_board.board_fen()}"
        capture_square = move.to_square
        captured_piece_type = self.chess_board.piece_type_at(capture_square)
        if captured_piece_type is not None:
            board_modifications.add_removal((capture_square, captured_piece_type, not self.chess_board.turn))

        # Update castling rights.
        self.chess_board.castling_rights &= ~to_bb & ~from_bb
        if piece_type == chess.KING and not promoted:
            if self.chess_board.turn == WHITE:
                self.chess_board.castling_rights &= ~chess.BB_RANK_1
            else:
                self.chess_board.castling_rights &= ~chess.BB_RANK_8
        elif captured_piece_type == chess.KING and not self.chess_board.promoted & to_bb:
            if self.chess_board.turn == WHITE and chess.square_rank(move.to_square) == 7:
                self.chess_board.castling_rights &= ~chess.BB_RANK_8
            elif self.chess_board.turn == BLACK and chess.square_rank(move.to_square) == 0:
                self.chess_board.castling_rights &= ~chess.BB_RANK_1

        # Handle special pawn moves.
        if piece_type == chess.PAWN:
            diff = move.to_square - move.from_square

            if diff == 16 and chess.square_rank(move.from_square) == 1:
                self.chess_board.ep_square = move.from_square + 8
            elif diff == -16 and chess.square_rank(move.from_square) == 6:
                self.chess_board.ep_square = move.from_square - 8
            elif move.to_square == ep_square and abs(diff) in [7, 9] and not captured_piece_type:
                # Remove pawns captured en passant.
                down = -8 if self.chess_board.turn == WHITE else 8
                capture_square = ep_square + down
                captured_color = self.chess_board.color_at(capture_square)
                captured_piece_type = self.chess_board._remove_piece_at(capture_square)
                board_modifications.add_removal((capture_square, captured_piece_type, not self.chess_board.turn))
                assert (not self.chess_board.turn == captured_color)

        # Promotion.
        if move.promotion:
            promoted = True
            piece_type = move.promotion

        # Castling.
        castling = piece_type == chess.KING and self.chess_board.occupied_co[self.chess_board.turn] & to_bb
        if castling:
            a_side = chess.square_file(move.to_square) < chess.square_file(move.from_square)

            self.chess_board._remove_piece_at(move.from_square)
            self.chess_board._remove_piece_at(move.to_square)
            board_modifications.add_removal((move.from_square, chess.KING, self.chess_board.turn))
            board_modifications.add_removal((move.to_square, chess.ROOK, self.chess_board.turn))

            if a_side:
                king_square = chess.C1 if self.chess_board.turn == chess.WHITE else chess.C8
                rook_square = chess.D1 if self.chess_board.turn == WHITE else chess.D8
                self.chess_board._set_piece_at(king_square, chess.KING, self.chess_board.turn)
                self.chess_board._set_piece_at(rook_square, chess.ROOK, self.chess_board.turn)
            else:
                king_square = chess.G1 if self.chess_board.turn == chess.WHITE else chess.G8
                rook_square = chess.F1 if self.chess_board.turn == WHITE else chess.F8
                self.chess_board._set_piece_at(king_square, chess.KING, self.chess_board.turn)
                self.chess_board._set_piece_at(rook_square, chess.ROOK, self.chess_board.turn)
            board_modifications.add_appearance((king_square, chess.KING, self.chess_board.turn))
            board_modifications.add_appearance((rook_square, chess.ROOK, self.chess_board.turn))

        # Put the piece on the target square.
        if not castling:
            was_promoted = bool(self.chess_board.promoted & to_bb)
            self.chess_board._set_piece_at(move.to_square, piece_type, self.chess_board.turn, promoted)
            board_modifications.add_appearance((move.to_square, piece_type, self.chess_board.turn))


            if captured_piece_type:
                self.chess_board._push_capture(move, capture_square, captured_piece_type, was_promoted)

        # Swap turn.
        self.chess_board.turn = not self.chess_board.turn

        return board_modifications

    def load_from_file(self, file_name):
        with  open('chipiron/runs/StartingBoards/' + file_name, "r") as f:
            asciiBoard = f.read()
            fen = convertToFen(asciiBoard)
        return fen

    def compute_key(self):
        string = str(self.chess_board.pawns) + str(self.chess_board.knights) \
                 + str(self.chess_board.bishops) + str(self.chess_board.rooks) \
                 + str(self.chess_board.queens) + str(self.chess_board.kings) \
                 + str(self.chess_board.turn) + str(self.chess_board.castling_rights) \
                 + str(self.chess_board.ep_square) + str(self.chess_board.halfmove_clock) \
                 + str(self.chess_board.occupied_co[WHITE]) + str(self.chess_board.occupied_co[BLACK]) \
                 + str(self.chess_board.promoted) \
                 + str(self.chess_board.fullmove_number)
        return string

    def fast_representation(self):
        # fastRep = board.chessBoard.fen()
        # return chess.polyglot.zobrist_hash(self.chess_board)
        return self.compute_key()

    def print_chess_board(self):
        print(self.chess_board)
        print(self.chess_board.fen())

    def __str__(self):
        return self.chess_board.__str__()

    def number_of_pieces_on_the_board(self):
        return bin(self.chess_board.occupied).count('1')

    def copy(self):
        return type(self)(None, self.chess_board.copy())

    def is_attacked(self, a_color):
        """ check if any piece of the color a_color is attacked"""
        all_squares_of_color = chess.SquareSet()
        for piece_type in [1, 2, 3, 4, 5, 6]:
            new_squares = self.chess_board.pieces(piece_type=piece_type, color=a_color)
            all_squares_of_color = all_squares_of_color.union(new_squares)
        # print('%%', all_squares_of_color)
        all_attackers = chess.SquareSet()
        for square in all_squares_of_color:
            new_attackers = self.chess_board.attackers(not a_color, square)
            all_attackers = all_attackers.union(new_attackers)
        if bool(all_attackers):
            print(self)
            print('attackers')
            print(all_attackers)
        return bool(all_attackers)
