import chess
import pickle

first_line: int = 8

dict_fen_move: dict[str, list[chess.Move]] = {}

with open(file='matein2.txt', mode='r') as file:
    content = file.readlines()
    line: int = first_line
    index: int = 0
    while first_line + index * 5 < len(content):
        fen_line: int = first_line + index * 5
        fen = content[fen_line]
        board = chess.Board(fen=fen)
        moves = content[fen_line + 1]
        moves = moves.split()
        if board.turn:
            moves_san = moves[1:3] + moves[4:]
        else:
            moves_san = moves[1:2] + moves[3:]
        chess_moves = []
        for move_san in moves_san:
            mmo = board.parse_san(move_san)
            chess_moves.append(mmo)
            board.push_san(move_san)

        dict_fen_move[fen] = chess_moves
        index += 1

with open('mate_in_2_db.pickle', 'wb') as file:
    pickle.dump(obj=dict_fen_move, file=file)
