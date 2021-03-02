import pandas as pd
import chess.pgn

pgn = open("chipiron/data/png_games/ficsgamesdb_2019_chess2000_nomovetimes_191952.pgn")
data_frame_file_name = 'chipiron/data/states_from_png.data'
the_dic = []

count = 0
while True:

    count += 1
    if count % 100 == 0:
        print('count', count)
    game = chess.pgn.read_game(pgn)

    if game == None:  # save
        print('%%',the_dic)
        new_data_frame_states = pd.DataFrame.from_dict(the_dic)
        new_data_frame_states.to_pickle(data_frame_file_name)
        break

    # Iterate through all moves and play them on a board.
    chess_board = game.board()
    for move in game.mainline_moves():
        try:
          chess_board.push(move)
          the_dic.append({'fen': chess_board.fen()})
        except:
            pass
        # print('@@',chess_board.fen())
        # print('@@',chess_board)

        # input("Press Enter to continue...")
