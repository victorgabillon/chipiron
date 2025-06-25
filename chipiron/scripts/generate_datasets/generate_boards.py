import chess.pgn
import pandas as pd
from pandas import DataFrame

from chipiron.environments.chess.board.utils import fen

# atm this file is not shared on GitHub so download one like this. This one was downloaded from the free online database of lichess
pgn = open(
    "/home/victor_old/pycharm/chipiron/data/lichess_db_standard_rated_2015-03.pgn"
)
data_frame_file_name = "/home/victor/good_games2025"
the_dic: list[dict[str, fen]] = []

count_game: int = 0
total_count_move: int = 0
recorded_board = 0
while recorded_board < 10000000:

    count_game += 1
    if count_game % 1000 == 0:
        print("count", count_game)
    game: chess.pgn.GameNode | None = chess.pgn.read_game(pgn)

    if count_game % 10000 == 0:  # save
        # weird that an array is given where a dictionary is expected
        new_data_frame_states: DataFrame = pd.DataFrame.from_dict(the_dic)
        print("%%", len(new_data_frame_states.index))
        recorded_board = len(new_data_frame_states.index)
        new_data_frame_states.to_pickle(data_frame_file_name)
    if game is None:
        print("GAME NONE")
        break
    else:
        # Iterate through all moves and play them on a board.
        chess_board = game.board()

        round_: int = 0
        for move in game.mainline_moves():
            total_count_move += 1
            round_ += 1
            #     try:
            chess_board.push(move)
            assert game is not None
            game = game.next()

            assert game is not None
            if game.eval() is not None and total_count_move % 50 == 0:
                the_dic.append({"fen": chess_board.fen()})
