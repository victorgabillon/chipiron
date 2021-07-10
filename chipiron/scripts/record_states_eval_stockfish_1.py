import pandas as pd
import chess.pgn
import chess
import random
from scripts.script import Script


class RecordStateEvalStockfish1(Script):

    def __init__(self):
        super().__init__()

        self.pgn = open('/home/victor/Downloads/lichess_db_standard_rated_2021-02.pgn')
        self.data_frame_file_name = '/home/victor/goodgames_plusvariation_stockfish_eval_train_t.1#4'
        self.engine = chess.engine.SimpleEngine.popen_uci(
            "/home/victor/stockfish_13_linux_x64/stockfish_13_linux_x64")
        self.the_dic = []

    def run(self):

        count = 0

        while True:
            count += 1
            if count % 100 == 0:
                print('count', count)

            if count % 100 == 0:  # save
                new_data_frame_states_eval = pd.DataFrame.from_dict(self.the_dic)
                print('adding ', len(new_data_frame_states_eval.index))
                try:
                    previous_data = pd.read_pickle(self.data_frame_file_name)
                except:
                    previous_data = pd.DataFrame()
                print('before ', len(previous_data.index))
                merged_data = pd.concat([previous_data ,new_data_frame_states_eval ], ignore_index=True)
                print('now ', len(merged_data.index))
                merged_data.to_pickle(self.data_frame_file_name)
                self.the_dic = []

            game = chess.pgn.read_game(self.pgn)

            if count < 0000:
                continue

            if game == None:
                print('ko')
                break

            # Iterate through all moves and play them on a board.
            chess_board = game.board()

            round_ = 0
            game_length = 0
            for move in game.mainline_moves():
                game_length += 1
            # print('game_length',game_length)

            if game_length < 5:
                continue
            random_length = random.randint(0, game_length - 2)

            for move in game.mainline_moves():
                round_ += 1
                chess_board.push(move)
                game = game.next()
                if round_ == random_length:
                    break
            # print('dddgame_length',game_length,list(chess_board.legal_moves))

            if list(chess_board.legal_moves) == []:
                continue

            extra_move = random.choice(list(chess_board.legal_moves))
            chess_board_copy = chess_board.copy()
            chess_board_copy.push(extra_move)
            my_board = chess_board_copy

            if my_board.is_valid():

                info = self.engine.analyse(my_board, chess.engine.Limit(time=.1))

                # try:
                #
                #
                #     info = egine.analyse(my_board, chess.engine.Limit(time=.01))
                #     problemos = False
                # except:
                #     problemos = True
                #     pass
                #
                # if problemos:
                #     continue

                    # if info["score"].is_mate()

                # print('^^^',info["score"].pov(chess.WHITE),info["score"].pov(chess.WHITE).score(mate_score=100000))

                self.the_dic.append({'fen': my_board.fen(), 'explored': 'engine',
                                     'stockfish_value': info["score"].pov(chess.WHITE).score(mate_score=100000)})
