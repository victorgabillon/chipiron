import pandas as pd
import chess.pgn
import chess
import random
from scripts.script import Script
import tracemalloc
import os

class RecordStateEvalStockfish1(Script):

    def __init__(self):
        tracemalloc.start()

        super().__init__()

        self.pgn = open('/home/victor/Downloads/lichess_db_standard_rated_2021-02.pgn')
        self.data_frame_file_name = '/home/victor/goodgames_plusvariation_stockfish_eval_train_t.1#17.csv'
        self.engine = chess.engine.SimpleEngine.popen_uci(
            "/home/victor/stockfish_13_linux_x64/stockfish_13_linux_x64")
        self.the_dic = []
        self.random_generator = random.Random(seed=0)

    def run(self):

        count = 0

        while True:
            count += 1
            #print('count', count)

            if count % 1000 == 0:
                print('count', count)

            if count < 8000000:
                chess.pgn.skip_game(self.pgn)
                continue
            else:
                game = chess.pgn.read_game(self.pgn)

            if game == None:
                print('ko')
                break

            if count % 1000 == 0 and self.the_dic:  # save
               # print('the dic',self.the_dic)
                new_data_frame_states_eval = pd.DataFrame.from_dict(self.the_dic)
                print('adding ', len(new_data_frame_states_eval.index))
                hdr = False if os.path.isfile(
                    self.data_frame_file_name) else True  # options to not  add heqder if file already exists
                try:
                  new_data_frame_states_eval.to_csv(self.data_frame_file_name, mode='a', header = hdr,index=False)  #append
                except:
                   new_data_frame_states_eval.to_csv(self.data_frame_file_name, mode='a', header=hdr,
                                                     index=False)  # append

                self.the_dic = []
                current, peak = tracemalloc.get_traced_memory()
                print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")




            # Iterate through all moves and play them on a board.
            chess_board = game.board()

            round_ = 0
            game_length = 0
            for move in game.mainline_moves():
                game_length += 1

            if game_length < 5:
                continue
            random_length = self.random_generator.randint(0, game_length - 2)

            for move in game.mainline_moves():
                round_ += 1
                chess_board.push(move)
                game = game.next()
                if round_ == random_length:
                    break

            if list(chess_board.legal_moves) == []:
                continue

            extra_move = self.random_generator.choice(list(chess_board.legal_moves))
            chess_board_copy = chess_board.copy()
            chess_board_copy.push(extra_move)
            my_board = chess_board_copy

            if my_board.is_valid():
                info = self.engine.analyse(my_board, chess.engine.Limit(time=.1))
                self.the_dic.append({'fen': my_board.fen(), 'explored': 'engine',
                                     'stockfish_value': info["score"].pov(chess.WHITE).score(mate_score=100000)})
