import time
import chess
import pickle
import global_variables
from src.players.boardevaluators.stockfish_evaluation import Stockfish


# todo a wrapper for chess.white chess.black


class GameManager:
    """
    Objet in charge of playing one game
    """

    GAME_RESULTS = [WIN_FOR_WHITE, WIN_FOR_BLACK, DRAW] = range(3)

    def __init__(self, board, syzygy, output_folder_path=None, args=None, player_white=None,
                 player_black=None):
        self.board = board
        self.syzygy = syzygy

        self.path_to_store_result = output_folder_path + '/games/' if output_folder_path is not None else None

        self.board_evaluator = Stockfish()

        self.args = args
        self.player_white = player_white
        self.player_black = player_black

        self.board_history = []

    def set(self, args, player_white, player_black):
        self.args = args
        self.player_white = player_white
        self.player_black = player_black

    def swap_players(self):
        player_temp = self.player_black
        self.player_black = self.player_white
        self.player_white = player_temp

    def stockfish_eval(self):

        return self.board_evaluator.score(self.board)

    def play_one_move(self, move):
        self.board.play_move(move)
        if self.syzygy.fast_in_table(self.board):
            print('Theoretically finished with value for white: ', self.syzygy.sting_result(self.board))

    def set_new_game(self, starting_position_arg):
        self.board.set_starting_position(starting_position_arg)
        self.board_history = [self.board.copy()]

    def play_one_game(self):

        self.set_new_game(self.args['starting_position'])
        time.sleep(.1)
        global_variables.global_lock.acquire()

        self.allow_display()
        color_names = ['Black', 'White']
        players = [self.player_black, self.player_white]

        half_move = self.board.ply()
        while not self.board.is_game_over() and self.game_continue_conditions():
            print('Half Move: ', half_move)
            self.player_selects_move(player=players[self.board.turn], color_of_player_str=color_names[self.board.turn])
            self.allow_display()
            half_move += + 1

        self.tell_results()
        global_variables.global_lock.release()
        return self.simple_results()

    def game_continue_conditions(self):
        half_move = self.board.ply()
        continue_bool = True
        if 'max_half_move' in self.args and half_move >= self.args['max_half_move']:
            continue_bool = False
        return continue_bool




    def allow_display(self):
        # TODO mqke it nicer thqn thqt
        global_variables.global_lock.release()
        time.sleep(.101)
        global_variables.global_lock.acquire()

    def player_selects_move(self, player, color_of_player_str):
        assert(~((player == self.player_white) ^ (color_of_player_str == 'White'))+2) # xnor testting the colors
        print('{} ({}) to play now...'.format(color_of_player_str, player.player_name_id))
        move = player.get_move(self.board, float(self.args['secondsPerMoveWhite']))
        print('{}: {} plays {}'.format(color_of_player_str, player.player_name_id, move))
        if player.player_name_id != 'Human':
            if not self.board.is_legal(move):  # check if its a legal moves!!!!!!
                raise Exception('illegal move from player white: ' + str(move))
            self.play_one_move(move)
        self.board.print_chess_board()

    def print_to_file(self, idx=0):
        if self.path_to_store_result is not None:
            path_file = self.path_to_store_result + '_' + str(
                idx) + '_W:' + self.player_white.player_name_id + '-vs-B:' + self.player_black.player_name_id
            path_file_txt = path_file + '.txt'
            path_file_obj = path_file + '.obj'
            with open(path_file_txt, 'a') as the_fileText:
                for counter, move in enumerate(self.board.move_stack):
                    if counter % 2 == 0:
                        move_1 = move
                    else:
                        the_fileText.write(str(move_1) + ' ' + str(move) + '\n')
            with open(path_file_obj, "wb") as f:
                pickle.dump(self.board, f)

    def tell_results(self):
        if self.syzygy.fast_in_table(self.board):
            print('Syzygy: Theoretical value for white', self.syzygy.sting_result(self.board))
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

    def simple_results(self):
        if self.board.result() == '*':
            if self.syzygy is None or not self.syzygy.fast_in_table(self.board):  # TODO when is this case useful?
                return (-10000, -10000, -10000)
            else:
                val = self.syzygy.value_white(self.board, chess.WHITE)
                if val == 0:
                    return self.DRAW
                if val == -1000:
                    return self.WIN_FOR_BLACK
                if val == 1000:
                    return self.WIN_FOR_WHITE
        else:
            result = self.board.result()
            if result == '1/2-1/2':
                return self.DRAW
            if result == '0-1':
                return self.WIN_FOR_BLACK
            if result == '1-0':
                return self.WIN_FOR_WHITE
