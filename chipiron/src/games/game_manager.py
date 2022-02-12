import chess
import pickle


# todo a wrapper for chess.white chess.black


class GameManager:
    """
    Objet in charge of playing one game
    """

    GAME_RESULTS = [WIN_FOR_WHITE, WIN_FOR_BLACK, DRAW] = range(3)

    def __init__(self,
                 observable_board,
                 syzygy,
                 display_board_evaluator,
                 output_folder_path,
                 args,
                 player_color_to_id,
                 main_thread_mailbox,
                 player_threads):
        """
        Constructor for the GameManager Class. If the args, and players are not given a value it is set to None,
         waiting for the set methods to be called. This is done like this so that the players can be changed
          (often swapped) easily.

        Args:
            board:
            syzygy:
            display_board_evaluator:  the board evaluator to display an independent evaluation. (Stockfish like)
            output_folder_path:
            args:
            player_white:
            player_black:
        """
        self.observable_board = observable_board
        self.board = observable_board.board # is it ugly? like to wrapping?
        self.syzygy = syzygy
        self.path_to_store_result = output_folder_path + '/games/' if output_folder_path is not None else None
        self.display_board_evaluator = display_board_evaluator
        self.args = args
        self.player_color_to_id = player_color_to_id
        self.board_history = []
        self.main_thread_mailbox = main_thread_mailbox
        self.player_threads = player_threads

    def stockfish_eval(self):
        return self.display_board_evaluator.get_evaluation(self.observable_board.board)  # TODO DON'T LIKE THIS writing

    def play_one_move(self, move):
        self.observable_board.play_move(move)
        if self.syzygy.fast_in_table(self.observable_board.board):
            print('Theoretically finished with value for white: ', self.syzygy.sting_result(self.observable_board.board))

    def set_new_game(self, starting_position_arg):
        self.observable_board.set_starting_position(starting_position_arg)

        self.board_history = [self.observable_board.board.copy()]

    def play_one_game(self):

        self.set_new_game(self.args['starting_position'])

        color_names = ['Black', 'White']
        half_move = self.observable_board.ply()

        while True:
            print('Half Move: ', half_move)
            color_to_move = self.observable_board.board.turn
            color_of_player_to_move_str = color_names[color_to_move]
            print('{} ({}) to play now...'.format(color_of_player_to_move_str,
                                                  self.player_color_to_id[color_to_move]))

            # waiting for a message
            mail = self.main_thread_mailbox.get()

            self.processing_mail(mail)
            eval = self.stockfish_eval()
            print('Stockfish evaluation:', eval)

            if self.observable_board.board.is_game_over() or not self.game_continue_conditions():
                break

        self.tell_results()
        return self.simple_results()

    def processing_mail(self, message):
        if message['type'] == 'move':
            # play the move
            move = message['move']
            print('receiving the move', move, type(self))
            #TODO CHECK IF THE MOVE CORRESPONDS TO THE CURRENT BOARD OTHER WISE KEEP FOR LATER? THINK! HOW TO DEAL with premoves if we dont know the board in advance?
            self.play_one_move(move)
            # Print the board
            self.observable_board.print_chess_board()
        if message['type'] == 'move_syzygy': # TODO IS THIS CLEAN?
            self.syzygy_player.best_move(self.observable_board.board)
            self.play_one_move(move)

        if message['type'] == 'back':
            pass

    def game_continue_conditions(self):
        half_move = self.observable_board.ply()
        continue_bool = True
        if 'max_half_move' in self.args and half_move >= self.args['max_half_move']:
            continue_bool = False
        return continue_bool

    def print_to_file(self, idx=0):
        if self.path_to_store_result is not None:
            path_file = self.path_to_store_result + '_' + str(
                idx) + '_W:' + self.player_color_to_id[chess.WHITE] + '-vs-B:' + self.player_color_to_id[chess.BLACK]
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

    def terminate_threads(self):
        for player_thread in self.player_threads:
            player_thread.stop()
            print('stopping the thread')
            player_thread.join()
            print('thread stopped')


