import chess
import pickle
import logging
from chipiron.utils import path
from .game import ObservableGame
from .game_args import GameArgs

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
file_handler = logging.FileHandler('sample.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


# todo a wrapper for chess.white chess.black


class GameManager:
    """
    Objet in charge of playing one game
    """

    GAME_RESULTS = [WIN_FOR_WHITE, WIN_FOR_BLACK, DRAW] = range(3)

    game: ObservableGame

    def __init__(self,
                 game: ObservableGame,
                 syzygy,
                 display_board_evaluator,
                 output_folder_path: path | None,
                 args: GameArgs,
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
        self.game = game
        self.syzygy = syzygy
        self.path_to_store_result = output_folder_path + '/games/' if output_folder_path is not None else None
        self.display_board_evaluator = display_board_evaluator
        self.args = args
        self.player_color_to_id = player_color_to_id
        self.board_history = []
        self.main_thread_mailbox = main_thread_mailbox
        self.player_threads = player_threads

    def external_eval(self):
        return self.display_board_evaluator.evaluate(self.game.board)  # TODO DON'T LIKE THIS writing

    def play_one_move(
            self,
            move: chess.Move
    ) -> None:
        self.game.play_move(move)
        if self.syzygy.fast_in_table(self.game.board):
            print('Theoretically finished with value for white: ',
                  self.syzygy.sting_result(self.game.board))

    def rewind_one_move(self):
        self.game.rewind_one_move()
        if self.syzygy.fast_in_table(self.game.board):
            print('Theoretically finished with value for white: ',
                  self.syzygy.sting_result(self.game.board))

    def set_new_game(self, starting_position_arg):
        self.game.set_starting_position(starting_position_arg)

    def play_one_game(self):
        board = self.game.board
        self.set_new_game(self.args.starting_position)

        color_names = ['Black', 'White']

        while True:
            half_move: int = board.ply()
            print(f'Half Move: {half_move} playing status {self.game.playing_status.status} ')
            color_to_move = board.turn
            color_of_player_to_move_str = color_names[color_to_move]
            print(f'{color_of_player_to_move_str} ({self.player_color_to_id[color_to_move]}) to play now...')

            # waiting for a message
            mail = self.main_thread_mailbox.get()
            self.processing_mail(mail)

            if board.is_game_over() or not self.game_continue_conditions():
                break

        self.tell_results()
        self.terminate_threads()

        return self.simple_results()

    def processing_mail(self, message):
        board = self.game.board

        # TODO maybe implement a class for the message, look at the command pattern
        if message['type'] == 'game_status':
            # update game status
            if message['status'] == 'play':
                self.game.play()
            if message['status'] == 'pause':
                self.game.pause()
        if message['type'] == 'move':
            # play the move
            move = message['move']
            print('receiving the move', move, type(self), self.game.playing_status, type(self.game.playing_status))
            if message['corresponding_board'] == board.fen() and \
                    self.game.playing_status.is_play() and \
                    message['player'] == self.player_color_to_id[board.turn]:
                # TODO THINK! HOW TO DEAL with premoves if we dont know the board in advance?
                self.play_one_move(move)
                eval_sto, eval_chi = self.external_eval()
                print(f'Stockfish evaluation:{eval_sto} and chipiron eval{eval_chi}')
                # Print the board
                board.print_chess_board()
            else:
                pass
                # put back in the queue
                # self.main_thread_mailbox.put(message)
            logger.debug(f'len main tread mailbox {self.main_thread_mailbox.qsize()}')

        if message['type'] == 'move_syzygy':  # TODO IS THIS CLEAN? is this ever called?
            move = self.syzygy.best_move(board)
            self.play_one_move(move)

        if message['type'] == 'back':
            self.rewind_one_move()
            self.game.pause()

    def game_continue_conditions(self):
        half_move = self.game.board.ply()
        continue_bool = True
        if self.args.max_half_moves is not None and half_move >= self.args.max_half_moves:
            assert (1 == 2)
            continue_bool = False
        return continue_bool

    def print_to_file(self, idx=0):
        if self.path_to_store_result is not None:
            path_file = self.path_to_store_result + '_' + str(
                idx) + '_W:' + self.player_color_to_id[chess.WHITE] + '-vs-B:' + self.player_color_to_id[chess.BLACK]
            path_file_txt = path_file + '.txt'
            path_file_obj = path_file + '.obj'
            with open(path_file_txt, 'a') as the_fileText:
                for counter, move in enumerate(self.game.board.move_stack):
                    if counter % 2 == 0:
                        move_1 = move
                    else:
                        the_fileText.write(str(move_1) + ' ' + str(move) + '\n')
            with open(path_file_obj, "wb") as f:
                pickle.dump(self.game.board, f)

    def tell_results(self):
        board = self.game.board
        if self.syzygy.fast_in_table(board):
            print('Syzygy: Theoretical value for white', self.syzygy.sting_result(board))
        if board.is_fivefold_repetition():
            print('is_fivefold_repetition')
        if board.is_seventyfive_moves():
            print('is seventy five  moves')
        if board.is_insufficient_material():
            print('is_insufficient_material')
        if board.is_stalemate():
            print('is_stalemate')
        if board.is_checkmate():
            print('is_checkmate')
        print(board.result())

    def simple_results(self):
        board = self.game.board

        if board.result() == '*':
            if self.syzygy is None or not self.syzygy.fast_in_table(board):  # TODO when is this case useful?
                return (-10000, -10000, -10000)
            else:
                val = self.syzygy.value_white(board, chess.WHITE)
                if val == 0:
                    return self.DRAW
                if val == -1000:
                    return self.WIN_FOR_BLACK
                if val == 1000:
                    return self.WIN_FOR_WHITE
        else:
            result = board.result()
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
