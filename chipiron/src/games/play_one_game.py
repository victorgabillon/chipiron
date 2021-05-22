from src.games.game import Game
from src.displays.display_one_game import DisplayOneGame
import time
import chess
import pickle
import global_variables
from src.players.boardevaluators.stockfish_evaluation import Stockfish

# todo a wrapper for chess.white chess.black


class PlayOneGame:
    """
    Objet in charge of playing one game
    """

    def __init__(self, args, player_white, player_black, chess_simulator, syzygy, path_to_store_result=None):
        self.game = Game(chess_simulator, args['starting_position'], syzygy)

        self.seconds_per_move_white = float(args['secondsPerMoveWhite'])
        self.seconds_per_move_black = float(args['secondsPerMoveBlack'])

        self.display_option = args['displayPositionOption']
        if self.display_option:
            self.displayOneGame = DisplayOneGame()

        self.player_white = player_white
        self.player_black = player_black
        self.path_to_store_result = path_to_store_result

        self.board_evaluator = Stockfish()

    def display(self, round_, color_to_play):
        if self.display_option:
            self.displayOneGame.displayBoard(self.game.board, round_, self.game.last_move(), color_to_play)

    def stockfish_eval(self):
        return self.board_evaluator.score(self.game.board.chess_board)

    def play_the_game(self):
        time.sleep(.1)
        global_variables.global_lock.acquire()

        self.allow_display()
        if not self.game.is_finished():
            starting_round = self.game.get_current_round()
            self.gameRound = starting_round  # probably the round counting is wrong if black starts
            # but this is minor at the moment
            if self.game.who_plays() == chess.BLACK:  # case where we are resuming a game stopped in the middle
                self.display(starting_round, chess.BLACK)
                self.black_play()
                self.allow_display()
                starting_round = starting_round + 1
            else:
                self.display(starting_round, chess.WHITE)

            self.gameRound = starting_round

            while not self.game.is_finished():
                print('round ', self.gameRound)
                self.white_play()
                self.allow_display()
                #  input("Press Enter to continue...")

                if global_variables.profiling_bool:
                    break

                if not self.game.is_finished():
                    self.black_play()
                    #   input("Press Enter to continue...")
                    self.allow_display()
                    self.gameRound = self.gameRound + 1
        #       input("Press Enter to continue...")

        self.game.tell_results()
        self.print_to_file()

        global_variables.global_lock.release()
        return self.game.simple_results()

    def allow_display(self):
        global_variables.global_lock.release()
        time.sleep(.101)
        global_variables.global_lock.acquire()

    def white_play(self):
        print('white (' + self.player_white.player_name + ') to play now... ')
        move1 = self.player_white.get_move(self.game.board, self.seconds_per_move_white)

        print('white: ' + self.player_white.player_name + ' plays ', move1)
        if self.player_white.player_name != 'Human':
            if not self.game.is_legal(move1):  # check if its a leagal moves!!!!!!
                raise Exception('illegal move from player white: ' + str(move1))
            self.game.play(move1)
        self.game.board.print_chess_board()

        self.display(2 * self.gameRound, chess.BLACK)

    def black_play(self):
        print('black (' + self.player_black.player_name + ') to play now... ')
        move2 = self.player_black.get_move(self.game.board, self.seconds_per_move_black)
        print('black: ' + self.player_black.player_name + ' plays ', move2, type(move2))

        if self.player_black.player_name != 'Human':
            if not self.game.is_legal(move2):  # check if its a legal moves!!!!!!
                #   print('illegal move from player 2')
                raise Exception('illegal move from player black: ' + str(move2))
            #     sys.exit("illegal move from player 2")
            self.game.play(move2)
        self.game.board.print_chess_board()

        self.display(2 * self.gameRound + 1, chess.WHITE)

    def print_to_file(self):
        if self.path_to_store_result is not None:
            path_file = self.path_to_store_result + '_W:' + self.player_white.player_name + '-vs-B:' + self.player_black.player_name
            path_file_txt = path_file + '.txt'
            path_file_obj = path_file + '.obj'
            with open(path_file_txt, 'a') as the_fileText:
                for counter, move in enumerate(self.game.board.chess_board.move_stack):
                    if counter % 2 == 0:
                        move_1 = move
                    else:
                        the_fileText.write(str(move_1) + ' ' + str(move) + '\n')
            with open(path_file_obj, "wb") as f:
                pickle.dump(self.game.board.chess_board, f)
