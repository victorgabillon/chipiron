import logging
import os
import pickle
import queue

import chess

import chipiron.players as players_m
from chipiron.environments import HalfMove
from chipiron.environments.chess.board import BoardChi
from chipiron.environments.chess.board.starting_position import AllStartingPositionArgs
from chipiron.games.game.game_playing_status import PlayingStatus
from chipiron.players.boardevaluators.board_evaluator import IGameBoardEvaluator
from chipiron.players.boardevaluators.table_base.syzygy import SyzygyTable
from chipiron.utils import path
from chipiron.utils.communication.gui_messages import GameStatusMessage, BackMessage
from chipiron.utils.communication.player_game_messages import MoveMessage
from chipiron.utils.is_dataclass import IsDataclass
from .final_game_result import GameReport, FinalGameResult
from .game import ObservableGame
from .game_args import GameArgs

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
# file_handler = logging.FileHandler('sample.log')
# file_handler.setLevel(logging.DEBUG)
# file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
# logger.addHandler(file_handler)
logger.addHandler(stream_handler)


# todo a wrapper for chess.white chess.black


class GameManager:
    """
    Objet in charge of playing one game
    """

    game: ObservableGame
    syzygy: SyzygyTable | None
    display_board_evaluator: IGameBoardEvaluator
    output_folder_path: path | None
    args: GameArgs
    player_color_to_id: dict[chess.Color, str]
    main_thread_mailbox: queue.Queue[IsDataclass]
    players: list[players_m.PlayerProcess | players_m.GamePlayer]

    def __init__(
                self,
                game: ObservableGame,
                syzygy: SyzygyTable | None,
                display_board_evaluator: IGameBoardEvaluator,
                output_folder_path: path | None,
                args: GameArgs,
                player_color_to_id: dict[chess.Color, str],
                main_thread_mailbox: queue.Queue[IsDataclass],
                players: list[players_m.PlayerProcess | players_m.GamePlayer],
        ) -> None:
            """
            Constructor for the GameManager Class. If the args, and players are not given a value it is set to None,
            waiting for the set methods to be called. This is done like this so that the players can be changed
            (often swapped) easily.

            Args:
                game (ObservableGame): The observable game object.
                syzygy (SyzygyTable | None): The syzygy table object or None.
                display_board_evaluator (IGameBoardEvaluator): The board evaluator to display an independent evaluation.
                output_folder_path (path | None): The output folder path or None.
                args (GameArgs): The game arguments.
                player_color_to_id (dict[chess.Color, str]): The dictionary mapping player color to player ID.
                main_thread_mailbox (queue.Queue[IsDataclass]): The main thread mailbox.
                players (list[players_m.PlayerProcess | players_m.GamePlayer]): The list of players.

            Returns:
                None
            """
            self.game = game
            self.syzygy = syzygy
            self.path_to_store_result = os.path.join(output_folder_path,
                                                     'games') if output_folder_path is not None else None
            self.display_board_evaluator = display_board_evaluator
            self.args = args
            self.player_color_to_id = player_color_to_id
            self.main_thread_mailbox = main_thread_mailbox
            self.players = players

    def external_eval(self) -> tuple[float, float]:
            """Evaluates the game board using the display board evaluator.

            Returns:
                tuple[float, float]: A tuple containing the evaluation scores.
            """
            return self.display_board_evaluator.evaluate(self.game.board)  # TODO DON'T LIKE THIS writing

    def play_one_move(
                self,
                move: chess.Move
        ) -> None:
            """Play one move in the game.

            Args:
                move (chess.Move): The move to be played.
            """
            self.game.play_move(move)
            if self.syzygy is not None and self.syzygy.fast_in_table(self.game.board):
                print('Theoretically finished with value for white: ',
                      self.syzygy.string_result(self.game.board))

    def rewind_one_move(self) -> None:
        self.game.rewind_one_move()
        if self.syzygy is not None and self.syzygy.fast_in_table(self.game.board):
            print('Theoretically finished with value for white: ',
                  self.syzygy.string_result(self.game.board))

    def set_new_game(
            self,
            starting_position_arg: AllStartingPositionArgs
    ) -> None:
        self.game.set_starting_position(starting_position_arg=starting_position_arg)

    def play_one_game(
            self
    ) -> GameReport:
        board = self.game.board
        self.set_new_game(self.args.starting_position)

        color_names = ['Black', 'White']

        while True:
            half_move: HalfMove = board.ply()
            print(f'Half Move: {half_move} playing status {self.game.playing_status.status} ')
            color_to_move: chess.Color = board.turn
            color_of_player_to_move_str = color_names[color_to_move]
            print(f'{color_of_player_to_move_str} ({self.player_color_to_id[color_to_move]}) to play now...')

            # waiting for a message
            mail = self.main_thread_mailbox.get()
            self.processing_mail(mail)

            if board.is_game_over() or not self.game_continue_conditions():
                break
            else:
                print(f'not game over at {board}')

        self.tell_results()
        self.terminate_processes()
        print('end play_one_game')

        game_results: FinalGameResult = self.simple_results()
        game_report: GameReport = GameReport(
            final_game_result=game_results,
            move_history=board.board.move_stack)
        return game_report

    def processing_mail(
            self,
            message: IsDataclass
    ) -> None:
        board: BoardChi = self.game.board

        match message:
            case MoveMessage():
                move_message: MoveMessage = message
                # play the move
                move: chess.Move = move_message.move
                print('receiving the move', move, type(self), self.game.playing_status, type(self.game.playing_status))
                if move_message.corresponding_board == board.fen() and \
                        self.game.playing_status.is_play() and \
                        message.player_name == self.player_color_to_id[board.turn]:
                    # TODO THINK! HOW TO DEAL with premoves if we dont know the board in advance?
                    print(f'play a move {move} at {board} {self.game.board.board.fen()}')
                    self.play_one_move(move)
                    print(f'now board is  {self.game.board}')

                    eval_sto, eval_chi = self.external_eval()
                    print(f'Stockfish evaluation:{eval_sto} and chipiron eval{eval_chi}')
                    # Print the board
                    board.print_chess_board()

                else:
                    print(f'the move is rejected because one of the following is false \n'
                          f' move_message.corresponding_board == board.fen(){move_message.corresponding_board == board.fen()} \n'
                          f'self.game.playing_status.is_play() {self.game.playing_status.is_play()}\n'
                          f'message.player_name == self.player_color_to_id[board.turn] {message.player_name == self.player_color_to_id[board.turn]}'
                          )
                    print(f'{message.player_name},{self.player_color_to_id[board.turn]}')
                    # put back in the queue
                    # self.main_thread_mailbox.put(message)
                if message.evaluation is not None:
                    self.display_board_evaluator.add_evaluation(
                        player_color=message.color_to_play,
                        evaluation=message.evaluation)
                logger.debug(f'len main tread mailbox {self.main_thread_mailbox.qsize()}')
            case GameStatusMessage():
                game_status_message: GameStatusMessage = message
                # update game status
                if game_status_message.status == PlayingStatus.PLAY:
                    self.game.play()
                if game_status_message.status == PlayingStatus.PAUSE:
                    self.game.pause()
            case BackMessage():
                self.rewind_one_move()
                self.game.pause()
            case other:
                raise ValueError(f'type of message received is not recognized {other} in file {__name__}')

    def game_continue_conditions(
            self
    ) -> bool:
        half_move: HalfMove = self.game.board.ply()
        continue_bool: bool = True
        if self.args.max_half_moves is not None and half_move >= self.args.max_half_moves:
            continue_bool = False
        return continue_bool

    def print_to_file(
            self,
            idx: int = 0
    ) -> None:
        if self.path_to_store_result is not None:
            path_file: path = (f'{self.path_to_store_result}_{idx}_W:{self.player_color_to_id[chess.WHITE]}'
                               f'-vs-B:{self.player_color_to_id[chess.BLACK]}')
            path_file_txt = f'{path_file}.txt'
            path_file_obj = f'{path_file}.obj'
            with open(path_file_txt, 'a') as the_fileText:
                for counter, move in enumerate(self.game.board.board.move_stack):
                    if counter % 2 == 0:
                        move_1 = move
                    else:
                        the_fileText.write(str(move_1) + ' ' + str(move) + '\n')
            with open(path_file_obj, "wb") as f:
                pickle.dump(self.game.board, f)

    def tell_results(self) -> None:
        board = self.game.board
        if self.syzygy is not None and self.syzygy.fast_in_table(board):
            print('Syzygy: Theoretical value for white', self.syzygy.string_result(board))
        if board.board.is_fivefold_repetition():
            print('is_fivefold_repetition')
        if board.board.is_seventyfive_moves():
            print('is seventy five  moves')
        if board.board.is_insufficient_material():
            print('is_insufficient_material')
        if board.board.is_stalemate():
            print('is_stalemate')
        if board.board.is_checkmate():
            print('is_checkmate')
        print(board.board.result())

    def simple_results(self) -> FinalGameResult:
        board = self.game.board

        res: FinalGameResult | None = None
        if board.board.result() == '*':
            if self.syzygy is None or not self.syzygy.fast_in_table(board):
                # useful when a game is stopped
                # before the end, for instance for debugging and profiling
                res = FinalGameResult.DRAW  # arbitrary meaningless choice
                # raise ValueError(f'Problem with figuring our game results in {__name__}')
            else:
                raise ValueError('this case is not coded atm think of what is the write thing to do here!')
        else:
            result = board.board.result()
            if result == '1/2-1/2':
                res = FinalGameResult.DRAW
            if result == '0-1':
                res = FinalGameResult.WIN_FOR_BLACK
            if result == '1-0':
                res = FinalGameResult.WIN_FOR_WHITE
        assert isinstance(res, FinalGameResult)
        return res

    def terminate_processes(self) -> None:
        player: players_m.PlayerProcess | players_m.GamePlayer
        for player in self.players:
            if isinstance(player, players_m.PlayerProcess):
                player.terminate()
                print('stopping the thread')
                # player_thread.join()
                print('thread stopped')
