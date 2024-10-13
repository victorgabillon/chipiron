"""
Module in charge of managing the game. It is the main class that will be used to play a game.
"""

import logging
import os
import queue
from dataclasses import asdict

import chess
import yaml

import chipiron.players as players_m
from chipiron.environments import HalfMove
from chipiron.environments.chess.board import BoardChi
from chipiron.games.game.game_playing_status import PlayingStatus
from chipiron.players.boardevaluators.board_evaluator import IGameBoardEvaluator
from chipiron.players.boardevaluators.table_base.syzygy import SyzygyTable
from chipiron.utils import path
from chipiron.utils.communication.gui_messages import GameStatusMessage, BackMessage
from chipiron.utils.communication.player_game_messages import MoveMessage
from chipiron.utils.dataclass import IsDataclass
from chipiron.utils.dataclass import custom_asdict_factory
from .final_game_result import GameReport, FinalGameResult
from .game import ObservableGame
from .game_args import GameArgs

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class GameManager:
    """
    Object in charge of playing one game
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
        return self.display_board_evaluator.evaluate(self.game.board)

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
        """
        Rewinds the game by one move.

        This method rewinds the game by one move, undoing the last move made.
        If the game has a Syzygy tablebase loaded and the current board position is in the tablebase,
        it prints the theoretically finished value for white.

        Returns:
            None
        """
        self.game.rewind_one_move()
        if self.syzygy is not None and self.syzygy.fast_in_table(self.game.board):
            print('Theoretically finished with value for white: ',
                  self.syzygy.string_result(self.game.board))

    def play_one_game(
            self
    ) -> GameReport:
        """
        Play one game.

        Returns:
            GameReport: The report of the game.
        """
        board = self.game.board

        color_names = ['Black', 'White']

        while True:

            half_move: HalfMove = board.ply()
            print(f'Half Move: {half_move} playing status {self.game.playing_status.status} ')
            color_to_move: chess.Color = board.turn
            color_of_player_to_move_str = color_names[color_to_move]
            print(f'{color_of_player_to_move_str} ({self.player_color_to_id[color_to_move]}) to play now...')

            # sending the current board to the player (and possibly gui) and asking for a move
            self.game.play()

            # waiting for a message
            mail = self.main_thread_mailbox.get()
            self.processing_mail(mail)

            if board.is_game_over() or not self.game_continue_conditions():
                break
            else:
                print(f'Not game over at {board}')

        self.tell_results()
        self.terminate_processes()
        print('end play_one_game')

        game_results: FinalGameResult = self.simple_results()
        game_report: GameReport = GameReport(
            final_game_result=game_results,
            move_history=self.game.move_history,
            fen_history=self.game.fen_history
        )
        return game_report

    def processing_mail(
            self,
            message: IsDataclass
    ) -> None:
        """
        Process the incoming mail message.

        Args:
            message (IsDataclass): The incoming mail message.

        Returns:
            None
        """

        board: BoardChi = self.game.board

        match message:
            case MoveMessage():
                move_message: MoveMessage = message
                # play the move
                move: chess.Move = move_message.move
                print('receiving the move', move, type(self), self.game.playing_status, type(self.game.playing_status))
                if move_message.corresponding_board == board.fen and \
                        self.game.playing_status.is_play() and \
                        message.player_name == self.player_color_to_id[board.turn]:
                    print(f'play a move {move} at {board} {self.game.board.fen}')
                    self.play_one_move(move)
                    print(f'now board is  {self.game.board}')

                    eval_sto, eval_chi = self.external_eval()
                    print(f'Stockfish evaluation:{eval_sto} and chipiron eval{eval_chi}')
                    # Print the board
                    board.print_chess_board()

                else:
                    print(f'the move is rejected because one of the following is false \n'
                          f' move_message.corresponding_board == board.fen{move_message.corresponding_board == board.fen} \n'
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
        """
        Checks the conditions for continuing the game.

        Returns:
            bool: True if the game should continue, False otherwise.
        """
        half_move: HalfMove = self.game.board.ply()
        continue_bool: bool = True
        if self.args.max_half_moves is not None and half_move >= self.args.max_half_moves:
            continue_bool = False

        return continue_bool

    def print_to_file(
            self,
            game_report: GameReport,
            idx: int = 0
    ) -> None:
        """
        Print the moves of the game to a yaml file and a more human-readable text file.

        Args:
            game_report(GameReport): a game report to be printed
            idx (int): The index to include in the file name (default is 0).

        Returns:
            None
        """
        # todo probably the txt file should be a valid PGN file : https://en.wikipedia.org/wiki/Portable_Game_Notation
        if self.path_to_store_result is not None:
            path_file: path = (f'{self.path_to_store_result}_{idx}_W:{self.player_color_to_id[chess.WHITE]}'
                               f'-vs-B:{self.player_color_to_id[chess.BLACK]}')
            path_file_obj = f'{path_file}_game_report.yaml'
            path_file_txt = f'{path_file}.txt'
            with open(path_file_txt, 'a') as the_fileText:
                for counter, move in enumerate(self.game.move_history):
                    if counter % 2 == 0:
                        move_1 = move
                    else:
                        the_fileText.write(str(move_1) + ' ' + str(move) + '\n')
            with open(path_file_obj, "w") as file:
                yaml.dump(asdict(game_report, dict_factory=custom_asdict_factory), file, default_flow_style=False)

    def tell_results(self) -> None:
        """
        Prints the results of the game based on the current state of the board.

        The method checks various conditions on the board and prints the corresponding result.
        It also checks for specific conditions like syzygy, fivefold repetition, seventy-five moves,
        insufficient material, stalemate, and checkmate.

        Returns:
            None
        """
        board = self.game.board
        if self.syzygy is not None and self.syzygy.fast_in_table(board):
            print('Syzygy: Theoretical value for white', self.syzygy.string_result(board))
        board.tell_result()

    def simple_results(self) -> FinalGameResult:
        """
        Determines the final result of the game based on the current state of the board.

        Returns:
            FinalGameResult: The final result of the game.
        """
        board = self.game.board

        res: FinalGameResult | None = None
        if board.result() == '*':
            if self.syzygy is None or not self.syzygy.fast_in_table(board):
                # useful when a game is stopped
                # before the end, for instance for debugging and profiling
                res = FinalGameResult.DRAW  # arbitrary meaningless choice
                # raise ValueError(f'Problem with figuring our game results in {__name__}')
            else:
                raise ValueError('this case is not coded atm think of what is the right thing to do here!')
        else:
            result = board.result()
            if result == '1/2-1/2':
                res = FinalGameResult.DRAW
            if result == '0-1':
                res = FinalGameResult.WIN_FOR_BLACK
            if result == '1-0':
                res = FinalGameResult.WIN_FOR_WHITE
        assert isinstance(res, FinalGameResult)
        return res

    def terminate_processes(self) -> None:
        """
        Terminates all player processes and stops the associated threads.

        This method iterates over the list of players and terminates any player processes found.
        If a player process is found, it is terminated and the associated thread is stopped.

        Note:
        - This method assumes that the `players` attribute is a list of `PlayerProcess` or `GamePlayer` objects.

        Returns:
        None
        """
        player: players_m.PlayerProcess | players_m.GamePlayer
        for player in self.players:
            if isinstance(player, players_m.PlayerProcess):
                player.terminate()
                print('stopping the thread')
                # player_thread.join()
                print('thread stopped')
