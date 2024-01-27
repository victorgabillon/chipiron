"""
the one match script
"""
import sys
import queue
import os
import multiprocessing
from PySide6.QtWidgets import QApplication
from scripts.script import Script, ScriptArgs
import chipiron as ch
from dataclasses import dataclass, field
import dacite
from chipiron.players.factory import PlayerArgs
from chipiron.players.utils import fetch_two_players_args_convert_and_save
from chipiron.games.match.utils import fetch_match_games_args_convert_and_save
import chipiron.games.match as match
import chipiron.games.game as game
from chipiron.utils import path


@dataclass
class OneMatchScriptArgs(ScriptArgs):
    """
    The input arguments needed by the one match script to run
    """
    # the seed
    seed: int = 0

    # whether to display the match in a GUI
    gui: bool = False

    experiment_output_folder: path = None

    # path to files with yaml config the players and the match setting.
    config_file_name: path = 'scripts/one_match/inputs/base/exp_options.yaml'
    file_name_player_one: path = 'RecurZipfBase3.yaml'
    file_name_player_two: path = 'Sequool.yaml'
    file_name_match_setting: path = 'setting_duda.yaml'

    # For players and match modification of the yaml file specified in a respective dict
    player_one: dict = field(default_factory=dict)
    player_two: dict = field(default_factory=dict)
    match: dict = field(default_factory=dict)


class OneMatchScript:
    """
    Script that plays a match between two players

    """

    base_experiment_output_folder = os.path.join(Script.base_experiment_output_folder, 'one_match/outputs/')
    base_script: Script

    def __init__(
            self,
            base_script: Script,
    ) -> None:
        """
        Builds the OneMatchScript object
        """

        self.base_script = base_script

        # Calling the init of Script that takes care of a lot of stuff, especially parsing the arguments into self.args
        args_dict: dict = self.base_script.initiate(self.base_experiment_output_folder)

        # Converting the args in the standardized dataclass
        args: OneMatchScriptArgs = dacite.from_dict(data_class=OneMatchScriptArgs,
                                                    data=args_dict)

        if args.gui and args.profiling:
            raise ValueError('Profiling does not work well atm with gui on')

        # Recovering args from yaml file for player and merging with extra args and converting to standardized dataclass
        player_one_args: PlayerArgs
        player_two_args: PlayerArgs
        player_one_args, player_two_args = fetch_two_players_args_convert_and_save(
            file_name_player_one=args.file_name_player_one,
            file_name_player_two=args.file_name_player_two,
            modification_player_one=args.player_one,
            modification_player_two=args.player_two,
            experiment_output_folder=args.experiment_output_folder
        )

        # Recovering args from yaml file for match and game and merging with extra args and converting
        # to standardized dataclass
        match_args: match.MatchArgs
        game_args: game.GameArgs
        match_args, game_args = fetch_match_games_args_convert_and_save(
            profiling=args.profiling,
            file_name_match_setting=args.file_name_match_setting,
            modification=args.match,
            experiment_output_folder=args.experiment_output_folder
        )

        # taking care of random
        ch.set_seeds(seed=args.seed)

        print('self.args.experiment_output_folder', args.experiment_output_folder)
        self.match_manager: ch.game.MatchManager = match.create_match_manager(
            args_match=match_args,
            args_player_one=player_one_args,
            args_player_two=player_two_args,
            output_folder_path=args.experiment_output_folder,
            seed=args.seed,
            args_game=game_args,
            gui=args.gui
        )

        if args.gui:
            # if we use a graphic user interface (GUI) we create it its own thread and
            # create its mailbox to communicate with other threads
            gui_thread_mailbox: queue.Queue = multiprocessing.Manager().Queue()
            self.chess_gui: QApplication = QApplication(sys.argv)
            self.window: ch.disp.MainWindow = ch.disp.MainWindow(
                gui_mailbox=gui_thread_mailbox,
                main_thread_mailbox=self.match_manager.game_manager_factory.main_thread_mailbox
            )
            self.match_manager.subscribe(gui_thread_mailbox)

        self.gui = args.gui

    def run(self) -> None:
        """
        Runs the match either with a GUI or not
        Returns:

        """

        print(' Script One MAtch go')
        # Qt Application needs to be in the main Thread, so we need to distinguish between GUI and no GUI
        if self.gui:  # case with GUI
            # Launching the Match Manager in a Thread
            process_match_manager = multiprocessing.Process(target=self.match_manager.play_one_match)
            process_match_manager.start()

            # Qt Application launched in the main thread
            self.window.show()
            self.chess_gui.exec_()
        else:  # No GUI
            self.match_manager.play_one_match()

        print("finish the run of the match")

        # TODO check the good closing of processes

    def terminate(self) -> None:
        self.base_script.terminate()
