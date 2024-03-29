"""
the one match script
"""
import multiprocessing
import os
import queue
import sys
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any

import dacite
import yaml
from PySide6.QtWidgets import QApplication

import chipiron as ch
from chipiron.games.match.match_args import MatchArgs
from chipiron.games.match.match_factories import create_match_manager_from_args
from chipiron.scripts.script import Script
from chipiron.scripts.script import ScriptArgs
from chipiron.utils.is_dataclass import IsDataclass


@dataclass
class MatchScriptArgs(ScriptArgs, MatchArgs):
    """
    The input arguments needed by the one match script to run
    """

    # whether to display the match in a GUI
    gui: bool = True


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
        args_dict: dict[str, Any] = self.base_script.initiate(
            base_experiment_output_folder=self.base_experiment_output_folder
        )

        # Converting the args in the standardized dataclass
        args: MatchScriptArgs = dacite.from_dict(
            data_class=MatchScriptArgs,
            data=args_dict
        )

        # creating the match manager
        self.match_manager: ch.game.MatchManager = create_match_manager_from_args(
            args=args,
            profiling=args.profiling,
            gui=args.gui,
            testing=args.testing
        )

        # saving the arguments of the script
        with open(os.path.join(str(args.experiment_output_folder), 'inputs_and_parsing/one_match_script_merge.yaml'),
                  'w') as one_match_script:
            yaml.dump(asdict(args), one_match_script, default_flow_style=False)

        # checking for some incompatibility
        if args.gui and args.profiling:
            raise ValueError('Profiling does not work well atm with gui on')

        # If we need a GUI
        if args.gui:
            # if we use a graphic user interface (GUI) we create it its own thread and
            # create its mailbox to communicate with other threads
            gui_thread_mailbox: queue.Queue[IsDataclass] = multiprocessing.Manager().Queue()
            self.chess_gui: QApplication = QApplication(sys.argv)
            self.window: ch.disp.MainWindow = ch.disp.MainWindow(
                gui_mailbox=gui_thread_mailbox,
                main_thread_mailbox=self.match_manager.game_manager_factory.main_thread_mailbox
            )
            self.match_manager.subscribe(gui_thread_mailbox)

        self.gui = args.gui

    def run(
            self
    ) -> None:
        """
        Runs the match either with a GUI or not
        Returns:

        """

        print(' Script One Match go')
        # Qt Application needs to be in the main Thread, so we need to distinguish between GUI and no GUI
        if self.gui:  # case with GUI
            # Launching the Match Manager in a Thread
            self.process_match_manager = multiprocessing.Process(target=self.match_manager.play_one_match)
            self.process_match_manager.start()

            # Qt Application launched in the main thread
            self.window.show()
            self.chess_gui.exec_()
        else:  # No GUI
            self.match_manager.play_one_match()

        print("finish the run of the match")

        # TODO check the good closing of processes

    def terminate(
            self
    ) -> None:
        print('terminating script')
        self.base_script.terminate()
        if self.gui:
            self.process_match_manager.terminate()
