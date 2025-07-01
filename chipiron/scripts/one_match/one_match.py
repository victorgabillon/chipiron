"""
the one match script
"""

import multiprocessing
import os
import queue
import sys
from dataclasses import asdict, dataclass, field
from typing import cast

import yaml
from PySide6.QtWidgets import QApplication

import chipiron as ch
import chipiron.displays as display
from chipiron.environments.chess.board import BoardFactory, create_board_factory
from chipiron.games.match.match_args import MatchArgs
from chipiron.games.match.match_factories import create_match_manager_from_args
from chipiron.scripts.chipiron_args import ImplementationArgs
from chipiron.scripts.script import Script
from chipiron.scripts.script_args import BaseScriptArgs
from chipiron.utils.dataclass import IsDataclass
from chipiron.utils.logger import chipiron_logger


@dataclass
class MatchScriptArgs:
    """
    The input arguments needed by the one match script to run
    """

    base_script_args: BaseScriptArgs = field(default_factory=BaseScriptArgs)
    match_args: MatchArgs = field(default_factory=MatchArgs)
    implementation_args: ImplementationArgs = field(default_factory=ImplementationArgs)
    # whether to display the match in a GUI
    gui: bool = True


class OneMatchScript:
    """
    Script that plays a match between two players

    """

    args_dataclass_name: type[MatchScriptArgs] = MatchScriptArgs

    base_experiment_output_folder = os.path.join(
        Script.base_experiment_output_folder, "one_match/outputs/"
    )
    base_script: Script[MatchScriptArgs]

    chess_gui: QApplication

    def __init__(
        self,
        base_script: Script[MatchScriptArgs],
    ) -> None:
        """
        Builds the OneMatchScript object
        """

        self.base_script = base_script

        # Calling the init of Script that takes care of a lot of stuff, especially parsing the arguments into args
        args: MatchScriptArgs = self.base_script.initiate(
            experiment_output_folder=self.base_experiment_output_folder,
        )

        # creating the match manager
        self.match_manager: ch.game.MatchManager = create_match_manager_from_args(
            match_args=args.match_args,
            base_script_args=args.base_script_args,
            implementation_args=args.implementation_args,
            gui=args.gui,
        )

        # saving the arguments of the script
        with open(
            os.path.join(
                str(args.base_script_args.experiment_output_folder),
                "inputs_and_parsing/one_match_script_merge.yaml",
            ),
            "w",
        ) as one_match_script:
            yaml.dump(asdict(args), one_match_script, default_flow_style=False)

        # checking for some incompatibility
        if args.gui and args.base_script_args.profiling:
            raise ValueError("Profiling does not work well atm with gui on")

        # If we need a GUI
        if args.gui:
            # if we use a graphic user interface (GUI) we create it its own thread and
            # create its mailbox to communicate with other threads
            gui_thread_mailbox: queue.Queue[IsDataclass] = (
                multiprocessing.Manager().Queue()
            )

            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)
            else:
                app = cast(QApplication, app)
            self.chess_gui = app

            board_factory: BoardFactory = create_board_factory(
                use_rust_boards=args.implementation_args.use_rust_boards,
                sort_legal_moves=args.base_script_args.universal_behavior,
            )
            self.window: display.MainWindow = display.MainWindow(
                gui_mailbox=gui_thread_mailbox,
                main_thread_mailbox=self.match_manager.game_manager_factory.main_thread_mailbox,
                board_factory=board_factory,
            )
            self.match_manager.subscribe(gui_thread_mailbox)

        self.gui = args.gui

    def run(self) -> None:
        """
        Runs the match either with a GUI or not
        Returns:

        """

        chipiron_logger.info(" Script One Match go")
        # Qt Application needs to be in the main Thread, so we need to distinguish between GUI and no GUI
        if self.gui:  # case with GUI
            # Launching the Match Manager in a Thread
            self.process_match_manager = multiprocessing.Process(
                target=self.match_manager.play_one_match
            )
            self.process_match_manager.start()

            # Qt Application launched in the main thread
            self.window.show()
            self.chess_gui.exec_()
        else:  # No GUI
            self.match_manager.play_one_match()

        chipiron_logger.info("Finish the run of the match")

        # TODO check the good closing of processes

    def terminate(self) -> None:
        """
        Terminates the script and cleans up any resources.
        """
        chipiron_logger.info("terminating script")
        self.base_script.terminate()
        if self.gui:
            self.process_match_manager.terminate()
