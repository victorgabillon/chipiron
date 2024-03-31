import os
import pickle
import sys
from dataclasses import dataclass

from PySide6.QtWidgets import QApplication

from chipiron.displays.gui_replay_games import MainWindow
from chipiron.environments.chess.board.board import BoardChi
from chipiron.scripts.script import Script


@dataclass
class ReplayScriptArgs:
    """
    The input arguments needed by the one match script to run
    """

    # path to the pickle file with the BoardChi stored
    file_path_game_pickle: str

    # whether to display the match in a GUI
    gui: bool = False


class ReplayGameScript:
    args_dataclass_name: type[ReplayScriptArgs] = ReplayScriptArgs

    base_script: Script
    chess_board: BoardChi

    base_experiment_output_folder = os.path.join(Script.base_experiment_output_folder, 'replay_game/outputs/')

    def __init__(
            self,
            base_script: Script,
    ) -> None:
        """
        Builds the OneMatchScript object
        """

        self.base_script = base_script

        # Calling the init of Script that takes care of a lot of stuff, especially parsing the arguments into self.args
        self.args: ReplayScriptArgs = self.base_script.initiate(
            args_dataclass_name=ReplayScriptArgs,
            base_experiment_output_folder=self.base_experiment_output_folder,

        )

        with open(self.args.file_path_game_pickle, 'rb') as fileGame:
            self.chess_board: BoardChi = pickle.load(fileGame)

    def run(self) -> None:
        if self.args.gui:
            chess_gui = QApplication(sys.argv)
            window = MainWindow(self.chess_board)
            window.show()
            chess_gui.exec_()
        else:
            # TODO: code a console version
            ...

    def terminate(self) -> None:
        """
        Finishing the script. Profiling or timing.
        """
        ...
