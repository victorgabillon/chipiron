"""
This module contains the `ReplayGameScript` class, which is responsible for replaying a chess game.
"""

import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import dacite
import yaml
from PySide6.QtWidgets import QApplication

from chipiron.displays.gui_replay_games import MainWindow
from chipiron.environments.chess.board.board_chi import BoardChi
from chipiron.environments.chess.board.factory import create_board_chi
from chipiron.environments.chess.board.utils import FenPlusHistory
from chipiron.games.game.final_game_result import GameReport
from chipiron.scripts.script import Script
from chipiron.scripts.script_args import BaseScriptArgs


@dataclass
class ReplayScriptArgs:
    """
    The input arguments needed by the replay game script to run.
    """

    # path to the yaml file with the Game Report stored
    file_game_report: str = (
        "chipiron/scripts/one_match/outputs/Sunday-10-13-2024--22:40:58:049918/games_0_W:Sequool-vs-B:Random_game_report.yaml"
    )

    base_script_args: BaseScriptArgs = field(default_factory=BaseScriptArgs)

    # whether to display the match in a GUI
    gui: bool = False


class ReplayGameScript:
    """
    The `ReplayGameScript` class is responsible for replaying a chess game.
    """

    args_dataclass_name: type[ReplayScriptArgs] = ReplayScriptArgs

    base_script: Script[ReplayScriptArgs]
    chess_board: BoardChi

    base_experiment_output_folder = os.path.join(
        Script.base_experiment_output_folder, "replay_game/outputs/"
    )

    def __init__(
        self,
        base_script: Script[ReplayScriptArgs],
    ) -> None:
        """
        Initializes the `ReplayGameScript` object.

        Args:
            base_script (Script): The base script object.

        """
        self.base_script = base_script

        # Calling the init of Script that takes care of a lot of stuff, especially parsing the arguments into self.args
        self.args: ReplayScriptArgs = self.base_script.initiate(
            experiment_output_folder=self.base_experiment_output_folder,
        )

        with open(self.args.file_game_report, "r") as fileGame:
            game_report_dict: dict[Any, Any] = yaml.safe_load(fileGame)
            game_report: GameReport = dacite.from_dict(
                data_class=GameReport,
                data=game_report_dict,
                config=dacite.Config(cast=[Enum]),
            )
            self.chess_board: BoardChi = create_board_chi(
                fen_with_history=FenPlusHistory(
                    current_fen=game_report.fen_history[0],
                    historical_moves=game_report.move_history,
                )
            )

    def run(self) -> None:
        """
        Runs the replay game script.

        If `gui` is set to True, it displays the match in a GUI. Otherwise, it runs a console version (TODO).

        """
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
        Finishes the script.

        Performs any necessary cleanup or finalization steps.

        """
        ...
