"""Document the module contains the implementation of the BaseTreeExplorationScript class.

The BaseTreeExplorationScript class is responsible for running a script that performs base tree exploration in a chess game.
"""

import os
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from atomheart.board.factory import create_board

from chipiron.players.adapters.chess_syzygy_oracle import (
    ChessSyzygyPolicyOracle,
    ChessSyzygyTerminalOracle,
    ChessSyzygyValueOracle,
)
from chipiron.players.boardevaluators.table_base.factory import create_syzygy
from chipiron.players.factory import create_chess_player
from chipiron.players.player_ids import PlayerConfigTag
from chipiron.scripts.chipiron_args import ImplementationArgs
from chipiron.scripts.script import Script
from chipiron.scripts.script_args import BaseScriptArgs

if TYPE_CHECKING:
    from atomheart.board.utils import FenPlusHistory

    from chipiron.environments.chess.types import ChessState
    from chipiron.players.chess_player_args import ChessPlayerArgs
    from chipiron.players.player import Player


@dataclass
class BaseTreeExplorationArgs:
    """Data container for BaseTreeExplorationArgs."""

    base_script_args: BaseScriptArgs = field(default_factory=BaseScriptArgs)
    implementation_args: ImplementationArgs = field(default_factory=ImplementationArgs)


class BaseTreeExplorationScript:
    """The BaseTreeExplorationScript."""

    args_dataclass_name: type[BaseTreeExplorationArgs] = BaseTreeExplorationArgs
    base_experiment_output_folder = os.path.join(
        Script.base_experiment_output_folder, "base_tree_exploration/outputs/"
    )
    base_script: Script[BaseTreeExplorationArgs]

    def __init__(self, base_script: Script[BaseTreeExplorationArgs]) -> None:
        """Initialize a new instance of the BaseTreeExploration class.

        Args:
            base_script (Script): The base script to be used for tree exploration.

        """
        self.base_script = base_script

        # Calling the init of Script that takes care of a lot of stuff, especially parsing the arguments into args
        self.args: BaseTreeExplorationArgs = self.base_script.initiate(
            experiment_output_folder=self.base_experiment_output_folder,
        )

    def run(self) -> None:
        """Run the base tree exploration script."""
        syzygy = create_syzygy(use_rust=self.args.implementation_args.use_rust_boards)

        player_one_args: ChessPlayerArgs = PlayerConfigTag.UNIFORM.get_players_args()

        random_generator = random.Random()
        random_generator.seed(self.args.implementation_args.use_rust_boards)
        player: Player[FenPlusHistory, ChessState] = create_chess_player(
            args=player_one_args,
            terminal_oracle=ChessSyzygyTerminalOracle(syzygy)
            if syzygy is not None
            else None,
            value_oracle=ChessSyzygyValueOracle(syzygy) if syzygy is not None else None,
            policy_oracle=ChessSyzygyPolicyOracle(syzygy)
            if syzygy is not None
            else None,
            random_generator=random_generator,
            implementation_args=self.args.implementation_args,
            universal_behavior=self.args.base_script_args.universal_behavior,
        )

        board = create_board(
            use_rust_boards=self.args.implementation_args.use_rust_boards,
            use_board_modification=self.args.implementation_args.use_board_modification,
        )
        player.select_move(
            state_snapshot=board.into_fen_plus_history(),
            seed=self.args.implementation_args.use_rust_boards,
        )

    def terminate(self) -> None:
        """Terminates the base tree exploration script."""
        self.base_script.terminate()

    @classmethod
    def get_args_dataclass_name(cls) -> type[BaseTreeExplorationArgs]:
        """Return the dataclass type that holds the arguments for the script.

        Returns:
            type: The dataclass type for the script's arguments.

        """
        return BaseTreeExplorationArgs
