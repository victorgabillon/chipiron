from dataclasses import dataclass
from typing import Literal

import chess

from chipiron.environments.chess.board import board as boards
from chipiron.utils import seed
from .move_selector import MoveRecommendation
from .move_selector_types import MoveSelectorTypes


@dataclass
class CommandLineHumanPlayerArgs:
    type: Literal[MoveSelectorTypes.CommandLineHuman]  # for serialization


@dataclass
class GuiHumanPlayerArgs:
    type: Literal[MoveSelectorTypes.GuiHuman]  # for serialization


class CommandLineHumanMoveSelector:

    def select_move(
            self,
            board: boards.BoardChi,
            move_seed: seed
    ) -> MoveRecommendation:
        print(f'Legal Moves {board.legal_moves}')
        legal_moves_uci: list[str] = []
        for move in board.legal_moves:
            legal_moves_uci.append(move.uci())
        print(f'Legal Moves {legal_moves_uci}')

        good_move: bool = False
        while not good_move:
            move_uci: str = input('Input your move')
            print(f'choice of move {move_uci}')
            if move_uci in legal_moves_uci:
                good_move = True
            else:
                print('Bad move, Not legal')

        assert move_uci in legal_moves_uci

        return MoveRecommendation(
            move=chess.Move.from_uci(move_uci),
            evaluation=None
        )
