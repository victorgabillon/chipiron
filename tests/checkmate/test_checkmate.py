import pickle
import random
from typing import TYPE_CHECKING

import pytest
from atomheart.board.utils import FenPlusHistory

from chipiron.players.boardevaluators.table_base.factory import create_syzygy
from chipiron.players.factory import create_chipiron_player, create_player
from chipiron.players.player_ids import PlayerConfigTag
from chipiron.scripts.chipiron_args import ImplementationArgs

if TYPE_CHECKING:
    import chess

    from chipiron.players import Player
    from chipiron.players.boardevaluators import table_base
    from valanga.policy import Recommendation


@pytest.mark.parametrize(("use_rusty_board"), (True, False))
def test_check_in_one(use_rusty_board: bool):
    random_generator: random.Random = random.Random(0)

    player: Player = create_chipiron_player(
        implementation_args=ImplementationArgs(use_rust_boards=use_rusty_board),
        universal_behavior=True,
        random_generator=random_generator,
    )

    move_reco: Recommendation = player.select_move(
        fen_plus_history=FenPlusHistory(
            current_fen="1nbqkbnr/rpppp2p/6P1/p6Q/8/8/PPPP1PPP/RNB1KBNR w KQk - 1 5"
        ),
        seed_int=0,
    )

    assert move_reco.move == "g6h7" or move_reco.move == "g6g7"


@pytest.mark.parametrize(("use_rusty_board"), (True, False))
def test_check_in_two(use_rusty_board: bool):
    with open("external_data/puzzles/mate_in_2_db_small.pickle", "rb") as file:
        dict_fen_move = pickle.load(file=file)

    assert dict_fen_move

    random_generator: random.Random = random.Random()

    fen: str
    moves: list[chess.Move]
    print(f"Testing  check in two on {len(dict_fen_move)} boards.")
    for fen, moves in dict_fen_move.items():
        print("fen", fen)

        implementation_args = ImplementationArgs(use_rust_boards=use_rusty_board)

        syzygy_table: table_base.SyzygyTable | None = create_syzygy(
            use_rust=implementation_args.use_rust_boards
        )

        player = create_player(
            args=PlayerConfigTag.UNIFORM_DEPTH_3.get_players_args(),
            syzygy=syzygy_table,
            random_generator=random_generator,
            implementation_args=implementation_args,
            universal_behavior=True,
        )
        move_reco: Recommendation = player.select_move(
            fen_plus_history=FenPlusHistory(current_fen=fen), seed_int=0
        )

        assert move_reco.move == moves[0].uci()


if __name__ == "__main__":
    test_check_in_one(use_rusty_board=True)
    test_check_in_one(use_rusty_board=False)

    test_check_in_two(use_rusty_board=True)
    test_check_in_two(use_rusty_board=False)
