from chipiron.games.match.match_manager import MatchManager
from chipiron.games.game.game_manager_factory import GameManagerFactory
from chipiron.games.match.match_results_factory import MatchResultsFactory
from chipiron.players.boardevaluators.factory import create_game_board_evaluator
from chipiron.utils import path
from chipiron.players.boardevaluators.table_base.syzygy import SyzygyTable
from chipiron.players.boardevaluators.table_base.factory import create_syzygy_thread
from chipiron.players.factory import PlayerArgs
from .match_args import MatchArgs
import chipiron.games.game as game
import multiprocessing
import queue


def create_match_manager(
        args_match: MatchArgs,
        args_player_one: PlayerArgs,
        args_player_two: PlayerArgs,
        args_game: game.GameArgs,
        seed: int | None = None,
        output_folder_path: path = None,
        gui: bool = False
) -> MatchManager:
    main_thread_mailbox: queue.Queue = multiprocessing.Manager().Queue()

    # Creation of the Syzygy table for perfect play in low pieces cases, needed by the GameManager
    # and can also be used by the players
    syzygy_mailbox: SyzygyTable = create_syzygy_thread()

    player_one_name: str = args_player_one.name
    player_two_name: str = args_player_two.name

    game_board_evaluator  = create_game_board_evaluator(gui=gui)

    game_manager_factory: GameManagerFactory = GameManagerFactory(
        syzygy_table=syzygy_mailbox,
        game_manager_board_evaluator=game_board_evaluator,
        output_folder_path=output_folder_path,
        main_thread_mailbox=main_thread_mailbox,
    )

    match_results_factory: MatchResultsFactory = MatchResultsFactory(
        player_one_name=player_one_name,
        player_two_name=player_two_name
    )

    game_args_factory: game.GameArgsFactory = game.GameArgsFactory(
        args_match=args_match,
        args_player_one=args_player_one,
        args_player_two=args_player_two,
        seed=seed,
        args_game=args_game
    )

    match_manager: MatchManager = MatchManager(
        player_one_id=player_one_name,
        player_two_id=player_two_name,
        game_manager_factory=game_manager_factory,
        game_args_factory=game_args_factory,
        match_results_factory=match_results_factory,
        output_folder_path=output_folder_path
    )
    return match_manager
