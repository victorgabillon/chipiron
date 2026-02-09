# pylint: disable=duplicate-code
"""Describe script generates datasets of chess board positions by sampling games from the Lichess monthly PGN database.

It supports dynamic, on-the-fly downloading and decompression of monthly PGN files, extracting board positions at
specified move intervals, and saving progress with metadata. The dataset creation process is configurable via
sampling frequency, offset, random seed, and month range. Intermediate and final saves include detailed statistics
and metadata for reproducibility. The script can be run as a CLI tool to automate dataset generation across multiple
months, with options to manage downloaded files and control sampling parameters.
Main functionalities:
- Download and verify monthly Lichess PGN dumps (.zst format).
- Decompress PGN files using either the zstd CLI or Python zstandard library.
- Stream games from PGN files, sample board positions at configurable intervals.
- Save progress and final datasets as pandas DataFrames with rich metadata.
- CLI interface for flexible configuration and automation.
Dependencies:
- python-chess
- pandas
- zstandard
- chipiron (internal utilities for logging and path management)
"""

import random
from pathlib import Path

import chess
import chess.pgn
from atomheart.board.utils import Fen

from chipiron.scripts.generate_datasets.monthly_pgn_pipeline import (
    add_monthly_cli_args,
    build_monthly_pipeline_config,
    build_monthly_run_kwargs,
    finalize_monthly_dataset_from_results,
    run_monthly_pipeline,
)
from chipiron.utils.logger import chipiron_logger
from chipiron.utils.path_variables import EXTERNAL_DATA_DIR

# Sampling configuration variables
DEFAULT_SAMPLING_FREQUENCY = 50
DEFAULT_OFFSET_MIN = 5
DEFAULT_RANDOM_SEED: int | None = 0  # Set to an int for reproducibility


def process_game(
    game: chess.pgn.GameNode,
    total_count_move: int,
    the_dic: list[dict[str, Fen]],
    sampling_frequency: int,
    offset_min: int = DEFAULT_OFFSET_MIN,
) -> int:
    """Process a single game and extract positions (no eval requirement)."""
    chess_board: chess.Board = game.board()
    current_node: chess.pgn.GameNode = game

    # Normalize offset_min
    offset_min = max(offset_min, 0)

    # Materialize moves to know game length
    moves_list = list(game.mainline_moves())
    game_total_moves = len(moves_list)
    effective_min = min(offset_min, game_total_moves)

    # Draw random offset within game length
    if game_total_moves >= effective_min:
        random_offset = random.randint(effective_min, game_total_moves)
    else:
        random_offset = game_total_moves  # degenerate case

    for game_move_count, move in enumerate(moves_list, start=1):
        total_count_move += 1
        chess_board.push(move)

        next_node: chess.pgn.GameNode | None = current_node.next()
        if next_node is None:
            break  # Safety check for end of game

        current_node = next_node

        # Check if we should sample this position based on offset + frequency
        # Sample at: offset, offset + frequency, offset + 2*frequency, etc.
        if (
            game_move_count >= random_offset
            and (game_move_count - random_offset) % sampling_frequency == 0
        ):
            # Only store FEN strings since that's all we need
            the_dic.append(
                {
                    "fen": chess_board.fen(),
                }
            )
    return total_count_move


# --- Dynamic monthly download helpers ---


def generate_board_dataset_multi_months(
    output_file_path: str,
    max_boards: int = 10_000_000,
    sampling_frequency: int = DEFAULT_SAMPLING_FREQUENCY,
    offset_min: int = DEFAULT_OFFSET_MIN,
    seed: int | None = DEFAULT_RANDOM_SEED,
    start_month: str = "2015-03",
    max_months: int | None = None,
    delete_pgn_after_use: bool = True,
    intermediate_every_games: int = 10_000,
    dest_dir: Path | None = None,
) -> None:
    """Generate dataset streaming through monthly Lichess dumps downloaded on-the-fly.

    Stops when max_boards collected or month limit reached. Each month PGN is deleted when done (optional).
    """
    cfg = build_monthly_pipeline_config(
        output_file_path=output_file_path,
        max_boards=max_boards,
        seed=seed,
        start_month=start_month,
        max_months=max_months,
        delete_pgn_after_use=delete_pgn_after_use,
        intermediate_every_games=intermediate_every_games,
        dest_dir=dest_dir,
    )

    the_dic: list[dict[str, Fen]] = []

    results = run_monthly_pipeline(
        cfg=cfg,
        the_dic=the_dic,
        read_game=chess.pgn.read_game,
        process_game_fn=process_game,
        sampling_frequency=sampling_frequency,
        offset_min=offset_min,
    )

    finalize_monthly_dataset_from_results(
        cfg=cfg,
        the_dic=the_dic,
        results=results,
        sampling_frequency=sampling_frequency,
        offset_min=offset_min,
    )


# --- CLI integration (dynamic only) ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate chess board dataset via on-the-fly monthly Lichess downloads (dynamic only)"
    )
    add_monthly_cli_args(
        parser,
        default_output=str(EXTERNAL_DATA_DIR / "datasets" / "only_boards.pkl"),
        default_max_boards=10_000_000,
        default_seed=0,
    )
    parser.add_argument(
        "--sampling-frequency", type=int, default=DEFAULT_SAMPLING_FREQUENCY
    )
    parser.add_argument("--offset-min", type=int, default=DEFAULT_OFFSET_MIN)
    args = parser.parse_args()

    chipiron_logger.info(
        "Running dynamic monthly download mode (legacy single-file mode disabled)"
    )
    generate_board_dataset_multi_months(
        **build_monthly_run_kwargs(args),
        sampling_frequency=args.sampling_frequency,
        offset_min=args.offset_min,
    )
