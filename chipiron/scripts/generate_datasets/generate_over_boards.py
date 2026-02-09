"""Generate chess over boards dataset from Lichess PGN dumps.

This module creates datasets of game-ending chess positions (checkmate, stalemate, insufficient material)
by processing monthly Lichess PGN dumps. It follows the same structure as generate_boards.py and reuses
many of its functions for consistency.

Key features:
- Download and process monthly Lichess PGN dumps (.zst format).
- Extract only game-ending positions from completed games.
- Save progress and final datasets with metadata.
- Support for both dynamic monthly processing and legacy single-file mode.
"""

from pathlib import Path
from typing import Any, cast, no_type_check

import chess
import chess.pgn
import pandas as pd

from chipiron.scripts.generate_datasets.generate_boards import DEFAULT_RANDOM_SEED
from chipiron.scripts.generate_datasets.monthly_pgn_pipeline import (
    add_monthly_cli_args,
    build_monthly_pipeline_config,
    build_monthly_run_kwargs,
    finalize_monthly_dataset_from_results,
    run_monthly_pipeline,
)
from chipiron.utils.logger import chipiron_logger
from chipiron.utils.path_variables import EXTERNAL_DATA_DIR


def is_game_over_position(board: chess.Board) -> bool:
    """Check if a board position represents a game-ending situation.

    Only considers checkmate, stalemate, and insufficient material.
    Since we only check final positions, this will only be called on game endings.

    Args:
        board: The chess board to check

    Returns:
        True if the position represents a game-ending situation

    """
    # Check for checkmate
    if board.is_checkmate():
        return True

    # Check for stalemate
    if board.is_stalemate():
        return True

    # Check for insufficient material draws
    return bool(board.is_insufficient_material())


def process_single_game_for_over_positions(
    game: chess.pgn.Game,
) -> dict[str, Any] | None:
    """Process a single game and extract the final game-ending board position if it represents a game-ending situation.

    This function follows the same pattern as process_single_game_for_boards in generate_boards.py
    but focuses only on game-ending positions.

    Args:
        game: The chess game to process

    Returns:
        Dictionary with board data if it's a game-ending position, None otherwise

    """
    try:
        # Access headers from the game object
        game_result = game.headers.get("Result", None)

        # Get the final position directly by following the mainline to the end
        final_node = game.end()
        chess_board = final_node.board()

        # Count total moves for statistics
        moves_list = list(game.mainline_moves())
        moves_processed = len(moves_list)

        # Only check the final position for game-ending situations
        if is_game_over_position(chess_board):
            # Determine the specific reason for game ending
            reason = "unknown"
            if chess_board.is_checkmate():
                reason = "checkmate"
            elif chess_board.is_stalemate():
                reason = "stalemate"
            elif chess_board.is_insufficient_material():
                reason = "insufficient_material"

            return {
                "fen": chess_board.fen(),
                "game_over_reason": reason,
                "result": game_result,
                "is_checkmate": chess_board.is_checkmate(),
                "is_stalemate": chess_board.is_stalemate(),
                "is_insufficient_material": chess_board.is_insufficient_material(),
                "is_final_position": True,  # Always true since we only check final positions
                "move_number": moves_processed,
            }
    except ValueError as e:
        chipiron_logger.warning("Error processing game: %s", e)
        return None
    return None


@no_type_check
def generate_over_boards_dataset_multi_months(
    output_file_path: str,
    max_boards: int = 1_000_000,
    seed: int | None = DEFAULT_RANDOM_SEED,
    start_month: str = "2015-03",
    max_months: int | None = None,
    delete_pgn_after_use: bool = True,
    intermediate_every_games: int = 10_000,
    dest_dir: Path | None = None,
) -> None:
    """Generate over boards dataset streaming through monthly Lichess dumps downloaded on-the-fly.

    Stops when max_boards collected or month limit reached. Each month PGN is deleted when done (optional).

    Args:
        output_file_path: Path for output dataset file
        max_boards: Maximum number of boards to collect
        seed: Random seed for reproducibility
        start_month: Starting month in YYYY-MM format
        max_months: Maximum number of months to process
        delete_pgn_after_use: Whether to delete PGN files after processing
        intermediate_every_games: Save progress every N games
        dest_dir: Directory for temporary PGN files

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

    the_dic: list[dict[str, Any]] = []

    def process_over_board_game(
        game: chess.pgn.Game,
        total_count_move: int,
        the_dic: list[dict[str, Any]],
        _sampling_frequency: int,
        _offset_min: int,
    ) -> int:
        # Count moves for statistics
        moves_list = list(game.mainline_moves())
        total_count_move += len(moves_list)

        # Process game and add to dataset if it's a game-ending position
        board_data = process_single_game_for_over_positions(game)
        if board_data is not None:
            the_dic.append(board_data)

        return total_count_move

    results = run_monthly_pipeline(
        cfg=cfg,
        the_dic=the_dic,
        read_game=chess.pgn.read_game,
        process_game_fn=process_over_board_game,
        sampling_frequency=0,
        offset_min=0,
    )

    def apply_over_boards_metadata(df: pd.DataFrame) -> None:
        df.attrs["dataset_type"] = "over_boards"
        df.attrs["filter_criteria"] = (
            "game-ending positions: checkmate, stalemate, insufficient material only"
        )

        if "game_over_reason" in df.columns:
            value_counts_series = cast(
                "pd.Series", df["game_over_reason"].value_counts()
            )
            reason_counts: dict[str, int] = {}
            for key, value in value_counts_series.items():
                reason_str = str(cast("object", key))
                count_int = int(cast("int", value))
                reason_counts[reason_str] = count_int
            df.attrs["game_over_reason_distribution"] = reason_counts

            chipiron_logger.info("Game-ending reason distribution:")
            for reason_str, count_int in reason_counts.items():
                chipiron_logger.info(
                    "  %s: %s (%s)",
                    reason_str,
                    f"{count_int:,}",
                    f"{count_int / len(df) * 100:.1f}%",
                )

    finalize_monthly_dataset_from_results(
        cfg=cfg,
        the_dic=the_dic,
        results=results,
        sampling_frequency=0,
        offset_min=0,
        postprocess=apply_over_boards_metadata,
    )


# --- CLI integration ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate chess over boards dataset via on-the-fly monthly Lichess downloads"
    )
    add_monthly_cli_args(
        parser,
        default_output=str(EXTERNAL_DATA_DIR / "over_boards_2025"),
        default_max_boards=1_000_000,
        default_seed=DEFAULT_RANDOM_SEED,
    )
    args = parser.parse_args()

    chipiron_logger.info("Running dynamic monthly download mode for over boards")
    generate_over_boards_dataset_multi_months(**build_monthly_run_kwargs(args))
