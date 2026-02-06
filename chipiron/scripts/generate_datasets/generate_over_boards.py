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

import random
from pathlib import Path
from typing import Any, no_type_check

import chess
import chess.pgn
import pandas as pd

from chipiron.scripts.generate_datasets.generate_boards import (
    DEFAULT_RANDOM_SEED,
    ensure_month_pgn,
    iterate_months,
    save_dataset_progress,
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

        return None

    except ValueError as e:
        chipiron_logger.warning("Error processing game: %s", e)
        return None


def process_month_for_over_boards(
    month: str,
    dest_dir: Path,
    max_boards: int,
    current_boards: int,
    the_dic: list[dict[str, Any]],
    count_game: int,
    total_count_move: int,
    intermediate_every_games: int,
    output_file_path: str,
    months_used: list[str],
    seed: int | None,
    delete_pgn_after_use: bool = True,
) -> tuple[int, int, int]:
    """Process a single month's PGN file for over boards, following the same pattern as generate_boards.py.

    Args:
        month: Month to process (YYYY-MM format)
        dest_dir: Directory for PGN files
        max_boards: Maximum boards to collect
        current_boards: Current number of boards collected
        the_dic: List to store board data
        count_game: Current game count
        total_count_move: Current total move count
        intermediate_every_games: Save interval
        output_file_path: Output file path
        months_used: List of months processed
        seed: Random seed
        delete_pgn_after_use: Whether to delete PGN after processing

    Returns:
        Tuple of (updated_boards, updated_games, updated_moves)

    """
    chipiron_logger.info("=== Processing month %s for over boards ===", month)
    pgn_path = ensure_month_pgn(month, dest_dir)
    months_used.append(month)

    with open(pgn_path, encoding="utf-8") as pgn_file:
        while current_boards < max_boards:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            count_game += 1

            # Count moves for statistics
            moves_list = list(game.mainline_moves())
            total_count_move += len(moves_list)

            # Process game and add to dataset if it's a game-ending position
            board_data = process_single_game_for_over_positions(game)
            if board_data is not None:
                the_dic.append(board_data)
                current_boards = len(the_dic)

            # Intermediate save check
            if count_game % intermediate_every_games == 0:
                current_boards = save_dataset_progress(
                    the_dic,
                    output_file_path,
                    count_game,
                    total_count_move,
                    max_boards,
                    None,
                    None,
                    f"months:{','.join(months_used)}",
                    0,  # no sampling frequency for over boards
                    0,  # no offset for over boards
                    seed,
                    is_final=False,
                )

    # Clean up PGN file if requested
    if delete_pgn_after_use:
        chipiron_logger.info("Deleting processed PGN for month %s: %s", month, pgn_path)
        try:
            Path(pgn_path).unlink(missing_ok=True)
        except OSError as exc:
            chipiron_logger.warning("Could not delete %s: %s", pgn_path, exc)

    return current_boards, count_game, total_count_move


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
    if seed is not None:
        random.seed(seed)
    if dest_dir is None:
        dest_dir = EXTERNAL_DATA_DIR / "lichess_pgn"
    dest_dir.mkdir(parents=True, exist_ok=True)

    the_dic: list[dict[str, Any]] = []
    months_used: list[str] = []
    count_game = 0
    total_count_move = 0
    recorded_board = 0

    month_iter = iterate_months(start_month)
    months_processed = 0

    while recorded_board < max_boards:
        month = next(month_iter)
        if max_months is not None and months_processed >= max_months:
            chipiron_logger.info("Reached max_months limit.")
            break

        recorded_board, count_game, total_count_move = process_month_for_over_boards(
            month=month,
            dest_dir=dest_dir,
            max_boards=max_boards,
            current_boards=recorded_board,
            the_dic=the_dic,
            count_game=count_game,
            total_count_move=total_count_move,
            intermediate_every_games=intermediate_every_games,
            output_file_path=output_file_path,
            months_used=months_used,
            seed=seed,
            delete_pgn_after_use=delete_pgn_after_use,
        )
        months_processed += 1

    # Final save using unified function
    if the_dic:
        df = pd.DataFrame(the_dic)

        # Add over-boards specific metadata
        df.attrs["dataset_type"] = "over_boards"
        df.attrs["filter_criteria"] = (
            "game-ending positions: checkmate, stalemate, insufficient material only"
        )

        # Add statistics about game-ending reasons with explicit type casting
        if "game_over_reason" in df.columns:
            # Cast to dict[str, int] to help type checker
            value_counts_series = df["game_over_reason"].value_counts()
            reason_counts = dict(
                zip(
                    [str(k) for k in value_counts_series.index],
                    [int(v) for v in value_counts_series.values.astype(int)],  # pyright: ignore[reportUnknownArgumentType]
                    strict=True,
                )
            )
            df.attrs["game_over_reason_distribution"] = reason_counts

        # Use unified save function for final save
        save_dataset_progress(
            the_dic,
            output_file_path,
            count_game,
            total_count_move,
            max_boards,
            None,
            None,
            f"months:{','.join(months_used)}",
            0,  # no sampling frequency for over boards
            0,  # no offset for over boards
            seed,
            is_final=True,
            months_used=months_used,
        )

        # Print distribution of game-ending reasons with explicit type conversion
        if "game_over_reason" in df.columns:
            chipiron_logger.info("Game-ending reason distribution:")
            value_counts_series = df["game_over_reason"].value_counts()
            for reason_key, count_value in value_counts_series.items():
                reason_str = str(reason_key)  # Explicit string conversion
                count_int = int(count_value)  # Explicit int conversion
                chipiron_logger.info(
                    "  %s: %s (%s)",
                    reason_str,
                    f"{count_int:,}",
                    f"{count_int / len(df) * 100:.1f}%",
                )


# --- CLI integration ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate chess over boards dataset via on-the-fly monthly Lichess downloads"
    )
    parser.add_argument(
        "--start-month", default="2015-03", help="Start month YYYY-MM for dynamic mode"
    )
    parser.add_argument(
        "--max-months", type=int, default=None, help="Maximum number of months to fetch"
    )
    parser.add_argument(
        "--max-boards", type=int, default=1_000_000, help="Maximum boards to collect"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(EXTERNAL_DATA_DIR / "over_boards_2025"),
        help="Output file path",
    )
    parser.add_argument(
        "--keep-pgn",
        action="store_true",
        help="Keep monthly PGN files after processing",
    )
    parser.add_argument(
        "--intermediate-games",
        type=int,
        default=10_000,
        help="Games interval for intermediate saves",
    )
    args = parser.parse_args()

    chipiron_logger.info("Running dynamic monthly download mode for over boards")
    generate_over_boards_dataset_multi_months(
        output_file_path=args.output,
        max_boards=args.max_boards,
        seed=args.seed,
        start_month=args.start_month,
        max_months=args.max_months,
        delete_pgn_after_use=not args.keep_pgn,
        intermediate_every_games=args.intermediate_games,
    )
