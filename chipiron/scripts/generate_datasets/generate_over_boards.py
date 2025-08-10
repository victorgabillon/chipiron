""" Generate chess over boards dataset from Lichess PGN dumps."""

import random
from pathlib import Path
from typing import Any

import chess
import chess.pgn
import pandas as pd

from chipiron.scripts.generate_datasets.generate_boards import (
    DEFAULT_RANDOM_SEED,
    ensure_month_pgn,
    iterate_months,
    save_dataset_progress,
)
from chipiron.utils.path_variables import EXTERNAL_DATA_DIR


def is_game_over_position(board: chess.Board) -> bool:
    """
    Check if a board position represents a game-ending situation.
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
    if board.is_insufficient_material():
        return True

    return False


def process_game_for_over_positions(
    game: chess.pgn.Game,
    total_count_move: int,
    the_dic: list[dict[str, Any]],
) -> int:
    """
    Process a single game and extract only the final game-ending board position.
    Since game-over situations only happen at the end, we only need to check the final position.

    Args:
        game: The chess game to process
        total_count_move: Current total move count across all games
        the_dic: List to append board positions to

    Returns:
        Updated total_count_move after processing this game
    """
    chess_board: chess.Board = game.board()
    # Access headers from the game object
    game_result = game.headers.get("Result", None)

    # Get the final position directly by following the mainline to the end
    final_node = game.end()
    chess_board = final_node.board()

    # Count total moves for statistics
    moves_list = list(game.mainline_moves())
    moves_processed = len(moves_list)
    total_count_move += moves_processed

    # Only check the final position for game-ending situations (removed eval requirement)
    if is_game_over_position(chess_board):
        # Determine the specific reason for game ending
        reason = "unknown"
        if chess_board.is_checkmate():
            reason = "checkmate"
        elif chess_board.is_stalemate():
            reason = "stalemate"
        elif chess_board.is_insufficient_material():
            reason = "insufficient_material"

        the_dic.append(
            {
                "fen": chess_board.fen(),
                "game_over_reason": reason,
                "result": game_result,
                "is_checkmate": chess_board.is_checkmate(),
                "is_stalemate": chess_board.is_stalemate(),
                "is_insufficient_material": chess_board.is_insufficient_material(),
                "is_final_position": True,  # Always true since we only check final positions
                "move_number": moves_processed,
            }
        )

    return total_count_move


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
            print("Reached max_months limit.")
            break
        print(f"\n=== Processing month {month} for over boards ===")
        pgn_path = ensure_month_pgn(month, dest_dir)
        months_used.append(month)
        months_processed += 1

        with open(pgn_path, "r", encoding="utf-8") as pgn_file:
            while recorded_board < max_boards:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                count_game += 1

                if count_game % intermediate_every_games == 0:
                    recorded_board = save_dataset_progress(
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

                total_count_move = process_game_for_over_positions(
                    game, total_count_move, the_dic
                )
                recorded_board = len(the_dic)

        if delete_pgn_after_use:
            print(f"Deleting processed PGN for month {month}: {pgn_path}")
            try:
                Path(pgn_path).unlink(missing_ok=True)
            except OSError as exc:
                print(f"Warning: could not delete {pgn_path}: {exc}")

    # Final save using unified function
    if the_dic:
        df = pd.DataFrame.from_dict(the_dic)

        # Add over-boards specific metadata
        df.attrs["dataset_type"] = "over_boards"
        df.attrs["filter_criteria"] = (
            "game-ending positions: checkmate, stalemate, insufficient material only"
        )

        # Add statistics about game-ending reasons
        if "game_over_reason" in df.columns:
            reason_counts = df["game_over_reason"].value_counts().to_dict()
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

        # Print distribution of game-ending reasons
        if "game_over_reason" in df.columns:
            print("\nGame-ending reason distribution:")
            for reason, count in df["game_over_reason"].value_counts().items():
                print(f"  {reason}: {count:,} ({count / len(df) * 100:.1f}%)")


def generate_over_boards_dataset_legacy(
    input_pgn_file_path: str,
    output_file_path: str,
    max_boards: int = 1_000_000,
    total_games_in_file: int | None = None,
    total_moves_in_file: int | None = None,
    intermediate_every_games: int = 10_000,
) -> None:
    """
    Generate a dataset of game-ending chess board positions from a single PGN file.
    Legacy function for backwards compatibility.
    """
    the_dic: list[dict[str, Any]] = []
    count_game: int = 0
    total_count_move: int = 0
    recorded_board = 0

    with open(input_pgn_file_path, "r", encoding="utf-8") as pgn:
        while recorded_board < max_boards:
            count_game += 1
            game: chess.pgn.Game | None = chess.pgn.read_game(pgn)

            if count_game % intermediate_every_games == 0:
                recorded_board = save_dataset_progress(
                    the_dic,
                    output_file_path,
                    count_game,
                    total_count_move,
                    max_boards,
                    total_games_in_file,
                    total_moves_in_file,
                    input_pgn_file_path,
                    0,  # no sampling frequency for over boards
                    0,  # no offset for over boards
                    None,  # no seed for legacy
                    is_final=False,
                )

            if game is None:
                print("GAME NONE")
                break
            else:
                total_count_move = process_game_for_over_positions(
                    game, total_count_move, the_dic
                )

    # Final save using unified function
    if the_dic:
        df = pd.DataFrame.from_dict(the_dic)

        # Add over-boards specific metadata
        df.attrs["dataset_type"] = "over_boards"
        df.attrs["filter_criteria"] = (
            "game-ending positions: checkmate, stalemate, insufficient material only"
        )

        # Add statistics about game-ending reasons
        if "game_over_reason" in df.columns:
            reason_counts = df["game_over_reason"].value_counts().to_dict()
            df.attrs["game_over_reason_distribution"] = reason_counts

        # Use unified save function for final save
        save_dataset_progress(
            the_dic,
            output_file_path,
            count_game,
            total_count_move,
            max_boards,
            total_games_in_file,
            total_moves_in_file,
            input_pgn_file_path,
            0,  # no sampling frequency for over boards
            0,  # no offset for over boards
            None,  # no seed for legacy
            is_final=True,
        )

        # Print distribution of game-ending reasons
        if "game_over_reason" in df.columns:
            print("\nGame-ending reason distribution:")
            for reason, count in df["game_over_reason"].value_counts().items():
                print(f"  {reason}: {count:,} ({count / len(df) * 100:.1f}%)")
    else:
        print("No over board positions found to save.")


# --- CLI integration (dynamic only) ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate chess over boards dataset via on-the-fly monthly Lichess downloads (dynamic only)"
    )
    parser.add_argument(
        "--start-month", default="2015-03", help="Start month YYYY-MM for dynamic mode"
    )
    parser.add_argument(
        "--max-months", type=int, default=None, help="Maximum number of months to fetch"
    )
    parser.add_argument("--max-boards", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=str, default=str(EXTERNAL_DATA_DIR / "over_boards_2025")
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
    parser.add_argument(
        "--legacy-file",
        type=str,
        help="Use legacy single file mode with specified PGN file path",
    )
    args = parser.parse_args()

    if args.legacy_file:
        print(f"Running legacy single-file mode with: {args.legacy_file}")
        generate_over_boards_dataset_legacy(
            input_pgn_file_path=args.legacy_file,
            output_file_path=args.output,
            max_boards=args.max_boards,
            intermediate_every_games=args.intermediate_games,
        )
    else:
        print("Running dynamic monthly download mode for over boards")
        generate_over_boards_dataset_multi_months(
            output_file_path=args.output,
            max_boards=args.max_boards,
            seed=args.seed,
            start_month=args.start_month,
            max_months=args.max_months,
            delete_pgn_after_use=not args.keep_pgn,
            intermediate_every_games=args.intermediate_games,
        )
