"""
This script generates datasets of chess board positions by sampling games from the Lichess monthly PGN database.
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
import shutil
import subprocess
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Generator

import chess
import chess.pgn
import pandas as pd
import zstandard
from pandas import DataFrame

from chipiron.environments.chess_env.board.utils import fen
from chipiron.utils.logger import chipiron_logger
from chipiron.utils.path_variables import (
    EXTERNAL_DATA_DIR,  # removed LICHESS_PGN_FILE usage
)

# Sampling configuration variables
DEFAULT_SAMPLING_FREQUENCY = 50
DEFAULT_OFFSET_MIN = 5
DEFAULT_RANDOM_SEED: int | None = 0  # Set to an int for reproducibility

# Lichess monthly database parameters
LICHESS_STANDARD_BASE_URL = "https://database.lichess.org/standard"
MONTHLY_FILE_TEMPLATE = "lichess_db_standard_rated_{month}.pgn.zst"  # month = YYYY-MM


def save_dataset_progress(
    the_dic: list[dict[str, fen]],
    output_file_path: str,
    count_game: int,
    total_count_move: int,
    max_boards: int,
    total_games_in_file: int | None,
    total_moves_in_file: int | None,
    input_pgn_file_path: str,
    sampling_frequency: int,
    offset_min: int,
    seed: int | None,
    is_final: bool = False,
    months_used: list[str] | None = None,
) -> int:
    """
    Save dataset progress (intermediate or final) and display statistics.

    Args:
        the_dic: Current list of board positions
        output_file_path: Path where to save the pickle file
        count_game: Number of games processed so far
        total_count_move: Number of moves processed so far
        max_boards: Maximum target board positions
        total_games_in_file: Total games in source file (optional)
        total_moves_in_file: Total moves in source file (optional)
        input_pgn_file_path: Source PGN file path
        sampling_frequency: Sampling frequency for moves
        offset_min: Minimum offset for sampling
        seed: Random seed used
        is_final: Whether this is the final save (adds additional metadata)
        months_used: List of months processed (for dynamic mode)

    Returns:
        Number of board positions recorded so far
    """
    new_data_frame_states: DataFrame = pd.DataFrame.from_dict(the_dic)
    recorded_board = len(new_data_frame_states.index)

    save_type = "Final" if is_final else "Progress"
    chipiron_logger.info(
        "%s: %s / %s board positions collected (%.1f%%)",
        save_type,
        f"{recorded_board:,}",
        f"{max_boards:,}",
        recorded_board / max_boards * 100,
    )

    # Enhanced progress with file totals
    games_progress = f"{count_game:,}"
    moves_progress = f"{total_count_move:,}"
    if total_games_in_file:
        games_progress += f" / {total_games_in_file:,} ({count_game / total_games_in_file * 100:.1f}%)"
    if total_moves_in_file:
        moves_progress += f" / {total_moves_in_file:,} ({total_count_move / total_moves_in_file * 100:.1f}%)"

    chipiron_logger.info("         Games processed: %s", games_progress)
    chipiron_logger.info("         Moves processed: %s", moves_progress)

    # Add metadata
    new_data_frame_states.attrs["source_pgn_file"] = input_pgn_file_path
    new_data_frame_states.attrs["sampling_frequency"] = sampling_frequency
    new_data_frame_states.attrs["offset_min"] = offset_min
    new_data_frame_states.attrs["offset_max_strategy"] = "per-game total move count"
    new_data_frame_states.attrs["random_seed"] = seed
    new_data_frame_states.attrs["creation_date"] = pd.Timestamp.now().isoformat()
    new_data_frame_states.attrs["filter_criteria"] = (
        "positions sampled at offset + k*frequency (no engine eval filter)"
    )
    new_data_frame_states.attrs["games_processed"] = count_game
    new_data_frame_states.attrs["moves_processed"] = total_count_move

    # Additional metadata for final saves
    if is_final:
        if months_used:
            new_data_frame_states.attrs["months_used"] = months_used
            new_data_frame_states.attrs["source_pgn_mode"] = "dynamic_monthly"
        new_data_frame_states.attrs["total_games_processed"] = count_game
        new_data_frame_states.attrs["total_moves_processed"] = total_count_move
        new_data_frame_states.attrs["final_dataset_size"] = len(new_data_frame_states)

    new_data_frame_states.to_pickle(output_file_path)

    if is_final and months_used:
        chipiron_logger.info(
            "Final dataset saved (%d positions) using months %s -> %s",
            len(new_data_frame_states),
            months_used,
            output_file_path,
        )

    return recorded_board


def process_game(
    game: chess.pgn.GameNode,
    total_count_move: int,
    the_dic: list[dict[str, fen]],
    sampling_frequency: int,
    offset_min: int = DEFAULT_OFFSET_MIN,
) -> int:
    """Process a single game and extract positions (no eval requirement)."""
    chess_board: chess.Board = game.board()
    current_node: chess.pgn.GameNode = game

    # Normalize offset_min
    if offset_min < 0:
        offset_min = 0

    # Materialize moves to know game length
    moves_list = list(game.mainline_moves())
    game_total_moves = len(moves_list)
    effective_min = min(offset_min, game_total_moves)

    # Draw random offset within game length
    if game_total_moves >= effective_min:
        random_offset = random.randint(effective_min, game_total_moves)
    else:
        random_offset = game_total_moves  # degenerate case

    # Track moves within this game
    game_move_count = 0

    for move in moves_list:
        total_count_move += 1
        game_move_count += 1
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


def iterate_months(start_month: str) -> Generator[str, None, None]:
    """Yield month strings (YYYY-MM) starting at start_month incrementing by one month indefinitely."""
    dt = datetime.strptime(start_month, "%Y-%m")
    while True:
        yield dt.strftime("%Y-%m")
        # increment month
        year = dt.year + (dt.month // 12)
        month = dt.month % 12 + 1
        dt = dt.replace(year=year, month=month)


def download_month_zst(month: str, dest_dir: Path) -> Path:
    """Download the compressed monthly PGN (.zst) file for a given month into dest_dir and return its path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    file_name = MONTHLY_FILE_TEMPLATE.format(month=month)
    url = f"{LICHESS_STANDARD_BASE_URL}/{file_name}"
    local_path = dest_dir / file_name

    # Check if file exists and verify its size
    if local_path.exists():
        try:
            # Get remote file size to compare

            with urllib.request.urlopen(url) as response:
                remote_size = int(response.headers.get("Content-Length", 0))

            local_size = local_path.stat().st_size

            if remote_size > 0 and local_size == remote_size:
                chipiron_logger.info(
                    "Compressed file already exists for %s: %s", month, local_path
                )
                return local_path
            else:
                chipiron_logger.info(
                    "Incomplete file detected for %s (local: %d, remote: %d). Re-downloading...",
                    month,
                    local_size,
                    remote_size,
                )
        except urllib.error.URLError as e:
            chipiron_logger.info(
                "Could not verify file size for %s: %s. Re-downloading...", month, e
            )

    chipiron_logger.info("Downloading %s -> %s", url, local_path)

    progress_thresholds = set()

    def progress_hook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            downloaded = block_num * block_size
            percent = min(100.0, downloaded * 100.0 / total_size)
            downloaded_mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            # Only log when passing a new 10% threshold
            threshold = int(percent // 10) * 10
            if threshold not in progress_thresholds and threshold > 0:
                progress_thresholds.add(threshold)
                chipiron_logger.info(
                    "Progress: %d%% (%.1f/%.1f MB)", threshold, downloaded_mb, total_mb
                )

    urllib.request.urlretrieve(url, local_path, reporthook=progress_hook)
    chipiron_logger.info("")  # New line after download completes
    return local_path


def decompress_zst(zst_path: Path, output_pgn_path: Path) -> None:
    """Decompress .zst to .pgn using zstd command line or Python fallback."""
    if output_pgn_path.exists():
        chipiron_logger.info("Decompressed PGN already present: %s", output_pgn_path)
        return

    chipiron_logger.info("Decompressing %s...", zst_path.name)

    # Try zstd command first
    if shutil.which("zstd"):
        try:
            cmd = ["zstd", "-d", "-f", "-o", str(output_pgn_path), str(zst_path)]
            chipiron_logger.info("Running: %s", " ".join(cmd))
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            chipiron_logger.info("Decompression completed with zstd command")
            return
        except subprocess.CalledProcessError as e:
            chipiron_logger.info(
                "zstd command failed (exit code %d): %s", e.returncode, e.stderr
            )
            chipiron_logger.info("Falling back to Python decompression...")

    # Fall back to Python zstandard
    try:
        chipiron_logger.info("Using Python zstandard library for decompression...")
        dctx = zstandard.ZstdDecompressor()
        with zst_path.open("rb") as src, output_pgn_path.open("wb") as dst:
            dctx.copy_stream(src, dst)
        chipiron_logger.info("Decompression completed with Python zstandard")
    except Exception as exc:
        raise RuntimeError(f"Both zstd CLI and Python zstandard failed: {exc}") from exc


def ensure_month_pgn(month: str, dest_dir: Path) -> Path:
    """Ensure decompressed monthly PGN exists; download & decompress if needed; return .pgn path."""
    file_name = MONTHLY_FILE_TEMPLATE.format(month=month)
    pgn_path = dest_dir / file_name.replace(".pgn.zst", ".pgn")

    # First check if decompressed file already exists
    if pgn_path.exists():
        chipiron_logger.info(
            "Decompressed PGN already exists for %s: %s", month, pgn_path
        )
        return pgn_path

    # If not, download compressed file and decompress
    compressed = download_month_zst(month, dest_dir)
    decompress_zst(compressed, pgn_path)

    # Always delete compressed file after decompression
    if compressed.exists():
        compressed.unlink(missing_ok=True)
    chipiron_logger.info("Deleted compressed file: %s", compressed)

    return pgn_path


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
    if seed is not None:
        random.seed(seed)
    if dest_dir is None:
        dest_dir = EXTERNAL_DATA_DIR / "lichess_pgn"
    dest_dir.mkdir(parents=True, exist_ok=True)

    the_dic: list[dict[str, fen]] = []
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
        chipiron_logger.info("=== Processing month %s ===", month)
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
                        sampling_frequency,
                        offset_min,
                        seed,
                        is_final=False,
                    )
                total_count_move = process_game(
                    game, total_count_move, the_dic, sampling_frequency, offset_min
                )
                recorded_board = len(the_dic)
        if delete_pgn_after_use:
            chipiron_logger.info(
                "Deleting processed PGN for month %s: %s", month, pgn_path
            )
            try:
                Path(pgn_path).unlink(missing_ok=True)
            except OSError as exc:
                chipiron_logger.info("Warning: could not delete %s: %s", pgn_path, exc)
        # continue to next month if need more boards

    # Final save using unified function
    if the_dic:
        save_dataset_progress(
            the_dic,
            output_file_path,
            count_game,
            total_count_move,
            max_boards,
            None,
            None,
            f"months:{','.join(months_used)}",
            sampling_frequency,
            offset_min,
            seed,
            is_final=True,
            months_used=months_used,
        )


# --- CLI integration (dynamic only) ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate chess board dataset via on-the-fly monthly Lichess downloads (dynamic only)"
    )
    parser.add_argument(
        "--start-month", default="2015-03", help="Start month YYYY-MM for dynamic mode"
    )
    parser.add_argument(
        "--max-months", type=int, default=None, help="Maximum number of months to fetch"
    )
    parser.add_argument("--max-boards", type=int, default=10_000_000)
    parser.add_argument(
        "--sampling-frequency", type=int, default=DEFAULT_SAMPLING_FREQUENCY
    )
    parser.add_argument("--offset-min", type=int, default=DEFAULT_OFFSET_MIN)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=str,
        default=str(EXTERNAL_DATA_DIR / "datasets" / "only_boards.pkl"),
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

    chipiron_logger.info(
        "Running dynamic monthly download mode (legacy single-file mode disabled)"
    )
    generate_board_dataset_multi_months(
        output_file_path=args.output,
        max_boards=args.max_boards,
        sampling_frequency=args.sampling_frequency,
        offset_min=args.offset_min,
        seed=args.seed,
        start_month=args.start_month,
        max_months=args.max_months,
        delete_pgn_after_use=not args.keep_pgn,
        intermediate_every_games=args.intermediate_games,
    )
