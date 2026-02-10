# pylint: disable=duplicate-code
"""Shared pipeline helpers for scripts that stream monthly Lichess PGN dumps.

This module intentionally holds the mechanics (seed, dest-dir setup, month loop helpers,
intermediate saving, and cleanup) so dataset scripts can focus on *what to extract* from games.
"""

from __future__ import annotations

import random
import shutil
import subprocess
import urllib.error
import urllib.request
from collections.abc import Callable, Generator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import zstandard
from pandas import DataFrame

from chipiron.utils.logger import chipiron_logger
from chipiron.utils.path_variables import EXTERNAL_DATA_DIR

if TYPE_CHECKING:
    from atomheart.board.utils import Fen

# Lichess monthly database parameters
LICHESS_STANDARD_BASE_URL = "https://database.lichess.org/standard"
MONTHLY_FILE_TEMPLATE = "lichess_db_standard_rated_{month}.pgn.zst"  # month = YYYY-MM


class DatasetDecompressionError(RuntimeError):
    """Raised when PGN decompression fails."""

    def __init__(self, original_error: Exception) -> None:
        """Initialize the error with the original exception."""
        super().__init__(f"Both zstd CLI and Python zstandard failed: {original_error}")


def save_dataset_progress(
    the_dic: list[dict[str, Fen]],
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
    extra_attrs: dict[str, Any] | None = None,
    postprocess: Callable[[DataFrame], None] | None = None,
) -> int:
    """Save dataset progress (intermediate or final) and display statistics."""
    new_data_frame_states: DataFrame = pd.DataFrame(the_dic)
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

    if extra_attrs:
        new_data_frame_states.attrs.update(extra_attrs)

    # Additional metadata for final saves
    if is_final:
        if months_used:
            new_data_frame_states.attrs["months_used"] = months_used
            new_data_frame_states.attrs["source_pgn_mode"] = "dynamic_monthly"
        new_data_frame_states.attrs["total_games_processed"] = count_game
        new_data_frame_states.attrs["total_moves_processed"] = total_count_move
        new_data_frame_states.attrs["final_dataset_size"] = len(new_data_frame_states)

    if postprocess:
        postprocess(new_data_frame_states)

    new_data_frame_states.to_pickle(output_file_path)

    if is_final and months_used:
        chipiron_logger.info(
            "Final dataset saved (%d positions) using months %s -> %s",
            len(new_data_frame_states),
            months_used,
            output_file_path,
        )

    return recorded_board


def iterate_months(start_month: str) -> Generator[str]:
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

    progress_thresholds: set[int] = set()

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
        except subprocess.CalledProcessError as e:
            chipiron_logger.info(
                "zstd command failed (exit code %d): %s", e.returncode, e.stderr
            )
            chipiron_logger.info("Falling back to Python decompression...")
        else:
            return

    # Fall back to Python zstandard
    try:
        chipiron_logger.info("Using Python zstandard library for decompression...")
        dctx = zstandard.ZstdDecompressor()
        with zst_path.open("rb") as src, output_pgn_path.open("wb") as dst:
            dctx.copy_stream(src, dst)
        chipiron_logger.info("Decompression completed with Python zstandard")
    except Exception as exc:
        raise DatasetDecompressionError(exc) from exc


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


@dataclass(frozen=True, slots=True)
class MonthlyPipelineConfig:
    """Configuration shared by monthly PGN streaming scripts."""

    output_file_path: str
    max_boards: int
    seed: int | None
    start_month: str
    max_months: int | None
    delete_pgn_after_use: bool
    intermediate_every_games: int
    dest_dir: Path


def init_monthly_pipeline_config(
    *,
    output_file_path: str,
    max_boards: int,
    seed: int | None,
    start_month: str,
    max_months: int | None,
    delete_pgn_after_use: bool,
    intermediate_every_games: int,
    dest_dir: Path | None,
) -> MonthlyPipelineConfig:
    """Normalize monthly pipeline settings and prepare destination directory."""
    if seed is not None:
        random.seed(seed)

    if dest_dir is None:
        dest_dir = EXTERNAL_DATA_DIR / "lichess_pgn"

    dest_dir.mkdir(parents=True, exist_ok=True)

    return MonthlyPipelineConfig(
        output_file_path=output_file_path,
        max_boards=max_boards,
        seed=seed,
        start_month=start_month,
        max_months=max_months,
        delete_pgn_after_use=delete_pgn_after_use,
        intermediate_every_games=intermediate_every_games,
        dest_dir=dest_dir,
    )


def build_monthly_pipeline_config(
    *,
    output_file_path: str,
    max_boards: int,
    seed: int | None,
    start_month: str,
    max_months: int | None,
    delete_pgn_after_use: bool,
    intermediate_every_games: int,
    dest_dir: Path | None,
) -> MonthlyPipelineConfig:
    """Build a monthly pipeline config from explicit arguments."""
    return init_monthly_pipeline_config(
        output_file_path=output_file_path,
        max_boards=max_boards,
        seed=seed,
        start_month=start_month,
        max_months=max_months,
        delete_pgn_after_use=delete_pgn_after_use,
        intermediate_every_games=intermediate_every_games,
        dest_dir=dest_dir,
    )


def monthly_iter(
    *, start_month: str, max_months: int | None
) -> Generator[tuple[str, int]]:
    """Yield (month, month_index) starting at start_month, stopping when max_months reached."""
    it = iterate_months(start_month)
    for i, month in enumerate(it):
        if max_months is not None and i >= max_months:
            break
        yield month, i


def open_month_pgn(*, month: str, dest_dir: Path) -> Path:
    """Ensure PGN for month exists and return the path (download+decompress if needed)."""
    chipiron_logger.info("=== Processing month %s ===", month)
    return ensure_month_pgn(month, dest_dir)


def maybe_save_progress(
    *,
    the_dic: list[dict[str, Any]],
    output_file_path: str,
    count_game: int,
    total_count_move: int,
    max_boards: int,
    months_used: list[str],
    sampling_frequency: int,
    offset_min: int,
    seed: int | None,
) -> int:
    """Intermediate save wrapper used by scripts."""
    return save_dataset_progress(
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


def final_save(
    *,
    the_dic: list[dict[str, Any]],
    output_file_path: str,
    count_game: int,
    total_count_move: int,
    max_boards: int,
    months_used: list[str],
    sampling_frequency: int,
    offset_min: int,
    seed: int | None,
    extra_attrs: dict[str, Any] | None = None,
    postprocess: Callable[[DataFrame], None] | None = None,
) -> None:
    """Save final dataset and log summary statistics, with support for extra metadata and postprocessing."""
    if not the_dic:
        return

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
        extra_attrs=extra_attrs,
        postprocess=postprocess,
    )


def maybe_delete_month_pgn(
    *, pgn_path: Path, month: str, delete_pgn_after_use: bool
) -> None:
    """Cleanup wrapper used by scripts."""
    if not delete_pgn_after_use:
        return
    chipiron_logger.info("Deleting processed PGN for month %s: %s", month, pgn_path)
    try:
        pgn_path.unlink(missing_ok=True)
    except OSError as exc:
        chipiron_logger.warning("Could not delete %s: %s", pgn_path, exc)


ProcessGameFn = Callable[[Any, int, list[dict[str, Any]], int, int], int]
ReadGameFn = Callable[[Any], Any]


def run_monthly_pipeline(
    *,
    cfg: MonthlyPipelineConfig,
    the_dic: list[dict[str, Any]],
    read_game: ReadGameFn,
    process_game_fn: ProcessGameFn,
    sampling_frequency: int,
    offset_min: int,
) -> tuple[list[str], int, int]:
    """Run the shared monthly PGN processing loop and return progress totals."""
    months_used: list[str] = []
    count_game = 0
    total_count_move = 0
    recorded_board = len(the_dic)

    for month, _ in monthly_iter(
        start_month=cfg.start_month, max_months=cfg.max_months
    ):
        if recorded_board >= cfg.max_boards:
            break

        pgn_path = open_month_pgn(month=month, dest_dir=cfg.dest_dir)
        months_used.append(month)

        with open(pgn_path, encoding="utf-8") as pgn_file:
            while recorded_board < cfg.max_boards:
                game = read_game(pgn_file)
                if game is None:
                    break

                count_game += 1
                if count_game % cfg.intermediate_every_games == 0:
                    recorded_board = maybe_save_progress(
                        the_dic=the_dic,
                        output_file_path=cfg.output_file_path,
                        count_game=count_game,
                        total_count_move=total_count_move,
                        max_boards=cfg.max_boards,
                        months_used=months_used,
                        sampling_frequency=sampling_frequency,
                        offset_min=offset_min,
                        seed=cfg.seed,
                    )

                total_count_move = process_game_fn(
                    game,
                    total_count_move,
                    the_dic,
                    sampling_frequency,
                    offset_min,
                )
                recorded_board = len(the_dic)

        maybe_delete_month_pgn(
            pgn_path=Path(pgn_path),
            month=month,
            delete_pgn_after_use=cfg.delete_pgn_after_use,
        )

    return months_used, count_game, total_count_move


def add_monthly_cli_args(
    parser: Any,
    *,
    default_output: str,
    default_max_boards: int,
    default_seed: int | None,
) -> None:
    """Add shared CLI arguments for monthly PGN scripts."""
    parser.add_argument(
        "--start-month", default="2015-03", help="Start month YYYY-MM for dynamic mode"
    )
    parser.add_argument(
        "--max-months", type=int, default=None, help="Maximum number of months to fetch"
    )
    parser.add_argument("--max-boards", type=int, default=default_max_boards)
    parser.add_argument("--seed", type=int, default=default_seed)
    parser.add_argument("--output", type=str, default=default_output)
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


def build_monthly_run_kwargs(args: Any) -> dict[str, Any]:
    """Build common runtime kwargs from CLI args."""
    return {
        "output_file_path": args.output,
        "max_boards": args.max_boards,
        "seed": args.seed,
        "start_month": args.start_month,
        "max_months": args.max_months,
        "delete_pgn_after_use": not args.keep_pgn,
        "intermediate_every_games": args.intermediate_games,
    }


def finalize_monthly_dataset(
    *,
    cfg: MonthlyPipelineConfig,
    the_dic: list[dict[str, Any]],
    months_used: list[str],
    count_game: int,
    total_count_move: int,
    sampling_frequency: int,
    offset_min: int,
    extra_attrs: dict[str, Any] | None = None,
    postprocess: Callable[[DataFrame], None] | None = None,
) -> None:
    """Finalize a dataset using config-derived metadata."""
    final_save(
        the_dic=the_dic,
        output_file_path=cfg.output_file_path,
        count_game=count_game,
        total_count_move=total_count_move,
        max_boards=cfg.max_boards,
        months_used=months_used,
        sampling_frequency=sampling_frequency,
        offset_min=offset_min,
        seed=cfg.seed,
        extra_attrs=extra_attrs,
        postprocess=postprocess,
    )


def finalize_monthly_dataset_simple(
    *,
    cfg: MonthlyPipelineConfig,
    the_dic: list[dict[str, Any]],
    months_used: list[str],
    count_game: int,
    total_count_move: int,
    sampling_frequency: int,
    offset_min: int,
    postprocess: Callable[[DataFrame], None] | None = None,
    extra_attrs: dict[str, Any] | None = None,
) -> None:
    """Finalize dataset with explicit month/game/move counts, for cases where totals are computed separately."""
    finalize_monthly_dataset(
        cfg=cfg,
        the_dic=the_dic,
        months_used=months_used,
        count_game=count_game,
        total_count_move=total_count_move,
        sampling_frequency=sampling_frequency,
        offset_min=offset_min,
        postprocess=postprocess,
        extra_attrs=extra_attrs,
    )


def finalize_monthly_dataset_from_results(
    *,
    cfg: MonthlyPipelineConfig,
    the_dic: list[dict[str, Any]],
    results: tuple[list[str], int, int],
    sampling_frequency: int,
    offset_min: int,
    postprocess: Callable[[DataFrame], None] | None = None,
    extra_attrs: dict[str, Any] | None = None,
) -> None:
    """Finalize dataset using (months_used, count_game, total_count_move) tuple."""
    months_used, count_game, total_count_move = results
    finalize_monthly_dataset_simple(
        cfg=cfg,
        the_dic=the_dic,
        months_used=months_used,
        count_game=count_game,
        total_count_move=total_count_move,
        sampling_frequency=sampling_frequency,
        offset_min=offset_min,
        postprocess=postprocess,
        extra_attrs=extra_attrs,
    )
