import sys

import chess
import chess.pgn
import pandas as pd
from pandas import DataFrame

from chipiron.environments.chess_env.board.utils import fen
from chipiron.utils.path_variables import EXTERNAL_DATA_DIR, LICHESS_PGN_FILE

# Path to the Lichess PGN file - can be downloaded with 'make lichess-pgn'
pgn_file_path = LICHESS_PGN_FILE

if not pgn_file_path.exists():
    print(f"PGN file not found at: {pgn_file_path}")
    print("Please run 'make lichess-pgn' to download the Lichess database first.")
    sys.exit(1)


def scan_pgn_file(pgn_path: str) -> tuple[int, int]:
    """
    Scan the PGN file to count total games and moves.

    Args:
        pgn_path: Path to the PGN file

    Returns:
        Tuple of (total_games, total_moves)
    """
    print("Scanning PGN file to count total games and moves...")
    total_games = 0
    total_moves = 0

    with open(pgn_path, "r", encoding="utf-8") as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break

            total_games += 1
            if total_games % 10_000 == 0:
                print(f"Scanned {total_games:,} games so far...")

            # Count moves in this game
            game_moves = sum(1 for _ in game.mainline_moves())
            total_moves += game_moves

    print(f"Scan complete: {total_games:,} total games, {total_moves:,} total moves")
    return total_games, total_moves


def save_final_dataset(
    the_dic: list[dict[str, fen]],
    output_file_path: str,
    input_pgn_file_path: str,
    sampling_frequency: int,
    count_game: int,
    total_count_move: int,
    max_boards: int,
    total_games_in_file: int | None,
    total_moves_in_file: int | None,
) -> None:
    """
    Save the final dataset with comprehensive metadata.

    Args:
        the_dic: List of board positions to save
        output_file_path: Path where to save the final pickle file
        input_pgn_file_path: Source PGN file path
        sampling_frequency: Sampling frequency used
        count_game: Total games processed
        total_count_move: Total moves processed
        max_boards: Maximum target board positions
        total_games_in_file: Total games in source file (optional)
        total_moves_in_file: Total moves in source file (optional)
    """
    if the_dic:
        final_data_frame: DataFrame = pd.DataFrame.from_dict(the_dic)

        # Add comprehensive metadata
        final_data_frame.attrs["source_pgn_file"] = input_pgn_file_path
        final_data_frame.attrs["sampling_frequency"] = sampling_frequency
        final_data_frame.attrs["creation_date"] = pd.Timestamp.now().isoformat()
        final_data_frame.attrs["filter_criteria"] = (
            "positions with engine evaluations only (no other filtering)"
        )
        final_data_frame.attrs["total_games_processed"] = count_game
        final_data_frame.attrs["total_moves_processed"] = total_count_move
        final_data_frame.attrs["max_boards_target"] = max_boards
        final_data_frame.attrs["final_dataset_size"] = len(final_data_frame)
        if total_games_in_file:
            final_data_frame.attrs["total_games_in_source_file"] = total_games_in_file
        if total_moves_in_file:
            final_data_frame.attrs["total_moves_in_source_file"] = total_moves_in_file

        final_data_frame.to_pickle(output_file_path)
        print(
            f"Final dataset saved with {len(final_data_frame)} positions to {output_file_path}"
        )
        print(
            f"Metadata: source file, sampling frequency ({sampling_frequency}), creation date, filter criteria"
        )


def save_intermediate_progress(
    the_dic: list[dict[str, fen]],
    output_file_path: str,
    count_game: int,
    total_count_move: int,
    max_boards: int,
    total_games_in_file: int | None,
    total_moves_in_file: int | None,
    input_pgn_file_path: str,
    sampling_frequency: int,
) -> int:
    """
    Save intermediate progress and display statistics.

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

    Returns:
        Number of board positions recorded so far
    """
    new_data_frame_states: DataFrame = pd.DataFrame.from_dict(the_dic)
    recorded_board = len(new_data_frame_states.index)
    print(
        f"Progress: {recorded_board:,} / {max_boards:,} board positions collected ({recorded_board / max_boards * 100:.1f}%)"
    )

    # Enhanced progress with file totals
    games_progress = f"{count_game:,}"
    moves_progress = f"{total_count_move:,}"
    if total_games_in_file:
        games_progress += f" / {total_games_in_file:,} ({count_game / total_games_in_file * 100:.1f}%)"
    if total_moves_in_file:
        moves_progress += f" / {total_moves_in_file:,} ({total_count_move / total_moves_in_file * 100:.1f}%)"

    print(f"         Games processed: {games_progress}")
    print(f"         Moves processed: {moves_progress}")

    # Add metadata to intermediate saves
    new_data_frame_states.attrs["source_pgn_file"] = input_pgn_file_path
    new_data_frame_states.attrs["sampling_frequency"] = sampling_frequency
    new_data_frame_states.attrs["creation_date"] = pd.Timestamp.now().isoformat()
    new_data_frame_states.attrs["filter_criteria"] = (
        "positions with engine evaluations only (no other filtering)"
    )
    new_data_frame_states.attrs["games_processed"] = count_game
    new_data_frame_states.attrs["moves_processed"] = total_count_move

    new_data_frame_states.to_pickle(output_file_path)

    return recorded_board


def process_game(
    game: chess.pgn.GameNode,
    total_count_move: int,
    the_dic: list[dict[str, fen]],
    sampling_frequency: int,
) -> int:
    """
    Process a single game and extract board positions with evaluations.

    Args:
        game: The chess game to process
        total_count_move: Current total move count across all games
        the_dic: List to append board positions to
        sampling_frequency: Sample every N moves

    Returns:
        Updated total_count_move after processing this game
    """
    chess_board: chess.Board = game.board()
    current_node: chess.pgn.GameNode = game

    for move in game.mainline_moves():
        total_count_move += 1
        chess_board.push(move)

        next_node: chess.pgn.GameNode | None = current_node.next()
        if next_node is None:
            break  # Safety check for end of game

        current_node = next_node
        if (
            current_node.eval() is not None
            and total_count_move % sampling_frequency == 0
        ):
            the_dic.append({"fen": chess_board.fen()})

    return total_count_move


def generate_board_dataset(
    input_pgn_file_path: str,
    output_file_path: str,
    max_boards: int = 10_000_000,
    total_games_in_file: int | None = None,
    total_moves_in_file: int | None = None,
    sampling_frequency: int = 50,
) -> None:
    """
    Generate a dataset of chess board positions from a PGN file.

    Args:
        input_pgn_file_path: Path to the PGN file containing chess games
        output_file_path: Path where the output pickle file will be saved
        max_boards: Maximum number of board positions to collect
        total_games_in_file: Total games in the PGN file (for progress display)
        total_moves_in_file: Total moves in the PGN file (for progress display)
        sampling_frequency: Sample every N moves (default: 50)
    """
    the_dic: list[dict[str, fen]] = []

    count_game: int = 0
    total_count_move: int = 0
    recorded_board = 0

    with open(input_pgn_file_path, "r", encoding="utf-8") as pgn:
        while recorded_board < max_boards:

            count_game += 1
            game: chess.pgn.GameNode | None = chess.pgn.read_game(pgn)

            if count_game % 10_000 == 0:  # save
                recorded_board = save_intermediate_progress(
                    the_dic,
                    output_file_path,
                    count_game,
                    total_count_move,
                    max_boards,
                    total_games_in_file,
                    total_moves_in_file,
                    input_pgn_file_path,
                    sampling_frequency,
                )
            if game is None:
                print("GAME NONE")
                break
            else:
                total_count_move = process_game(
                    game, total_count_move, the_dic, sampling_frequency
                )

    # Final save with metadata
    save_final_dataset(
        the_dic,
        output_file_path,
        input_pgn_file_path,
        sampling_frequency,
        count_game,
        total_count_move,
        max_boards,
        total_games_in_file,
        total_moves_in_file,
    )


if __name__ == "__main__":
    # Use the centralized path configuration
    if not LICHESS_PGN_FILE.exists():
        print(f"PGN file not found at: {LICHESS_PGN_FILE}")
        print("Please run 'make lichess-pgn' to download the Lichess database first.")
        sys.exit(1)

    data_frame_file_name = EXTERNAL_DATA_DIR / "good_games2025"

    print(f"Starting dataset generation from {LICHESS_PGN_FILE}")
    print(f"Output will be saved to {data_frame_file_name}")

    # Optional: scan the file to get totals (can be skipped for very large files)
    # Set to False by default to avoid long scanning times
    do_initial_scan = False

    if do_initial_scan:
        print("\nStep 1: Scanning PGN file...")
        total_games, total_moves = scan_pgn_file(str(LICHESS_PGN_FILE))
        print("\nStep 2: Generating dataset...")
        generate_board_dataset(
            input_pgn_file_path=str(LICHESS_PGN_FILE),
            output_file_path=str(data_frame_file_name),
            total_games_in_file=total_games,
            total_moves_in_file=total_moves,
        )
    else:
        print(
            "\nSkipping initial PGN scan (can be enabled by setting do_initial_scan=True)"
        )
        print("Generating dataset without progress percentages...")
        generate_board_dataset(
            input_pgn_file_path=str(LICHESS_PGN_FILE),
            output_file_path=str(data_frame_file_name),
        )
