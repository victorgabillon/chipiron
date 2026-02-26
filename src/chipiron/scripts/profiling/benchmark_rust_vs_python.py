# pylint: disable=ungrouped-imports
"""Profiling script to compare performance of important functions with and without Rust implementations.

This script benchmarks:
- Move generation
- Base tree exploration
- Board operations
- Other core chess engine functions

During benchmarking, logging is suppressed to ERROR level to avoid noise in timing measurements.
Results are printed to console and saved to a timestamped file.
"""

import datetime
import os
import statistics
import sys
import time
import traceback
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from anemone.progress_monitor.progress_monitor import (
    TreeBranchLimitArgs,
)
from atomheart.games.chess.board.iboard import IBoard

from chipiron.players.move_selector.tree_and_value_args import TreeAndValueAppArgs
from chipiron.utils.logger import (
    chipiron_logger,
    suppress_all_logging,
    suppress_logging,
)

if TYPE_CHECKING:
    from atomheart.games.chess.move.imove import MoveKey

    from chipiron.environments.chess.players.args.chess_player_args import (
        ChessPlayerArgs,
    )


# Suppress PyTorch CUDA warnings for benchmarking
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import logging
    import random

    from atomheart.games.chess.board.factory import create_board
    from atomheart.games.chess.board.utils import FenPlusHistory

    from chipiron.environments.chess.players.evaluators.boardevaluators.table_base.factory import (
        create_syzygy,
    )
    from chipiron.environments.chess.players.factory.factory import create_player
    from chipiron.players.player_ids import PlayerConfigTag
    from chipiron.scripts.chipiron_args import ImplementationArgs

    # Configure parsley_coco logging to reduce noise during benchmarking
    try:
        from parsley.logger import set_verbosity

        set_verbosity(logging.WARNING)
    except ImportError:
        # parsley_coco might not be available in all environments
        pass
except ImportError as e:
    print(f"Error importing chipiron modules: {e}")
    print("Make sure you're running this from the correct directory")
    sys.exit(1)


class BenchmarkResult:
    """Container for benchmark results."""

    def __init__(self, name: str, rust_enabled: bool) -> None:
        """Initialize a new benchmark result.

        Args:
            name (str): The name of the benchmark.
            rust_enabled (bool): Whether the Rust implementation is enabled.

        """
        self.name = name
        self.rust_enabled = rust_enabled
        self.times: list[float] = []
        self.errors: list[str] = []

    def add_time(self, time_seconds: float) -> None:
        """Add a timing result to the benchmark.

        Args:
            time_seconds (float): The time taken for the benchmark in seconds.

        """
        self.times.append(time_seconds)

    def add_error(self, error: str) -> None:
        """Add an error message to the benchmark result.

        Args:
            error (str): The error message.

        """
        self.errors.append(error)

    @property
    def mean_time(self) -> float:
        """Calculate the mean time of the benchmark.

        Returns:
            float: The mean time of the benchmark.

        """
        return statistics.mean(self.times) if self.times else float("inf")

    @property
    def median_time(self) -> float:
        """Calculate the median time of the benchmark.

        Returns:
            float: The median time of the benchmark.

        """
        return statistics.median(self.times) if self.times else float("inf")

    @property
    def std_dev(self) -> float:
        """Calculate the standard deviation of the benchmark times.

        Returns:
            float: The standard deviation of the times.

        """
        return statistics.stdev(self.times) if len(self.times) > 1 else 0.0

    @property
    def success_rate(self) -> float:
        """Calculate the success rate of the benchmark.

        Returns:
            float: The success rate as a percentage.

        """
        total = len(self.times) + len(self.errors)
        return len(self.times) / total if total > 0 else 0.0


class ChessEngineBenchmark:
    """Benchmark suite for chess engine functions."""

    def benchmark_play_move(self, iterations: int | None = None) -> None:
        """Benchmark playing a move (applying a legal move to the board)."""
        print("\n🕹️ Benchmarking Play Move (Board Update)")
        if iterations is None:
            iterations = self.move_generation_iterations

        for rust_enabled in [False, True]:

            def play_move_for_positions(rust_enabled: bool = rust_enabled) -> None:
                """Play a move for each position in the test set.

                Args:
                    rust_enabled (bool, optional): Whether to use the Rust implementation. Defaults to rust_enabled.

                """
                for fen in self.test_positions:
                    board: IBoard = self.create_board_for_fen(fen, rust_enabled)
                    moves: list[MoveKey] = list(board.legal_moves.get_all())
                    if not moves:
                        continue
                    move = moves[0]  # Play the first legal move
                    board.play_move_key(move)

            result = self.benchmark_function(
                play_move_for_positions,
                "play_move",
                rust_enabled,
                iterations,
            )

            key = "play_move"
            if key not in self.results:
                self.results[key] = {}
            self.results[key][rust_enabled] = result

    def __init__(self) -> None:
        """Initialize the benchmark results."""
        self.results: dict[str, dict[bool, BenchmarkResult]] = {}
        self.test_positions = [
            # Starting position
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            # Mid-game position
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
            # Complex position
            "r2q1rk1/ppp2ppp/2n1bn2/2b1p3/3pP3/3P1NP1/PPP1NPB1/R1BQ1RK1 b - - 0 9",
            # Endgame position
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        ]
        self.move_generation_iterations = 500
        self.base_tree_exploration_iterations = 50
        self.base_tree_exploration_limits = [1000, 5000, 10000]

    def create_implementation_args(self, use_rust: bool) -> ImplementationArgs:
        """Create implementation args for the given configuration."""
        return ImplementationArgs(
            use_rust_boards=use_rust,
            use_board_modification=False,  # Keep consistent
        )

    def create_board_for_fen(self, fen: str, use_rust: bool) -> IBoard:
        """Create a board for the given FEN using proper factory method."""
        implementation_args = self.create_implementation_args(use_rust)
        board: IBoard = create_board(
            use_rust_boards=implementation_args.use_rust_boards,
            use_board_modification=implementation_args.use_board_modification,
            fen_with_history=FenPlusHistory(current_fen=fen, historical_moves=[]),
        )
        return board

    def benchmark_function(
        self,
        func: Callable[[], None],
        name: str,
        rust_enabled: bool,
        iterations: int = 10,
    ) -> BenchmarkResult:
        """Benchmark a single function."""
        result = BenchmarkResult(name, rust_enabled)

        print(f"  Benchmarking {name} ({'Rust' if rust_enabled else 'Python'})...")

        for i in range(iterations):
            try:
                # Suppress all logging during benchmarking to avoid noise in timing
                with suppress_all_logging(level=logging.WARNING):
                    start_time = time.perf_counter()
                    func()
                    end_time = time.perf_counter()
                    run_time = end_time - start_time
                    result.add_time(run_time)
                    print(f"    Run {i + 1}: {run_time:.6f} seconds")
            except (ImportError, AttributeError, ValueError) as e:
                result.add_error(str(e))
                print(f"    Error in iteration {i + 1}: {e}")

        return result

    def benchmark_move_generation(self, iterations: int | None = None) -> None:
        """Benchmark move generation for different positions."""
        print("\n🔍 Benchmarking Move Generation")
        if iterations is None:
            iterations = self.move_generation_iterations

        for rust_enabled in [False, True]:

            def generate_moves_for_positions(rust_enabled: bool = rust_enabled) -> None:
                """Generate legal moves for each position in the test set.

                Args:
                    rust_enabled (bool, optional): Whether to use the Rust implementation. Defaults to rust_enabled.

                """
                for fen in self.test_positions:
                    board = self.create_board_for_fen(fen, rust_enabled)
                    # Generate all legal moves
                    moves = list(board.legal_moves.get_all())
                    # Also test move validation
                    for i, move in enumerate(moves):
                        if i >= 5:  # Test first 5 moves only
                            break
                        # Just iterate through moves - board.is_legal_move may not exist
                        _ = str(move)

            result = self.benchmark_function(
                generate_moves_for_positions,
                "move_generation",
                rust_enabled,
                iterations,
            )

            key = "move_generation"
            if key not in self.results:
                self.results[key] = {}
            self.results[key][rust_enabled] = result

    def benchmark_base_tree_exploration(self, iterations: int | None = None) -> None:
        """Benchmark actual tree exploration using the BaseTreeExploration pattern."""
        print("\n🌲 Benchmarking Base Tree Exploration (Real Implementation)")
        if iterations is None:
            iterations = self.base_tree_exploration_iterations
        tree_branch_limits = self.base_tree_exploration_limits

        for tree_branch_limit in tree_branch_limits:
            print(f"\n  Testing with TreeMoveLimit: {tree_branch_limit}")

            for rust_enabled in [False, True]:

                def run_base_tree_exploration(
                    rust_enabled: bool = rust_enabled,
                    tree_branch_limit: int = tree_branch_limit,
                ) -> None:
                    """Run base tree exploration for a specific position.

                    Args:
                        rust_enabled (bool, optional): Whether to use the Rust implementation. Defaults to rust_enabled.
                        tree_branch_limit (int, optional): The maximum number of moves to explore. Defaults to tree_branch_limit.

                    """
                    # Create implementation args
                    implementation_args = self.create_implementation_args(rust_enabled)

                    # Create syzygy
                    syzygy = create_syzygy(use_rust=implementation_args.use_rust_boards)

                    # Create player args and set tree move limit
                    player_args: ChessPlayerArgs = (
                        PlayerConfigTag.UNIFORM.get_players_args()
                    )

                    # TODO: find a prettier way to do this
                    assert isinstance(
                        player_args.main_move_selector, TreeAndValueAppArgs
                    )
                    assert isinstance(
                        player_args.main_move_selector.anemone_args.stopping_criterion,
                        TreeBranchLimitArgs,
                    )
                    player_args.main_move_selector.anemone_args.stopping_criterion.tree_branch_limit = tree_branch_limit

                    # Create random generator
                    random_generator = random.Random()
                    random_generator.seed(42)  # Fixed seed for consistency

                    # Create player
                    player = create_player(
                        args=player_args,
                        syzygy=syzygy,
                        random_generator=random_generator,
                        implementation_args=implementation_args,
                        universal_behavior=True,
                    )

                    # Create board
                    board = create_board(
                        use_rust_boards=implementation_args.use_rust_boards,
                        use_board_modification=implementation_args.use_board_modification,
                    )

                    # Run tree exploration
                    player.select_move(
                        state_snapshot=board.into_fen_plus_history(),
                        seed=42,
                    )

                result = self.benchmark_function(
                    run_base_tree_exploration,
                    f"base_tree_exploration_{tree_branch_limit}",
                    rust_enabled,
                    iterations,
                )

                key = f"base_tree_exploration_{tree_branch_limit}"
                if key not in self.results:
                    self.results[key] = {}
                self.results[key][rust_enabled] = result

    def print_results(self) -> None:
        """Print benchmark results to console."""
        print("\n" + "=" * 80)
        print("🏁 BENCHMARK RESULTS")
        print("=" * 80)
        print("Benchmark iteration counts:")
        print(
            f"- Move generation: {self.move_generation_iterations} iterations per implementation (Python/Rust)"
        )
        print(
            f"- Base tree exploration: {self.base_tree_exploration_iterations} iterations per implementation, per tree move limit {self.base_tree_exploration_limits}"
        )

        for function_name, results in self.results.items():
            print(f"\n📊 {function_name.upper()}")
            print("-" * 60)

            python_result = results.get(False)
            rust_result = results.get(True)

            if python_result and rust_result:
                print(
                    f"{'Implementation':<15} {'Mean (ms)':<12} {'Median (ms)':<14} {'Std Dev':<10} {'Success Rate'}"
                )
                print("-" * 60)

                py_mean = python_result.mean_time * 1000
                py_median = python_result.median_time * 1000
                py_std = python_result.std_dev * 1000

                rust_mean = rust_result.mean_time * 1000
                rust_median = rust_result.median_time * 1000
                rust_std = rust_result.std_dev * 1000

                print(
                    f"{'Python':<15} {py_mean:<12.3f} {py_median:<14.3f} {py_std:<10.3f} {python_result.success_rate:<10.1%}"
                )
                print(
                    f"{'Rust':<15} {rust_mean:<12.3f} {rust_median:<14.3f} {rust_std:<10.3f} {rust_result.success_rate:<10.1%}"
                )

                if python_result.mean_time > 0:
                    speedup = python_result.mean_time / rust_result.mean_time
                    print(f"\n💨 Speedup: {speedup:.2f}x faster with Rust")
                    if speedup > 1:
                        improvement = (
                            (python_result.mean_time - rust_result.mean_time)
                            / python_result.mean_time
                        ) * 100
                        print(f"📈 Performance improvement: {improvement:.1f}%")
            else:
                if python_result:
                    py_mean = python_result.mean_time * 1000
                    print(
                        f"Python: {py_mean:.3f}ms (Success: {python_result.success_rate:.1%})"
                    )
                if rust_result:
                    rust_mean = rust_result.mean_time * 1000
                    print(
                        f"Rust: {rust_mean:.3f}ms (Success: {rust_result.success_rate:.1%})"
                    )

    def save_results(self, filename: str) -> None:
        """Save benchmark results to file."""
        report_dir = os.path.dirname(filename)
        if report_dir and not os.path.exists(report_dir):
            os.makedirs(report_dir, exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            # Write header
            timestamp = datetime.datetime.now()
            f.write("Chipiron Chess Engine Benchmark Results\n")
            f.write(f"Generated on: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'=' * 80}\n\n")

            # Write system info
            f.write("System Information:\n")
            f.write(f"Python version: {sys.version}\n")
            f.write(f"Platform: {sys.platform}\n")
            f.write(f"Test positions: {len(self.test_positions)}\n\n")
            f.write("Benchmark iteration counts:\n")
            f.write(
                f"- Move generation: {self.move_generation_iterations} iterations per implementation (Python/Rust)\n"
            )
            f.write(
                f"- Base tree exploration: {self.base_tree_exploration_iterations} iterations per implementation, per tree move limit {self.base_tree_exploration_limits}\n\n"
            )

            # Write detailed results
            for function_name, results in self.results.items():
                f.write(f"\n{function_name.upper()}\n")
                f.write("-" * 60 + "\n")

                python_result = results.get(False)
                rust_result = results.get(True)

                if python_result:
                    f.write("\nPython Implementation:\n")
                    f.write(f"  Mean time: {python_result.mean_time * 1000:.3f}ms\n")
                    f.write(
                        f"  Median time: {python_result.median_time * 1000:.3f}ms\n"
                    )
                    f.write(f"  Std deviation: {python_result.std_dev * 1000:.3f}ms\n")
                    f.write(f"  Success rate: {python_result.success_rate:.1%}\n")
                    f.write(f"  Successful runs: {len(python_result.times)}\n")
                    f.write(f"  Errors: {len(python_result.errors)}\n")

                if rust_result:
                    f.write("\nRust Implementation:\n")
                    f.write(f"  Mean time: {rust_result.mean_time * 1000:.3f}ms\n")
                    f.write(f"  Median time: {rust_result.median_time * 1000:.3f}ms\n")
                    f.write(f"  Std deviation: {rust_result.std_dev * 1000:.3f}ms\n")
                    f.write(f"  Success rate: {rust_result.success_rate:.1%}\n")
                    f.write(f"  Successful runs: {len(rust_result.times)}\n")
                    f.write(f"  Errors: {len(rust_result.errors)}\n")

                if python_result and rust_result and python_result.mean_time > 0:
                    speedup = python_result.mean_time / rust_result.mean_time
                    f.write("\nPerformance Comparison:\n")
                    f.write(f"  Speedup: {speedup:.2f}x\n")
                    improvement = (
                        (python_result.mean_time - rust_result.mean_time)
                        / python_result.mean_time
                    ) * 100
                    f.write(f"  Improvement: {improvement:.1f}%\n")

                f.write("\n")

    def run_full_benchmark(self) -> None:
        """Run the complete benchmark suite."""
        print("🚀 Starting Chipiron Performance Benchmark")
        print(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Run benchmarks
        try:
            # Suppress logging during benchmarking to avoid noise in timing
            with suppress_logging(chipiron_logger, level=logging.ERROR):
                self.benchmark_move_generation()
                self.benchmark_base_tree_exploration()
                self.benchmark_play_move()
        except (ImportError, AttributeError, ValueError) as e:
            print(f"Error during benchmarking: {e}")

            traceback.print_exc()

        # Display results
        self.print_results()

        # Save results in dedicated folder
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = "package://scripts/profiling/benchmark_reports"
        filename = os.path.join(report_dir, f"benchmark_results_{timestamp}.txt")
        self.save_results(filename)
        print(f"\n💾 Results saved to: {filename}")


def main() -> None:
    """Run the main entry point."""
    benchmark = ChessEngineBenchmark()
    benchmark.run_full_benchmark()


if __name__ == "__main__":
    main()
