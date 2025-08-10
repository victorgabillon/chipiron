"""
Path utilities for the Chipiron project.

This module provides centralized path definitions to ensure consistent
path handling across the entire project.
"""

from pathlib import Path

# Define the project root directory
# This file is at: chipiron/utils/path_variables.py
# So we go up 2 levels to reach project root
PROJECT_ROOT = Path(__file__).parents[2]  # Go up from chipiron/utils/ to project root

# Common directory paths
EXTERNAL_DATA_DIR = PROJECT_ROOT / "external_data"
LICHESS_PGN_DIR = EXTERNAL_DATA_DIR / "lichess_pgn"
SYZYGY_TABLES_DIR = EXTERNAL_DATA_DIR / "syzygy-tables"
STOCKFISH_DIR = EXTERNAL_DATA_DIR / "stockfish"
SCRIPTS_DIR = PROJECT_ROOT / "chipiron" / "scripts"

# Specific file paths
LICHESS_PGN_FILE = LICHESS_PGN_DIR / "lichess_db_standard_rated_2015-03.pgn"
STOCKFISH_BINARY_PATH = STOCKFISH_DIR / "stockfish" / "stockfish-ubuntu-x86-64-avx2"

# MLflow paths (keep existing ones but make them relative to project root)
ML_FLOW_URI_PATH = f"sqlite:///{PROJECT_ROOT}/chipiron/scripts/default_output_folder/mlflow_data/mlruns.db"
ML_FLOW_URI_PATH_TEST = f"sqlite:///{PROJECT_ROOT}/chipiron/scripts/default_output_folder/mlflow_data/mlruns_test.db"


def main() -> None:
    """
    Print all defined paths in absolute form for debugging and verification.
    """
    print("Chipiron Project Paths:")
    print("=" * 50)
    print(f"PROJECT_ROOT:        {PROJECT_ROOT.absolute()}")
    print()
    print("Directory Paths:")
    print(f"EXTERNAL_DATA_DIR:   {EXTERNAL_DATA_DIR.absolute()}")
    print(f"LICHESS_PGN_DIR:     {LICHESS_PGN_DIR.absolute()}")
    print(f"SYZYGY_TABLES_DIR:   {SYZYGY_TABLES_DIR.absolute()}")
    print(f"STOCKFISH_DIR:       {STOCKFISH_DIR.absolute()}")
    print(f"SCRIPTS_DIR:         {SCRIPTS_DIR.absolute()}")
    print()
    print("File Paths:")
    print(f"LICHESS_PGN_FILE:    {LICHESS_PGN_FILE.absolute()}")
    print(f"STOCKFISH_BINARY_PATH: {STOCKFISH_BINARY_PATH.absolute()}")
    print()
    print("MLflow Paths:")
    print(f"ML_FLOW_URI_PATH:    {ML_FLOW_URI_PATH}")
    print(f"ML_FLOW_URI_PATH_TEST: {ML_FLOW_URI_PATH_TEST}")
    print()
    print("Path Existence Check:")
    print(f"PROJECT_ROOT exists:     {PROJECT_ROOT.exists()}")
    print(f"EXTERNAL_DATA_DIR exists: {EXTERNAL_DATA_DIR.exists()}")
    print(f"LICHESS_PGN_FILE exists:  {LICHESS_PGN_FILE.exists()}")


if __name__ == "__main__":
    main()
