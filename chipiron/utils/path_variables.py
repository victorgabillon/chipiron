"""
Path utilities for the Chipiron project.

This module provides centralized path definitions to ensure consistent
path handling across the entire project.
"""

import os
from pathlib import Path

# Define the project root directory
# This file is at: chipiron/utils/path_variables.py
# So we go up 2 levels to reach project root
PROJECT_ROOT = Path(__file__).parents[2]  # Go up from chipiron/utils/ to project root

# Load .env file if it exists
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                if "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key, value)


def get_env_path(env_var: str, default: str) -> Path:
    """Get a path from environment variable or use default relative to PROJECT_ROOT."""
    env_value = os.getenv(env_var, default)
    if os.path.isabs(env_value):
        return Path(env_value)
    return PROJECT_ROOT / env_value


# Common directory paths
EXTERNAL_DATA_DIR = get_env_path("EXTERNAL_DATA_DIR", "external_data")
LICHESS_PGN_DIR = get_env_path("LICHESS_PGN_DIR", "external_data/lichess_pgn")
SYZYGY_TABLES_DIR = get_env_path("SYZYGY_TABLES_DIR", "external_data/syzygy-tables")
STOCKFISH_DIR = get_env_path("STOCKFISH_DIR", "external_data/stockfish")
GUI_DIR = get_env_path("GUI_DIR", "external_data/gui")
SCRIPTS_DIR = PROJECT_ROOT / "chipiron" / "scripts"

# Specific file paths
LICHESS_PGN_FILE = get_env_path(
    "LICHESS_PGN_FILE",
    "external_data/lichess_pgn/lichess_db_standard_rated_2015-03.pgn",
)
STOCKFISH_BINARY_PATH = get_env_path(
    "STOCKFISH_BINARY_PATH",
    "external_data/stockfish/stockfish/stockfish-ubuntu-x86-64-avx2",
)

# MLflow paths (use environment variables with defaults)
ML_FLOW_URI_PATH = os.getenv(
    "ML_FLOW_URI_PATH",
    f"sqlite:///{PROJECT_ROOT}/chipiron/scripts/default_output_folder/mlflow_data/mlruns.db",
)
ML_FLOW_URI_PATH_TEST = os.getenv(
    "ML_FLOW_URI_PATH_TEST",
    f"sqlite:///{PROJECT_ROOT}/chipiron/scripts/default_output_folder/mlflow_data/mlruns_test.db",
)


def main() -> None:
    """
    Print all defined paths in absolute form for debugging and verification.
    """
    print("Chipiron Project Paths (with .env integration):")
    print("=" * 50)
    print(f"PROJECT_ROOT:        {PROJECT_ROOT.absolute()}")
    print()
    print("Directory Paths:")
    print(f"EXTERNAL_DATA_DIR:   {EXTERNAL_DATA_DIR.absolute()}")
    print(f"LICHESS_PGN_DIR:     {LICHESS_PGN_DIR.absolute()}")
    print(f"SYZYGY_TABLES_DIR:   {SYZYGY_TABLES_DIR.absolute()}")
    print(f"STOCKFISH_DIR:       {STOCKFISH_DIR.absolute()}")
    print(f"GUI_DIR:             {GUI_DIR.absolute()}")
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
    print("Environment Variables Used:")
    env_vars = [
        "EXTERNAL_DATA_DIR",
        "LICHESS_PGN_DIR",
        "SYZYGY_TABLES_DIR",
        "STOCKFISH_DIR",
        "GUI_DIR",
        "LICHESS_PGN_FILE",
        "STOCKFISH_BINARY_PATH",
    ]
    for var in env_vars:
        env_value = os.getenv(var, "NOT SET")
        print(f"{var}: {env_value}")
    print()
    print("Path Existence Check:")
    print(f"PROJECT_ROOT exists:     {PROJECT_ROOT.exists()}")
    print(f"EXTERNAL_DATA_DIR exists: {EXTERNAL_DATA_DIR.exists()}")
    print(f"GUI_DIR exists:          {GUI_DIR.exists()}")
    print(f"LICHESS_PGN_FILE exists:  {LICHESS_PGN_FILE.exists()}")


if __name__ == "__main__":
    main()
