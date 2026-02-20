"""Centralized path configuration for Chipiron.

This module defines paths for external inputs (user-provided) and runtime outputs.
Defaults are environment-overridable and safe for installed packages.
"""

import os
from pathlib import Path

from platformdirs import user_data_dir

APP_NAME = "chipiron"


def get_env_path(env_var: str, default: str) -> Path:
    """
    Get a path from environment variable, else default.

    - If default is absolute -> use it.
    - If default is relative -> interpret relative to current working directory.
      (Keeps dev ergonomics without relying on repo root.)
    """
    env_value = os.getenv(env_var, default)
    p = Path(env_value)
    return p if p.is_absolute() else (Path.cwd() / p)


# ---------- External data (user-provided / not packaged) ----------
EXTERNAL_DATA_DIR = get_env_path("EXTERNAL_DATA_DIR", "external_data")
LICHESS_PGN_DIR = get_env_path(
    "LICHESS_PGN_DIR", str(EXTERNAL_DATA_DIR / "lichess_pgn")
)
SYZYGY_TABLES_DIR = get_env_path(
    "SYZYGY_TABLES_DIR", str(EXTERNAL_DATA_DIR / "syzygy-tables")
)
STOCKFISH_DIR = get_env_path("STOCKFISH_DIR", str(EXTERNAL_DATA_DIR / "stockfish"))
GUI_DIR = get_env_path("GUI_DIR", str(EXTERNAL_DATA_DIR / "gui"))

LICHESS_PGN_FILE = get_env_path(
    "LICHESS_PGN_FILE",
    str(LICHESS_PGN_DIR / "lichess_db_standard_rated_2015-03.pgn"),
)

STOCKFISH_BINARY_PATH = get_env_path(
    "STOCKFISH_BINARY_PATH",
    str(STOCKFISH_DIR / "stockfish" / "stockfish-ubuntu-x86-64-avx2"),
)

# External puzzles (user-provided / not packaged)
PUZZLES_DIR = get_env_path("PUZZLES_DIR", str(EXTERNAL_DATA_DIR / "puzzles"))

MATE_IN_2_DB_SMALL = get_env_path(
    "MATE_IN_2_DB_SMALL",
    str(PUZZLES_DIR / "mate_in_2_db_small.pickle"),
)


# ---------- Runtime outputs (must be writable) ----------
def _runtime_data_dir() -> Path:
    """Return the persistent per-user writable data directory.

    The directory can be overridden by setting CHIPIRON_OUTPUT_DIR.
    """
    override = os.environ.get("CHIPIRON_OUTPUT_DIR")
    base = Path(override) if override else Path(user_data_dir(APP_NAME))
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_mlflow_db_path(*, filename: str = "mlruns.db") -> Path:
    """Return the path to the MLflow SQLite database.

    The location can be overridden by setting ML_FLOW_DB_PATH to an absolute path.
    """
    override = os.environ.get("ML_FLOW_DB_PATH")
    p = Path(override) if override else (_runtime_data_dir() / "mlflow" / filename)
    p = p.expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def get_mlflow_uri(
    *, filename: str = "mlruns.db", env_var: str = "ML_FLOW_URI_PATH"
) -> str:
    """Return the MLflow tracking URI.

    If `env_var` is set, return its value. Otherwise, use a SQLite database stored
    under the Chipiron runtime data directory.
    """
    override_uri = os.environ.get(env_var)
    if override_uri:
        return override_uri
    db_path = get_mlflow_db_path(filename=filename)
    return f"sqlite:///{db_path.as_posix()}"


ML_FLOW_URI_PATH = get_mlflow_uri(filename="mlruns.db", env_var="ML_FLOW_URI_PATH")
ML_FLOW_URI_PATH_TEST = get_mlflow_uri(
    filename="mlruns_test.db", env_var="ML_FLOW_URI_PATH_TEST"
)
