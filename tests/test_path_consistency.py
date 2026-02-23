"""
Tests to ensure that all paths defined in `chipiron.utils.path_variables` are absolute paths, which is important for consistent behavior across different working directories and environments.
"""

from pathlib import Path
from urllib.parse import urlparse

from chipiron.utils.path_variables import (
    EXTERNAL_DATA_DIR,
    GUI_DIR,
    LICHESS_PGN_DIR,
    LICHESS_PGN_FILE,
    ML_FLOW_URI_PATH,
    ML_FLOW_URI_PATH_TEST,
    STOCKFISH_BINARY_PATH,
    STOCKFISH_DIR,
    SYZYGY_TABLES_DIR,
)


def _is_sqlite_uri(uri: str) -> bool:
    return uri.startswith(("sqlite:///", "sqlite:////"))


def _sqlite_path_from_uri(uri: str) -> Path:
    """
    Convert sqlite:///... or sqlite:////... into a filesystem Path.

    - sqlite:////abs/path -> /abs/path
    - sqlite:///abs/path  -> /abs/path
    - sqlite:///rel/path  -> rel/path  (allowed but then not absolute)
    """
    parsed = urlparse(uri)

    # For sqlite URIs, the path is in parsed.path.
    # Example:
    #   sqlite:////home/x.db -> parsed.path == '//home/x.db'
    #   sqlite:///home/x.db  -> parsed.path == '/home/x.db'
    #   sqlite:///rel/x.db   -> parsed.path == '/rel/x.db' (still looks absolute)
    p = parsed.path

    # Normalize multiple leading slashes down to one.
    # Path('//home/x') is absolute on POSIX, but we normalize anyway.
    while p.startswith("//"):
        p = p[1:]

    return Path(p)


def test_external_paths_are_absolute() -> None:
    paths = [
        EXTERNAL_DATA_DIR,
        LICHESS_PGN_DIR,
        SYZYGY_TABLES_DIR,
        STOCKFISH_DIR,
        GUI_DIR,
        LICHESS_PGN_FILE,
        STOCKFISH_BINARY_PATH,
    ]
    for p in paths:
        assert Path(p).is_absolute()


def test_mlflow_uris_are_defined() -> None:
    assert isinstance(ML_FLOW_URI_PATH, str) and ML_FLOW_URI_PATH
    assert isinstance(ML_FLOW_URI_PATH_TEST, str) and ML_FLOW_URI_PATH_TEST


def test_mlflow_sqlite_paths_are_absolute_when_sqlite() -> None:
    for uri in [ML_FLOW_URI_PATH, ML_FLOW_URI_PATH_TEST]:
        if _is_sqlite_uri(uri):
            db_path = _sqlite_path_from_uri(uri)
            assert db_path.is_absolute()
