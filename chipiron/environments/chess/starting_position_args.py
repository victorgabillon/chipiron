"""Module for starting position args."""

from dataclasses import dataclass
from enum import StrEnum
from importlib import resources
from pathlib import Path
from typing import Literal

from valanga import StateTag

from chipiron.environments.chess.tags import ChessStartTag
from chipiron.environments.starting_position import StartingPositionArgs


class StartingPositionArgsType(StrEnum):
    """Startingpositionargstype implementation."""

    FEN = "fen"
    FROM_FILE = "from_file"


@dataclass(frozen=True)
class FenStartingPositionArgs(StartingPositionArgs):
    """Fenstartingpositionargs implementation."""

    type: Literal[StartingPositionArgsType.FEN] = StartingPositionArgsType.FEN
    fen: str = ""

    def get_start_tag(self) -> StateTag:
        """Return start tag."""
        if not self.fen:
            raise ValueError("Empty fen in FenStartingPositionArgs")
        return ChessStartTag(fen=self.fen)


@dataclass(frozen=True)
class FileStartingPositionArgs(StartingPositionArgs):
    """Filestartingpositionargs implementation."""

    type: Literal[StartingPositionArgsType.FROM_FILE] = (
        StartingPositionArgsType.FROM_FILE
    )
    file_name: str = ""

    def get_start_tag(self) -> StateTag:
        """Return start tag."""
        if not self.file_name:
            raise ValueError("Empty file_name in FileStartingPositionArgs")
        fen = _load_fen_from_file(self.file_name)
        return ChessStartTag(fen=fen)


AllStartingPositionArgs = FenStartingPositionArgs | FileStartingPositionArgs


def _load_fen_from_file(file_name: str) -> str:
    """Load a FEN string from a file or packaged resource."""
    path = Path(file_name)
    if not path.is_file():
        path = _resolve_starting_board_path(file_name)
    fen = _load_fen_from_path(path)
    if not fen:
        raise ValueError(f"Starting position file is empty: {path}")
    return fen


def _resolve_starting_board_path(file_name: str) -> Path:
    """Resolve a packaged starting-board path by filename."""
    try:
        starting_boards = resources.files("chipiron.data").joinpath("starting_boards")
    except ModuleNotFoundError as exc:
        raise FileNotFoundError(
            f"Starting position file not found: {file_name}"
        ) from exc
    resource_path = starting_boards.joinpath(file_name)
    if not resource_path.is_file():
        raise FileNotFoundError(f"Starting position file not found: {file_name}")
    with resources.as_file(resource_path) as resolved:
        return resolved


def _load_fen_from_path(path: Path) -> str:
    """Load and normalize FEN content from a path."""
    contents = path.read_text(encoding="utf-8").strip()
    if not contents:
        return ""
    if _looks_like_fen(contents):
        return contents
    lines = [line.strip() for line in contents.splitlines() if line.strip()]
    if len(lines) < 9:
        raise ValueError(f"Invalid starting position file: {path}")
    board_lines = lines[:8]
    suffix = " ".join(lines[8:])
    board_fen = "/".join(_compress_rank(line) for line in board_lines)
    return f"{board_fen} {suffix}".strip()


def _looks_like_fen(contents: str) -> bool:
    """Return whether the contents resemble a FEN header."""
    parts = contents.split()
    if len(parts) < 6:
        return False
    board = parts[0]
    return board.count("/") == 7


def _compress_rank(rank: str) -> str:
    """Compress a board rank into FEN digit form."""
    if len(rank) != 8:
        raise ValueError(f"Invalid board rank: {rank!r}")
    result: list[str] = []
    empty_count = 0
    for char in rank:
        if char in {"1", ".", "0", "-"}:
            empty_count += 1
        else:
            if empty_count:
                result.append(str(empty_count))
                empty_count = 0
            result.append(char)
    if empty_count:
        result.append(str(empty_count))
    return "".join(result)
