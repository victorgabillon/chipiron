"""Bounded metadata reader shared by lifecycle previews and immutable fork origins."""

import hashlib
import json
from itertools import islice
from typing import TYPE_CHECKING, Any, cast

from .generation_publication_io import require

if TYPE_CHECKING:
    from pathlib import Path


class Metadata:
    """Limit the whole preview, including source fingerprints and metadata joins."""

    def __init__(self) -> None:
        """Initialize the per-preview byte allowance."""
        self.remaining = 64 * 2**20

    def data(self, path: Path, maximum: int = 4 * 2**20) -> bytes:
        """Refuse symlinks/oversize inputs before reading bounded bytes."""
        require(path.is_file() and not path.is_symlink(), f"Missing metadata: {path}")
        size = path.stat().st_size
        require(
            size <= min(maximum, self.remaining), "Preview metadata budget exceeded."
        )
        with path.open("rb") as stream:
            data = stream.read(min(maximum, self.remaining) + 1)
        require(len(data) == size, "Metadata changed during preview.")
        self.remaining -= len(data)
        return data

    def read(self, path: Path) -> dict[str, Any]:
        """Read finite object metadata only."""
        value = json.loads(self.data(path), parse_constant=_invalid_constant)
        require(isinstance(value, dict), f"Expected metadata object: {path}")
        return cast("dict[str, Any]", value)

    def digest(self, path: Path) -> str:
        """Hash small metadata, never scientific payloads."""
        return hashlib.sha256(self.data(path)).hexdigest()


def _invalid_constant(value: str) -> None:
    require(False, f"Nonfinite JSON metadata: {value}")


def entries(path: Path, maximum: int = 2048) -> list[Path]:
    """Bound a metadata directory listing before traversal."""
    values = list(islice(path.iterdir(), maximum + 1))
    require(len(values) <= maximum, "Preview directory bound exceeded.")
    return values
