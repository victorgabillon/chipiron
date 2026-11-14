"""Optional caller-owned output allowance; ordinary writers retain their behavior."""

import io
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Buffer, Iterator


@dataclass
class _Allowance:
    root: Path
    remaining: int


_ALLOWANCE: ContextVar[_Allowance | None] = ContextVar("output_allowance", default=None)


def active(path: str | Path) -> bool:
    """Return whether this path belongs to the current private output reservation."""
    value = _ALLOWANCE.get()
    return value is not None and Path(path).absolute().is_relative_to(value.root)


@contextmanager
def output_allowance(root: Path, maximum: int | None) -> Iterator[None]:
    """Bound cumulative writes, including overwritten temporary files, without renewal."""
    if maximum is None:
        yield
        return
    if maximum <= 0 or _ALLOWANCE.get() is not None:
        message = "Invalid or nested output reservation."
        raise ValueError(message)
    token = _ALLOWANCE.set(_Allowance(root.absolute(), maximum))
    try:
        yield
    finally:
        _ALLOWANCE.reset(token)


class _Writer(io.BufferedIOBase):
    """Check capacity before each actual write, including serializer seeks."""

    def __init__(self, path: Path, allowance: _Allowance) -> None:
        super().__init__()
        self._file = path.open("wb", buffering=0)
        self._allowance = allowance

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def write(self, data: Buffer) -> int:
        count = memoryview(data).nbytes
        if count > self._allowance.remaining:
            message = (
                "Continuous resource_backpressure: private output allowance exhausted."
            )
            raise OSError(message)
        self._allowance.remaining -= count
        return self._file.write(data)

    def tell(self) -> int:
        return self._file.tell()

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._file.seek(offset, whence)

    def fileno(self) -> int:
        return self._file.fileno()

    def flush(self) -> None:
        if not self._file.closed:
            self._file.flush()

    def close(self) -> None:
        try:
            super().close()
        finally:
            self._file.close()


def output_file(
    path: str | Path, mode: str = "w", *, encoding: str | None = None
) -> Any:
    """Open a scoped output with its quota, or use the exact ordinary open path."""
    value = _ALLOWANCE.get()
    if value is None or not active(path):
        return open(path, mode, encoding=encoding)
    if mode not in {"w", "wb"}:
        message = "Unsupported scoped output mode."
        raise ValueError(message)
    writer = _Writer(Path(path), value)
    return (
        writer
        if mode == "wb"
        else io.TextIOWrapper(writer, encoding=encoding or "utf-8")
    )
