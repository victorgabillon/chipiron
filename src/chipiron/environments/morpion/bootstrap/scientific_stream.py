"""One lossless encoding beneath the existing scientific JSON parser.

jsonl-zstd-1 is one checksummed frame: level 3, 1 MiB window, no dictionary,
content-size field or threads. Block boundaries bound decompressobj output to
128 KiB (Zstandard specification); libzstd owns all actual decompression and
checksum validation. Framing here also proves EOF, unlike stream_reader alone.
"""

# ruff: noqa: TRY003

from __future__ import annotations

import hashlib
import io
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import zstandard as zstd

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from typing import BinaryIO

ENCODING = "jsonl-zstd-1"
BLOCK_BYTES = 128 * 1024
# Magic, checksum flag (no size/dictionary), window descriptor (log2=20).
FRAME_HEADER = bytes.fromhex("28b52ffd0450")


class ScientificRowsWriter:
    """Serialize neither schema nor rows; hash and optionally compress caller bytes."""

    def __init__(self, stream: BinaryIO, encoding: str | None) -> None:
        """Bind a private output stream to the single explicit encoding."""
        if encoding not in (None, ENCODING):
            raise ValueError("Unsupported scientific encoding")
        self.stream = stream
        self.encoding = encoding
        self.compressor = (
            zstd.ZstdCompressor(
                compression_params=zstd.ZstdCompressionParameters.from_level(
                    3,
                    window_log=20,
                    write_checksum=1,
                    write_content_size=0,
                    write_dict_id=0,
                    threads=0,
                )
            ).compressobj()
            if encoding
            else None
        )
        self.digest = hashlib.sha256()
        self.decoded_digest = hashlib.sha256()
        self.size = self.decoded_size = self.lines = 0

    def _emit(self, data: bytes) -> None:
        self.stream.write(data)
        self.digest.update(data)
        self.size += len(data)

    def write(self, data: bytes) -> None:
        """Consume one existing JSONL row; compression has bounded input slices."""
        self.decoded_digest.update(data)
        self.decoded_size += len(data)
        self.lines += data.count(b"\n")
        for start in range(0, len(data), BLOCK_BYTES):
            part = data[start : start + BLOCK_BYTES]
            self._emit(self.compressor.compress(part) if self.compressor else part)

    def finish(self, path: str, generation: int, count: int) -> dict[str, Any]:
        """Finish the single frame before exposing its descriptor to publication."""
        if self.lines != count + 1:
            raise ValueError("Scientific JSONL row count mismatch")
        if self.compressor:
            self._emit(self.compressor.flush())
        descriptor = {
            "path": path,
            "generation": generation,
            "count": count,
            "sha256": self.digest.hexdigest(),
        }
        if self.encoding:
            descriptor.update(
                encoding=self.encoding,
                bytes=self.size,
                decoded_bytes=self.decoded_size,
                decoded_sha256=self.decoded_digest.hexdigest(),
            )
        return descriptor


def _decoded_blocks(stream: BinaryIO, descriptor: dict[str, Any]) -> Iterator[bytes]:
    """Check one complete frame with bounded input/output and incremental hashes."""
    compressed_hash, decoded_hash = hashlib.sha256(), hashlib.sha256()
    compressed_size = decoded_size = lines = 0
    last_byte = b""

    def take(size: int) -> bytes:
        nonlocal compressed_size
        data = stream.read(size)
        if len(data) != size:
            raise ValueError("Truncated scientific compression frame")
        compressed_hash.update(data)
        compressed_size += size
        return data

    try:
        header = take(len(FRAME_HEADER))
        if header != FRAME_HEADER:
            raise ValueError("Unsupported scientific zstd frame parameters")
        decoder = zstd.ZstdDecompressor(max_window_size=1024 * 1024).decompressobj()
        if decoder.decompress(header):
            raise ValueError("Unexpected scientific frame header output")
        while True:
            block_header = take(3)
            value = int.from_bytes(block_header, "little")
            kind, size = (value >> 1) & 3, value >> 3
            if kind == 3 or size > BLOCK_BYTES:
                raise ValueError("Invalid scientific zstd block")
            data = decoder.decompress(block_header + take(1 if kind == 1 else size))
            decoded_size += len(data)
            if len(data) > BLOCK_BYTES or decoded_size > descriptor["decoded_bytes"]:
                raise ValueError("Scientific decoded byte limit exceeded")
            decoded_hash.update(data)
            lines += data.count(b"\n")
            if data:
                last_byte = data[-1:]
                yield data
            if value & 1:
                break
        if decoder.decompress(take(4)) or not decoder.eof or decoder.unused_data:
            raise ValueError("Incomplete scientific compression frame")
        if stream.read(1):
            raise ValueError("Trailing scientific compressed bytes or frames")
    except zstd.ZstdError as exc:
        raise ValueError("Invalid scientific compression frame") from exc
    if (
        compressed_size,
        compressed_hash.hexdigest(),
        decoded_size,
        decoded_hash.hexdigest(),
        lines,
        last_byte,
    ) != (
        descriptor["bytes"],
        descriptor["sha256"],
        descriptor["decoded_bytes"],
        descriptor["decoded_sha256"],
        descriptor["count"] + 1,
        b"\n",
    ):
        raise ValueError("Scientific compressed/decoded identity or row count mismatch")


class _BlockReader(io.RawIOBase):
    """Adapt bounded library-decoded blocks to the existing file-based parser."""

    def __init__(self, blocks: Iterator[bytes]) -> None:
        self.blocks = blocks
        self.pending = memoryview(b"")

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        while not self.pending:
            self.pending = memoryview(next(self.blocks, b""))
            if not self.pending:
                return 0
        count = min(len(buffer), len(self.pending))
        buffer[:count] = self.pending[:count]
        self.pending = self.pending[count:]
        return count


@contextmanager
def open_scientific_stream(
    path: Path, descriptor: dict[str, Any] | None = None
) -> Iterator[BinaryIO]:
    """Open declared bytes, never infer encoding or materialize decoded files."""
    with path.open("rb") as source:
        if descriptor is None or "encoding" not in descriptor:
            yield source
        elif descriptor["encoding"] == ENCODING:
            with io.BufferedReader(
                _BlockReader(_decoded_blocks(source, descriptor))
            ) as decoded:
                yield decoded
        else:
            raise ValueError("Unsupported scientific encoding")
