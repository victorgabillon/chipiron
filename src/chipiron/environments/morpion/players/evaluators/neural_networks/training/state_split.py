"""Stable held-out membership for evolving, accepted Morpion datasets."""

import hashlib
import json
import os
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from chipiron.environments.morpion.bootstrap.generation_publication_io import (
    file_digest,
)
from chipiron.environments.morpion.learning import (
    decode_morpion_state_ref_payload,
    iter_morpion_supervised_rows_from_path,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from atomheart.games.morpion.state import MorpionState as AtomMorpionState

    from chipiron.environments.morpion.learning import MorpionSupervisedRow

    from .args import MorpionTrainingArgs

POLICY = "canonical_state_sha256_v1"


def canonical_identity(row: MorpionSupervisedRow) -> str:
    """Use the game's rooted D4 identity, independent of node ID and replay order."""
    return _canonical_identity_from_state(
        decode_morpion_state_ref_payload(row.state_ref_payload)
    )


def _canonical_identity_from_state(state: AtomMorpionState) -> str:
    """Use a state already certified by the private validated-row producer."""
    payload = json.dumps(
        [state.variant.value, state.canonical_tag], separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class StateSplit:
    """Compact indices plus exact held-out population and target-contract evidence."""

    validation_indices: frozenset[int]
    row_count: int
    validation_population_sha256: str


type _SplitKey = tuple[str, str, float, int, int | None]


@dataclass(frozen=True)
class _SuppliedSplit:
    key: _SplitKey
    value: StateSplit


_SUPPLIED: ContextVar[_SuppliedSplit | None] = ContextVar(
    "prepared_split", default=None
)


@lru_cache(maxsize=2)
def _split(
    path: str, digest: str, fraction: float, seed: int, max_rows: int | None
) -> StateSplit:
    """Reuse one immutable input split across candidates without retaining states."""
    supplied = _SUPPLIED.get()
    if supplied is not None:
        if supplied.key != (
            str(Path(path).resolve()),
            digest,
            fraction,
            seed,
            max_rows,
        ):
            message = "Prepared split source or recipe changed"
            raise ValueError(message)
        return supplied.value
    # Digest remains in the original LRU key. The default producer is unchanged.
    return _split_identified_rows(
        _native_identified_rows(path, max_rows), fraction=fraction, seed=seed
    )


def _native_identified_rows(
    path: str, max_rows: int | None
) -> Iterator[tuple[MorpionSupervisedRow, str]]:
    """Preserve the native one-extra-validated-row boundary at a finite limit."""
    for index, row in enumerate(iter_morpion_supervised_rows_from_path(path)):
        if max_rows is not None and index >= max_rows:
            break
        yield row, canonical_identity(row)


def _split_identified_rows(
    rows: Iterable[tuple[MorpionSupervisedRow, str]], *, fraction: float, seed: int
) -> StateSplit:
    """Apply the original membership/population fold for both producers."""
    indices: set[int] = set()
    held_out: list[str] = []
    count = 0
    for index, (row, identity) in enumerate(rows):
        count += 1
        bucket = hashlib.sha256(f"{POLICY}:{seed}:{identity}".encode()).digest()
        if int.from_bytes(bucket, "big") < int(fraction * 2**256):
            indices.add(index)
            held_out.append(
                json.dumps(
                    [
                        identity,
                        row.target_value,
                        row.metadata.get("target_source"),
                        row.metadata.get("target_confidence"),
                    ],
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
    population = hashlib.sha256("\n".join(sorted(held_out)).encode()).hexdigest()
    return StateSplit(frozenset(indices), count, population)


def _uncached_value(
    path: str, digest: str, fraction: float, seed: int, max_rows: int | None
) -> tuple[frozenset[int], int, str]:
    """Adapt the unchanged algorithm to operational storage, without another loop."""
    result = _split.__wrapped__(path, digest, fraction, seed, max_rows)
    return (
        result.validation_indices,
        result.row_count,
        result.validation_population_sha256,
    )


def prepare_state_split(
    args: MorpionTrainingArgs, max_rows: int | None = None
) -> tuple[MorpionTrainingArgs, dict[str, object]]:
    """Bind every training/evaluation kernel to the same stable membership."""
    if not args.stable_validation_split:
        return args, {}
    # Operational opt-in only: never a training/config/manifest hyperparameter.
    cache_directory = os.environ.get("CHIPIRON_STATE_SPLIT_CACHE_DIR")
    if cache_directory is not None:
        from .state_split_cache import load_or_build

        if not cache_directory.strip():
            message = "CHIPIRON_STATE_SPLIT_CACHE_DIR must name a directory"
            raise ValueError(message)
        result = StateSplit(
            *load_or_build(
                Path(cache_directory),
                Path(args.dataset_file),
                args.validation_fraction,
                args.validation_seed,
                max_rows,
                POLICY,
                _uncached_value,
            )
        )
    else:
        result = _split(
            str(args.dataset_file),
            file_digest(Path(args.dataset_file)),
            args.validation_fraction,
            args.validation_seed,
            max_rows,
        )
    return replace(args, validation_indices=result.validation_indices), {
        "policy": POLICY,
        "seed": args.validation_seed,
        "fraction": args.validation_fraction,
        "row_count": result.row_count,
        "train_count": result.row_count - len(result.validation_indices),
        "validation_count": len(result.validation_indices),
        "validation_population_sha256": result.validation_population_sha256,
    }


def _install_prepared_split(
    args: MorpionTrainingArgs,
    max_rows: int | None,
    *,
    source_sha256: str,
    value: StateSplit,
) -> tuple[MorpionTrainingArgs, dict[str, object]]:
    """Seed the existing API/LRUs from a verified producer, with no second replay.

    Only the completed shared-preparation consumer calls this private seam. Its
    source/semantic/recipe checks precede this handoff. The temporary supplied
    result is context-local, bound again to the actual source and split recipe,
    and reset even if persistent publication fails. It is not a decoder cache.
    """
    if not args.stable_validation_split:
        message = "Prepared split requires stable validation"
        raise ValueError(message)
    if file_digest(Path(args.dataset_file)) != source_sha256:
        message = "Prepared split source changed"
        raise ValueError(message)
    supplied = _SuppliedSplit(
        (
            str(Path(args.dataset_file).resolve()),
            source_sha256,
            args.validation_fraction,
            args.validation_seed,
            max_rows,
        ),
        value,
    )
    token = _SUPPLIED.set(supplied)
    try:
        prepared, metadata = prepare_state_split(args, max_rows)
    finally:
        _SUPPLIED.reset(token)
    if (
        prepared.validation_indices != value.validation_indices
        or metadata["row_count"] != value.row_count
        or metadata["validation_population_sha256"]
        != value.validation_population_sha256
    ):
        message = "Prepared split disagrees with existing exact-content result"
        raise ValueError(message)
    return prepared, metadata
